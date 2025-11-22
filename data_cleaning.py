import pandas as pd
import numpy as np
import os
import gc # Thư viện quản lý bộ nhớ (Garbage Collector)
import glob # Để tìm kiếm files tự động
import time # Để đo thời gian xử lý

# --- 1. HÀM TÍNH KHOẢNG CÁCH HAVERSINE (meters) ---
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Tính khoảng cách Haversine giữa hai điểm tọa độ (Lng, Lat) 
    trên một mảng Numpy (vectorized) và trả về kết quả bằng mét.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km * 1000 # Trả về mét

# =========================================================================
# PHA 1: LÀM SẠCH BAN ĐẦU (SORT, TÍNH SPEED, CẮT ĐẦU ĐUÔI)
# =========================================================================

def process_one_file(file_path, output_dir):
    """
    PHA 1: Sắp xếp, tính toán tốc độ, làm sạch dữ liệu ngoài giờ và nhiễu đầu/cuối.
    Lưu kết quả ra file _final_clean.csv.
    """
    file_name = os.path.basename(file_path)
    
    # A. Đọc File
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"    ❌ Lỗi đọc file {file_name}: {e}")
        return None

    if 'anonymized_driver' in df.columns:
        df = df.drop(columns=['anonymized_driver'])
    
    # B. Sắp xếp (Sửa lỗi thời gian lộn xộn)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by=['anonymized_vehicle', 'datetime']).reset_index(drop=True)

    # C. Tính Speed
    df['prev_lat'] = df.groupby('anonymized_vehicle')['lat'].shift(1)
    df['prev_lng'] = df.groupby('anonymized_vehicle')['lng'].shift(1)
    df['prev_time'] = df.groupby('anonymized_vehicle')['datetime'].shift(1)

    dist_meters = haversine_np(df['prev_lng'], df['prev_lat'], df['lng'], df['lat'])
    time_diff_seconds = (df['datetime'] - df['prev_time']).dt.total_seconds()

    with np.errstate(divide='ignore', invalid='ignore'):
        gps_speed = (dist_meters / time_diff_seconds) * 3.6
    
    gps_speed = gps_speed.replace([np.inf, -np.inf], np.nan)
    df['gps_speed_calculated'] = gps_speed

    mask_null = df['speed'].isnull()
    df.loc[mask_null, 'speed'] = df.loc[mask_null, 'gps_speed_calculated'] 
    mask_not_null = ~mask_null & df['gps_speed_calculated'].notnull()
    df.loc[mask_not_null, 'speed'] = (df.loc[mask_not_null, 'speed'] + df.loc[mask_not_null, 'gps_speed_calculated']) / 2 
    
    # D. Xóa dữ liệu ngoài giờ 23h - 4h
    df['hour'] = df['datetime'].dt.hour
    df = df[~((df['hour'] >= 23) | (df['hour'] < 4))].copy()

    # E. Smart Trim (Cắt đầu đuôi nhiễu)
    if not df.empty:
        df['is_moving'] = df['speed'] > 3.0 
        
        grouper = df.groupby('anonymized_vehicle')['is_moving']
        cumsum_fwd = grouper.cumsum()
        cumsum_bwd = df.groupby('anonymized_vehicle')['is_moving'].transform(lambda x: x[::-1].cumsum()[::-1])
        
        mask_core = (cumsum_fwd > 0) & (cumsum_bwd > 0)

        df['mask_core_temp'] = mask_core 
        mask_start_buffer = df.groupby('anonymized_vehicle')['mask_core_temp'].shift(-1).fillna(False)
        mask_end_buffer = df.groupby('anonymized_vehicle')['mask_core_temp'].shift(1).fillna(False)

        final_mask = mask_core | mask_start_buffer | mask_end_buffer
        df = df[final_mask].copy()

    # F. Lưu file output
    cols_to_drop = ['prev_lat', 'prev_lng', 'prev_time', 'gps_speed_calculated', 
                    'hour', 'is_moving', 'mask_core_temp']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    output_name = file_name.replace('_raw', '_final_clean')
    output_path = os.path.join(output_dir, output_name)
    
    df.to_csv(output_path, index=False)
    print(f"    ✅ PHA 1 Xong! Đã tạo file làm sạch: {output_name} ({len(df)} bản ghi)")
    
    row_count = len(df)
    del df
    gc.collect()
    return row_count

# =========================================================================
# PHA 2: NÉN DỮ LIỆU TĨNH VÀ LƯU ĐÈ (Compress and Overwrite)
# =========================================================================

def compress_and_overwrite(file_path):
    """
    PHA 2: Đọc file _final_clean, áp dụng nén tĩnh (theo tọa độ và trạng thái cửa), 
    và lưu đè lên chính file đó để rút gọn data.
    """
    file_name = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
        # Chuyển đổi cột datetime (cần để tính toán)
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        print(f"    ❌ Lỗi đọc file {file_name}: {e}")
        return 

    # --- BƯỚC NÉN DỮ LIỆU TĨNH ---
    # 1. Tạo chữ ký nén (Bao gồm tọa độ và trạng thái cửa)
    df['compression_signature'] = (
        df['lng'].round(5).astype(str) + '_' + 
        df['lat'].round(5).astype(str) + '_' +
        df['door_up'].astype(str) + '_' + 
        df['door_down'].astype(str)
    )
    
    # 2. Xác định bản ghi cần giữ
    mask_start_of_vehicle = df['anonymized_vehicle'].shift(1) != df['anonymized_vehicle']
    mask_change = (
        df.groupby('anonymized_vehicle')['compression_signature'].shift(1) != df['compression_signature']
    )
    mask_keep = mask_change | mask_start_of_vehicle
    
    # 3. Áp dụng mask nén
    initial_rows = len(df)
    df = df[mask_keep].copy()
    
    # Xóa cột phụ và reset index
    df.drop(columns=['compression_signature'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 4. Lưu đè lên file gốc
    df.to_csv(file_path, index=False)

    print(f"    ✅ PHA 2 Xong! Đã nén và lưu đè. Giảm từ {initial_rows} bản ghi xuống còn {len(df)} bản ghi.")
    
    del df
    gc.collect()

# =========================================================================
# CHƯƠNG TRÌNH CHÍNH (ĐIỀU PHỐI HAI PHA XỬ LÝ)
# =========================================================================

def main_full_process():
    
    # !!! CẬP NHẬT ĐƯỜNG DẪN NÀY ĐỂ TRỎ ĐÚNG ĐẾN THƯ MỤC 'raw_GPS' CỦA BẠN !!!
    RAW_GPS_FOLDER = r"D:\HCMUT-workplace\BDC_Hackathon\raw_GPS"
    
    if not os.path.isdir(RAW_GPS_FOLDER):
        print(f"❌ LỖI: Không tìm thấy thư mục GPS tại đường dẫn: {RAW_GPS_FOLDER}")
        return

    # --- PHA 1: LÀM SẠCH BAN ĐẦU ---
    print("\n" + "="*80)
    print("PHA 1: BẮT ĐẦU LÀM SẠCH BAN ĐẦU (SORT, SPEED, TRIM)")
    print("="*80)
    
    start_time_1 = time.time()
    search_raw = os.path.join(RAW_GPS_FOLDER, 'anonymized_raw_2025-04-*.csv')
    all_raw_files = sorted(glob.glob(search_raw))
    
    if not all_raw_files:
        print(f"⚠️ Không tìm thấy file 'raw' nào. Kiểm tra lại đường dẫn và tên file.")
        return

    for file_path in all_raw_files:
        process_one_file(file_path, RAW_GPS_FOLDER)

    end_time_1 = time.time()
    print(f"\n🎉 HOÀN THÀNH PHA 1. Tổng thời gian: {end_time_1 - start_time_1:.2f} giây.")

    # --- PHA 2: NÉN DỮ LIỆU TĨNH VÀ LƯU ĐÈ ---
    print("\n" + "="*80)
    print("PHA 2: BẮT ĐẦU NÉN DỮ LIỆU TĨNH (RÚT GỌN FILES ĐÃ LÀM SẠCH)")
    print("="*80)

    start_time_2 = time.time()
    search_clean = os.path.join(RAW_GPS_FOLDER, 'anonymized_final_clean_2025-04-*.csv')
    all_clean_files = sorted(glob.glob(search_clean))

    if not all_clean_files:
        print(f"⚠️ Không tìm thấy file 'final_clean' nào để nén. Đã dừng lại.")
        return
    
    for file_path in all_clean_files:
        compress_and_overwrite(file_path)

    end_time_2 = time.time()

    print("\n" + "="*80)
    print(f"🎉 HOÀN TẤT TOÀN BỘ XỬ LÝ! Đã xử lý {len(all_clean_files)} files.")
    print(f"Tổng thời gian PHA 1: {end_time_1 - start_time_1:.2f} giây.")
    print(f"Tổng thời gian PHA 2: {end_time_2 - start_time_2:.2f} giây.")
    print("="*80)
    print("Dữ liệu đã được làm sạch và rút gọn tối đa, sẵn sàng cho phân tích Insight.")

if __name__ == "__main__":
    main_full_process()