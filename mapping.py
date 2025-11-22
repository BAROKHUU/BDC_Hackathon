import os
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
ROUTE_DIR = r"D:\HCMUT-workplace\BDC_Hackathon\HCMC_bus_routes"
GPS_DIR = r"D:\HCMUT-workplace\BDC_Hackathon\processed_GPS"
OUTPUT_FILE = "Master_Vehicle_Route_Mapping.csv"

# --- 1. HÀM TẠO KHUNG TUYẾN (SKELETON) ---
def build_route_skeletons(route_dir):
    print("--- Đang học lộ trình của 30 tuyến xe... ---")
    route_shapes = {} # Dictionary lưu trữ hình dạng tuyến
    
    subfolders = [f.path for f in os.scandir(route_dir) if f.is_dir()]
    
    for folder in subfolders:
        try:
            # Lấy ID tuyến (ưu tiên lấy từ route_by_id, nếu lỗi lấy tên folder)
            folder_name = os.path.basename(folder)
            route_id = folder_name
            
            # Đọc file route_by_id để lấy tên chính xác (ví dụ: "01", "152")
            route_info_path = os.path.join(folder, 'route_by_id.csv')
            if os.path.exists(route_info_path):
                df_info = pd.read_csv(route_info_path)
                if not df_info.empty:
                    route_id = str(df_info.iloc[0]['RouteNo'])
            
            # Gom trạm đi và về để tạo thành 1 hình dáng tổng thể cho tuyến đó
            # (Lý do: Để định danh xe thuộc tuyến nào, ta chỉ cần biết nó nằm trên trục đường đó là đủ)
            stops_files = ['stops_by_var.csv', 'rev_stops_by_var.csv']
            all_points = []
            
            for f_name in stops_files:
                f_path = os.path.join(folder, f_name)
                if os.path.exists(f_path):
                    df = pd.read_csv(f_path)
                    if 'Lng' in df.columns and 'Lat' in df.columns:
                        all_points.extend(list(zip(df['Lng'], df['Lat'])))
            
            if len(all_points) > 1:
                # Tạo đường LineString từ các điểm dừng
                route_shapes[route_id] = LineString(all_points)
                
        except Exception as e:
            print(f"Lỗi khi đọc tuyến {folder}: {e}")
            
    print(f"Đã học xong {len(route_shapes)} tuyến đường.\n")
    return route_shapes

# --- 2. HÀM ĐỊNH DANH XE (MATCHING) ---
def identify_vehicles_in_file(file_path, route_shapes):
    try:
        # Đọc file GPS
        df_gps = pd.read_csv(file_path, usecols=['lng', 'lat', 'anonymized_vehicle'])
        
        # === [FIX 1] LỌC DỮ LIỆU RÁC ===
        # Loại bỏ ngay các dòng thiếu tọa độ hoặc thiếu ID xe
        original_len = len(df_gps)
        df_gps = df_gps.dropna(subset=['lng', 'lat', 'anonymized_vehicle'])
        
        # Chuyển đổi cột tọa độ sang numeric để tránh lỗi chuỗi (nếu có)
        df_gps['lng'] = pd.to_numeric(df_gps['lng'], errors='coerce')
        df_gps['lat'] = pd.to_numeric(df_gps['lat'], errors='coerce')
        df_gps = df_gps.dropna(subset=['lng', 'lat']) # Lọc lần 2 sau khi convert
        
        if len(df_gps) == 0:
            print(f"File {os.path.basename(file_path)} không có dữ liệu hợp lệ.")
            return pd.DataFrame()

        unique_vehicles = df_gps['anonymized_vehicle'].unique()
        results = []
        
        filename = os.path.basename(file_path)
        print(f"Đang xử lý file: {filename} - Tìm thấy {len(unique_vehicles)} xe.")
        
        for veh_id in unique_vehicles:
            # Lấy dữ liệu của xe đó
            veh_data = df_gps[df_gps['anonymized_vehicle'] == veh_id]
            
            # Nếu xe này không có dòng dữ liệu nào (hiếm gặp nhưng an toàn) thì bỏ qua
            if veh_data.empty:
                continue

            # Tối ưu: Lấy mẫu 50 điểm
            if len(veh_data) > 50:
                sample = veh_data.sample(50)
            else:
                sample = veh_data
            
            # Tạo các điểm Point
            sample_points = [Point(xy) for xy in zip(sample['lng'], sample['lat'])]
            
            # === [FIX 2] KIỂM TRA DANH SÁCH RỖNG ===
            if not sample_points:
                continue

            best_route = "Unknown"
            min_score = float('inf')
            
            # So khớp với 30 tuyến mẫu
            for r_id, r_shape in route_shapes.items():
                distances = [pt.distance(r_shape) for pt in sample_points]
                
                # === [FIX 3] KIỂM TRA TRƯỚC KHI CHIA ===
                if len(distances) > 0:
                    avg_dist = sum(distances) / len(distances)
                    
                    if avg_dist < min_score:
                        min_score = avg_dist
                        best_route = r_id
            
            # Logic ngưỡng sai số (Threshold)
            if min_score < 0.003:
                results.append({
                    'Date_File': filename,
                    'Vehicle_ID': veh_id,
                    'Predicted_Route_No': best_route,
                    'Confidence_Score': round(min_score, 6)
                })
            else:
                results.append({
                    'Date_File': filename,
                    'Vehicle_ID': veh_id,
                    'Predicted_Route_No': 'Off-Duty/Unknown',
                    'Confidence_Score': round(min_score, 6)
                })
                
        return pd.DataFrame(results)

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi đọc file {file_path}: {e}")
        return pd.DataFrame()

# --- 3. MAIN LOOP ---
if __name__ == "__main__":
    # Bước 1: Build routes
    route_shapes = build_route_skeletons(ROUTE_DIR)
    
    if not route_shapes:
        print("Không tìm thấy dữ liệu tuyến đường!")
        exit()
        
    # Bước 2: Tìm tất cả file CSV trong thư mục GPS
    gps_files = glob.glob(os.path.join(GPS_DIR, "*.csv"))
    print(f"Tìm thấy {len(gps_files)} file dữ liệu GPS ngày.")
    
    all_mappings = []
    
    # Bước 3: Loop qua từng ngày
    for i, f_path in enumerate(gps_files):
        print(f"[{i+1}/{len(gps_files)}] Processing...", end=" ")
        df_mapping = identify_vehicles_in_file(f_path, route_shapes)
        
        if not df_mapping.empty:
            all_mappings.append(df_mapping)
            
    # Bước 4: Lưu kết quả tổng hợp
    if all_mappings:
        final_df = pd.concat(all_mappings, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n--- HOÀN TẤT! ---")
        print(f"Đã định danh được {len(final_df)} lượt xe.")
        print(f"Kết quả lưu tại: {OUTPUT_FILE}")
        print("Ví dụ 5 dòng đầu:")
        print(final_df.head())
    else:
        print("Không trích xuất được dữ liệu nào.")