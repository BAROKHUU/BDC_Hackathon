import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta

# --- CẤU HÌNH ---
# Đường dẫn chứa file GPS raw
GPS_FOLDER = r"D:\HCMUT-workplace\BDC_Hackathon\processed_GPS"
# Đường dẫn chứa file Mapping (kết quả của bước Map Matching trước đó)
MAPPING_FILE = r"D:\HCMUT-workplace\BDC_Hackathon\Master_Vehicle_Route_Mapping.csv" 
# Đường dẫn file trạm dừng Tuyến 01 
STOPS_FILE = r"D:\HCMUT-workplace\BDC_Hackathon\HCMC_bus_routes\88\stops_by_var.csv"

def create_travel_time_dataset():
    # 1. Load dữ liệu
    print("Đang load dữ liệu...")
    if not os.path.exists(MAPPING_FILE):
        print(f"LỖI: Chưa có file {MAPPING_FILE}. Hãy chạy code Map Matching ở bước trước!")
        return
    
    df_mapping = pd.read_csv(MAPPING_FILE)
    
    # Lọc chỉ lấy các xe chạy Tuyến 01 để phân tích (cho nhẹ máy)
    target_route = "88" # Hoặc "1", tùy file config của bạn
    vehicles_route_01 = df_mapping[df_mapping['Predicted_Route_No'] == target_route]['Vehicle_ID'].unique()
    print(f"Tìm thấy {len(vehicles_route_01)} xe chạy tuyến {target_route}.")

    # Load danh sách trạm Tuyến 01
    df_stops = pd.read_csv(STOPS_FILE)
    # Chỉ lấy Lat/Lng và ID trạm
    stops_coords = df_stops[['StopId', 'Lat', 'Lng', 'Name']].to_dict('records')

    dataset = []

    # 2. Quét qua các file GPS hàng ngày
    gps_files = glob.glob(os.path.join(GPS_FOLDER, "*.csv"))
    
    for f_path in gps_files:
        print(f"Đang xử lý file: {os.path.basename(f_path)}")
        try:
            df_gps = pd.read_csv(f_path)
            # Chỉ giữ lại các xe thuộc Tuyến 01
            df_gps = df_gps[df_gps['anonymized_vehicle'].isin(vehicles_route_01)].copy()
            
            if df_gps.empty: continue

            df_gps['datetime'] = pd.to_datetime(df_gps['datetime'])
            df_gps = df_gps.sort_values(['anonymized_vehicle', 'datetime'])

            # 3. Thuật toán tính thời gian giữa các trạm
            for veh_id, trip_data in df_gps.groupby('anonymized_vehicle'):
                
                # Tìm thời điểm xe đi qua từng trạm
                # Logic đơn giản: Tìm điểm GPS gần trạm nhất (< 100m)
                stop_times = {}
                
                for stop in stops_coords:
                    # Tính khoảng cách (Manhattan distance cho nhanh)
                    # 0.001 độ ~ 100m
                    mask = (abs(trip_data['lat'] - stop['Lat']) < 0.001) & \
                           (abs(trip_data['lng'] - stop['Lng']) < 0.001)
                    
                    nearby_points = trip_data[mask]
                    
                    if not nearby_points.empty:
                        # Lấy thời điểm đầu tiên chạm vào trạm
                        arrival_time = nearby_points.iloc[0]['datetime']
                        stop_times[stop['StopId']] = arrival_time

                # Tính thời gian di chuyển giữa các cặp trạm liền kề
                # Giả sử trạm trong file stops_by_var đã sắp xếp theo thứ tự lộ trình
                for i in range(len(stops_coords) - 1):
                    start_node = stops_coords[i]
                    end_node = stops_coords[i+1]
                    
                    t1 = stop_times.get(start_node['StopId'])
                    t2 = stop_times.get(end_node['StopId'])
                    
                    if t1 and t2 and t2 > t1:
                        duration = (t2 - t1).total_seconds() / 60.0 # Phút
                        
                        # Lọc nhiễu: Nếu đi 1 trạm mà mất > 60 phút hoặc < 0.5 phút -> Sai số GPS -> Bỏ
                        if 0.5 < duration < 60:
                            dataset.append({
                                'Date': t1.date(),
                                'Hour': t1.hour,
                                'DayOfWeek': t1.dayofweek, # 0=Mon, 6=Sun
                                'From_Stop': start_node['Name'],
                                'To_Stop': end_node['Name'],
                                'Segment_Index': i,
                                'Duration_Minutes': duration
                            })

        except Exception as e:
            print(f"Lỗi file {f_path}: {e}")

    # 4. Lưu kết quả
    df_final = pd.DataFrame(dataset)
    df_final.to_csv("AI_Training_Data_Route01.csv", index=False)
    print(f"\nHoàn tất! Đã tạo file dữ liệu huấn luyện: AI_Training_Data_Route01.csv ({len(df_final)} dòng)")
    print(df_final.head())

if __name__ == "__main__":
    create_travel_time_dataset()