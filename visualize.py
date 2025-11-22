import os
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
import random
import json
from datetime import datetime

# --- CẤU HÌNH ---
# 1. Thư mục chứa 30 tuyến
ROUTE_ROOT_DIR = r"D:\HCMUT-workplace\BDC_Hackathon\HCMC_bus_routes"
# 2. File GPS của MỘT NGÀY cụ thể (Chọn 1 ngày để visualize thôi nhé)
GPS_FILE_PATH = r"D:\HCMUT-workplace\BDC_Hackathon\processed_GPS\anonymized_final_clean_2025-04-30.csv"
# 3. File kết quả định danh xe (Nếu có - để tô màu xe theo tuyến)
MAPPING_FILE = "Master_Vehicle_Route_Mapping.csv"

# --- HÀM HỖ TRỢ ---
def get_random_hex_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Tạo từ điển màu cho từng tuyến để đồng bộ
route_colors = {}

# --- BƯỚC 1: VẼ LỚP NỀN (LỘ TRÌNH & TRẠM) ---
def draw_static_routes(m):
    print("--- Đang vẽ lớp lộ trình tĩnh (Static Layer)... ---")
    subfolders = [f.path for f in os.scandir(ROUTE_ROOT_DIR) if f.is_dir()]
    
    for folder in subfolders:
        try:
            # Lấy ID tuyến
            folder_name = os.path.basename(folder)
            route_id = folder_name
            
            # Đọc route_by_id để lấy tên chuẩn
            info_path = os.path.join(folder, 'route_by_id.csv')
            if os.path.exists(info_path):
                df_info = pd.read_csv(info_path)
                if not df_info.empty:
                    route_id = str(df_info.iloc[0]['RouteNo'])
            
            # Gán màu cố định cho tuyến này
            if route_id not in route_colors:
                route_colors[route_id] = get_random_hex_color()
            color = route_colors[route_id]
            
            # Vẽ đường đi (Outbound only cho đỡ rối)
            path_out = os.path.join(folder, 'stops_by_var.csv')
            if os.path.exists(path_out):
                df = pd.read_csv(path_out)
                points = df[['Lat', 'Lng']].dropna().values.tolist()
                if points:
                    folium.PolyLine(
                        points, color=color, weight=2, opacity=0.5,
                        tooltip=f"Tuyến {route_id}"
                    ).add_to(m)
                    
        except Exception as e:
            print(f"Lỗi vẽ tuyến {folder}: {e}")

# --- BƯỚC 2: VẼ LỚP ĐỘNG (XE DI CHUYỂN) ---
def create_gps_animation_data(gps_file, mapping_file):
    print(f"--- Đang xử lý dữ liệu GPS động từ file: {os.path.basename(gps_file)} ---")
    
    # 1. Load Mapping (Để biết xe nào thuộc tuyến nào)
    veh_to_route = {}
    if os.path.exists(mapping_file):
        df_map = pd.read_csv(mapping_file)
        # Tạo dict: {'xe_abc': '01', 'xe_xyz': '152'}
        veh_to_route = pd.Series(df_map.Predicted_Route_No.values, index=df_map.Vehicle_ID).to_dict()
    else:
        print("Không tìm thấy file Mapping. Xe sẽ hiển thị màu mặc định.")

    # 2. Load GPS
    # Lấy mẫu: Chỉ lấy 10000 dòng đầu hoặc lấy mẫu ngẫu nhiên để tránh trình duyệt bị đơ
    # Nếu máy mạnh có thể bỏ nrows
    df_gps = pd.read_csv(gps_file) 
    
    # Lọc dữ liệu rác
    df_gps = df_gps.dropna(subset=['lat', 'lng', 'datetime'])
    
    # Convert datetime sang format chuẩn ISO 8601 cho Javascript
    df_gps['datetime'] = pd.to_datetime(df_gps['datetime'])
    # Chỉ lấy dữ liệu trong khoảng thời gian ngắn (ví dụ 1 tiếng buổi sáng) để demo cho nhẹ
    # Bạn có thể comment dòng dưới nếu muốn chạy cả ngày
    df_gps = df_gps[(df_gps['datetime'].dt.hour >= 6) & (df_gps['datetime'].dt.hour < 7)]
    
    df_gps['time_str'] = df_gps['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    features = []
    
    print(f"Đang tạo animation cho {len(df_gps)} điểm dữ liệu...")

    for _, row in df_gps.iterrows():
        veh_id = row['anonymized_vehicle']
        
        # Xác định màu xe:
        # Nếu xe đó thuộc tuyến 01 -> Lấy màu của tuyến 01
        # Nếu không -> Màu xám
        color = '#333333' # Mặc định xám đen
        route_of_veh = veh_to_route.get(veh_id, 'Unknown')
        
        if route_of_veh in route_colors:
            color = route_colors[route_of_veh]
        
        # Cấu trúc GeoJSON Feature cho TimestampedGeoJson
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['lng'], row['lat']]
            },
            'properties': {
                'time': row['time_str'], # Thời gian hiển thị
                'style': {'color': color}, # Màu viền
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': 0.8,
                    'stroke': 'false',
                    'radius': 5 # Kích thước chấm tròn xe
                },
                'popup': f"Xe: {veh_id}<br>Tuyến: {route_of_veh}"
            }
        }
        features.append(feature)

    return features

# --- MAIN ---
if __name__ == "__main__":
    # 1. Khởi tạo bản đồ
    m = folium.Map(location=[10.7769, 106.7009], zoom_start=12, tiles='CartoDB positron')

    # 2. Vẽ lớp Tĩnh (Đường đi)
    draw_static_routes(m)

    # 3. Xử lý lớp Động (Xe)
    if os.path.exists(GPS_FILE_PATH):
        geo_features = create_gps_animation_data(GPS_FILE_PATH, MAPPING_FILE)
        
        if geo_features:
            print("Đang thêm Plugin Animation vào bản đồ...")
            TimestampedGeoJson(
                {'type': 'FeatureCollection', 'features': geo_features},
                period='PT10S',    # Thời gian giữa các bước nhảy (10 giây)
                duration='PT1M',   # Thời gian tồn tại của điểm (1 phút - tạo hiệu ứng đuôi)
                add_last_point=True,
                auto_play=False,
                loop=False,
                max_speed=10,
                loop_button=True,
                date_options='YYYY-MM-DD HH:mm:ss',
                time_slider_drag_update=True
            ).add_to(m)
        else:
            print("Không có dữ liệu GPS hợp lệ trong khoảng thời gian lọc.")
    else:
        print(f"Không tìm thấy file GPS: {GPS_FILE_PATH}")

    # 4. Lưu file
    output_file = "Bus_Simulation_Map.html"
    m.save(output_file)
    print(f"\nHOÀN TẤT! Mở file '{output_file}' để xem mô phỏng.")
    
    # Mở file tự động
    try:
        os.startfile(output_file)
    except:
        pass