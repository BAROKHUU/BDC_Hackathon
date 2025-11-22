import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

# --- CẤU HÌNH ---
MODEL_FILE = r"D:\HCMUT-workplace\BDC_Hackathon\bus_travel_time_model_xgb.pkl"
STOPS_FILE = r"D:\HCMUT-workplace\BDC_Hackathon\HCMC_bus_routes\88\stops_by_var.csv"

def generate_smart_schedule_real():
    print(f"--- LẬP LỊCH XUẤT PHÁT THÔNG MINH (DỰA TRÊN TRẠM THỰC TẾ) ---")
    
    # 1. Load Model AI
    if not os.path.exists(MODEL_FILE):
        print("Lỗi: Chưa có file model. Hãy chạy bước train trước!")
        return
    model = joblib.load(MODEL_FILE)

    # 2. Đọc file Trạm dừng
    if not os.path.exists(STOPS_FILE):
        print(f"Lỗi: Không tìm thấy file trạm dừng tại {STOPS_FILE}")
        return
    
    df_stops = pd.read_csv(STOPS_FILE)

    real_num_segments = len(df_stops) - 1
    
    print(f"Đã đọc file trạm dừng: {len(df_stops)} trạm.")
    print(f"-> Hệ thống sẽ tính toán tổng thời gian của {real_num_segments} đoạn đường nối tiếp nhau.\n")

    # 3. Thiết lập ngày mai
    tomorrow = datetime.now() + timedelta(days=1)
    target_date = tomorrow.date()
    day_of_week = tomorrow.weekday()
    
    start_target = datetime.combine(target_date, datetime.strptime("06:00", "%H:%M").time())
    end_target   = datetime.combine(target_date, datetime.strptime("09:00", "%H:%M").time())
    
    current_target = start_target
    schedule_table = []

    print("Đang tính toán... (Vui lòng đợi AI quét qua toàn bộ lộ trình)")

    while current_target <= end_target:
        hour_check = current_target.hour

        # --- Tạo input data cho tất cả segments ---
        input_data = pd.DataFrame({
            'Hour': [hour_check] * real_num_segments,
            'DayOfWeek': [day_of_week] * real_num_segments,
            'Segment_Index': list(range(real_num_segments))
        })
        
        # --- Dự đoán ---
        predictions = model.predict(input_data)

        # --- FIX LỖI: chuyển numpy.float32 -> float ---
        total_duration = float(np.sum(predictions))

        # --- Tính giờ xuất phát ---
        departure_time = current_target - timedelta(minutes=total_duration)

        schedule_table.append({
            "Giờ Đến Đích (Target)": current_target.strftime("%H:%M"),
            "Tổng Thời Gian (Phút)": round(total_duration, 2),
            "GIỜ XUẤT BẾN GỢI Ý": departure_time.strftime("%H:%M"),
            "Trạng Thái": "🔴 Cao điểm" if total_duration > 45 else "🟢 Bình thường"
        })
        
        current_target += timedelta(minutes=15)

    # 5. Xuất kết quả
    df_schedule = pd.DataFrame(schedule_table)
    print("\nBẢNG KẾT QUẢ CHI TIẾT:")
    print(df_schedule.to_string(index=False))
    
    df_schedule.to_csv("Real_Smart_Schedule.csv", index=False)
    print("\n-> Đã lưu vào file: Real_Smart_Schedule.csv")

if __name__ == "__main__":
    generate_smart_schedule_real()
