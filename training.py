import pandas as pd
import numpy as np
from xgboost import XGBRegressor  # <--- THAY ĐỔI QUAN TRỌNG
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

# --- CẤU HÌNH ---
DATA_FILE = "AI_Training_Data_Route01.csv"
MODEL_FILE = "bus_travel_time_model_xgb.pkl" # Đổi tên file model chút cho ngầu

def train_model_xgboost():
    print("--- HUẤN LUYỆN AI VỚI XGBOOST (STATE-OF-THE-ART) ---")
    
    # 1. Load dữ liệu
    if not os.path.exists(DATA_FILE):
        print("Chưa có dữ liệu! Hãy chạy Bước 1 trước.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Dữ liệu đầu vào: {len(df)} dòng.")
    
    X = df[['Hour', 'DayOfWeek', 'Segment_Index']]
    y = df['Duration_Minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Khởi tạo & Train XGBoost
    # - n_estimators=500: Tạo 500 cây sửa sai liên tiếp
    # - learning_rate=0.05: Học chậm mà chắc (giúp mô hình thông minh hơn)
    # - max_depth=7: Độ sâu của cây (đủ sâu để hiểu quy luật phức tạp)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        n_jobs=-1, # Dùng hết CPU để chạy cho nhanh
        random_state=42
    )
    
    print("Đang training XGBoost... (Tốc độ tên lửa)")
    model.fit(X_train, y_train)
    
    # 3. Đánh giá
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Tính % độ chính xác (cho dễ chém gió)
    mean_time = np.mean(y_test)
    accuracy_percentage = 100 * (1 - (mae / mean_time))

    print(f"\n>>> KẾT QUẢ ẤN TƯỢNG:")
    print(f"Sai số trung bình (MAE): {mae:.2f} phút")
    print(f"Độ chính xác ước tính: {accuracy_percentage:.1f}%")

    # 4. Lưu model
    joblib.dump(model, MODEL_FILE)
    print(f"Đã lưu siêu mô hình vào: {MODEL_FILE}")
    
    # --- VISUALIZATION ---
    print("\nĐang vẽ biểu đồ so sánh...")
    hours_range = np.arange(5, 21, 0.5) # Mịn hơn
    test_input = pd.DataFrame({
        'Hour': hours_range,
        'DayOfWeek': [0] * len(hours_range), # Thứ 2
        'Segment_Index': [0] * len(hours_range) 
    })
    
    predicted_times = model.predict(test_input)
    
    plt.figure(figsize=(12, 6))
    # Vẽ vùng nền (để đẹp hơn)
    plt.fill_between(hours_range, 0, predicted_times, color='red', alpha=0.1)
    plt.plot(hours_range, predicted_times, color='red', linewidth=2.5, label='XGBoost Prediction')
    
    plt.title(f"Dự báo XGBoost: Thời gian di chuyển Thứ 2\n(Độ chính xác: {accuracy_percentage:.1f}%)", fontsize=14)
    plt.xlabel("Giờ trong ngày")
    plt.ylabel("Thời gian (Phút)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("xgboost_prediction.png")
    plt.show()

if __name__ == "__main__":
    train_model_xgboost()
