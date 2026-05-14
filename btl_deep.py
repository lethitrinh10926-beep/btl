import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Trích xuất đặc trưng từ URL
# -------------------------------
def extract_features(url):
    """Nhận URL string, trả về vector đặc trưng dạng list."""
    features = []
    
    # Độ dài URL
    features.append(len(url))
    
    # Số lượng dấu chấm
    features.append(url.count('.'))
    
    # Số lượng dấu gạch ngang
    features.append(url.count('-'))
    
    # Số lượng dấu gạch chéo
    features.append(url.count('/'))
    
    # Số lượng dấu hỏi
    features.append(url.count('?'))
    
    # Số lượng dấu bằng
    features.append(url.count('='))
    
    # Số lượng các ký tự đặc biệt
    special_chars = '@_&%#'
    features.append(sum(url.count(ch) for ch in special_chars))
    
    # Số lượng chữ số
    features.append(sum(c.isdigit() for c in url))
    
    # Số lượng chữ in hoa
    features.append(sum(c.isupper() for c in url))
    
    # Có chứa địa chỉ IP?
    ip_pattern = r'\d+\.\d+\.\d+\.\d+'
    features.append(1 if re.search(ip_pattern, url) else 0)
    
    # Có sử dụng HTTPS?
    features.append(1 if url.startswith('https://') else 0)
    
    # Số lượng từ khóa đáng ngờ
    suspicious_keywords = ['login', 'verify', 'secure', 'account', 'update', 'bank', 'paypal', 'signin']
    count_keywords = sum(url.lower().count(kw) for kw in suspicious_keywords)
    features.append(count_keywords)
    
    # Tỷ lệ ký tự không phải chữ cái
    non_alpha = sum(1 for c in url if not c.isalpha())
    ratio_non_alpha = non_alpha / len(url) if len(url) > 0 else 0
    features.append(ratio_non_alpha)
    
    return np.array(features).reshape(1, -1)

# -------------------------------
# 2. Huấn luyện mô hình (binary)
# -------------------------------
def train_model(csv_path, model_save_path='url_model.pkl', scaler_save_path='scaler.pkl'):
    """Đọc dataset, gộp nhãn thành binary (0=benign, 1=malicious), train model."""
    df = pd.read_csv(csv_path)
    
    # Xác định cột URL và cột nhãn
    url_col = 'url' if 'url' in df.columns else 'URL'
    label_col = 'label' if 'label' in df.columns else 'type'
    
    # Chuyển nhãn về dạng nhị phân: 0 = benign, 1 = độc hại (gồm defacement, malware, phishing)
    if label_col in df.columns:
        # Nếu nhãn dạng string
        if df[label_col].dtype == 'object':
            df['binary_label'] = (df[label_col] != 'benign').astype(int)
        else:  # nhãn dạng số: giả sử 0 = benign, >0 = độc hại
            df['binary_label'] = (df[label_col] != 0).astype(int)
    else:
        raise ValueError("Không tìm thấy cột nhãn (label/type) trong dataset")
    
    X = []
    y = df['binary_label'].values
    
    print("Đang trích xuất đặc trưng từ URL...")
    for url in df[url_col]:
        X.append(extract_features(url).flatten())
    X = np.array(X)
    
    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Huấn luyện Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['benign', 'malicious']))
    
    # Lưu model và scaler
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Đã lưu model vào {model_save_path} và scaler vào {scaler_save_path}")
    return model, scaler

# -------------------------------
# 3. Hàm dự đoán cho một URL (đã sửa lỗi index)
# -------------------------------
def predict_url(url, model_path='url_model.pkl', scaler_path='scaler.pkl'):
    """Dự đoán URL có độc hại hay không."""
    # Load model và scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Trích xuất đặc trưng và chuẩn hóa
    features = extract_features(url)
    features_scaled = scaler.transform(features)
    
    # Dự đoán (chắc chắn lấy scalar)
    pred = model.predict(features_scaled)
    # Nếu pred là mảng 1D, lấy phần tử đầu
    if isinstance(pred, np.ndarray):
        pred = pred.item() if pred.size == 1 else pred[0]
    else:
        pred = int(pred)
    
    # Xác suất (lấy hàng đầu tiên)
    proba = model.predict_proba(features_scaled)[0]  # shape (n_classes,)
    confidence = proba[pred] * 100
    
    label = "độc hại (malicious)" if pred == 1 else "an toàn (benign)"
    
    return {
        'url': url,
        'prediction': pred,
        'label': label,
        'confidence': f"{confidence:.2f}%"
    }

# -------------------------------
# 4. Chạy thử (train + demo)
# -------------------------------
if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\python\khdl\balanced_urls.csv"  # <<< Đường dẫn dataset của bạn
    
    try:
        train_model(DATASET_PATH)
    except FileNotFoundError:
        print(f"Không tìm thấy file {DATASET_PATH}. Bỏ qua bước train.")
        print("Tải dataset từ Kaggle hoặc tạo file csv với cột 'url' và 'label'.")
    
    print("\n--- DEMO DỰ ĐOÁN URL ---")
    test_urls = [
        "https://www.google.com",
        "http://paypal.com.verify-account.login.secure.com",
        "https://facebook.com",
        "http://192.168.1.1/update/bank/login"
    ]
    for u in test_urls:
        result = predict_url(u)
        print(f"URL: {result['url']}\n  => {result['label']} (độ tin cậy: {result['confidence']})\n")