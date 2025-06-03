#  Nhận Diện Cảm Xúc Từ Hình Ảnh Khuôn Mặt

Dự án này xây dựng hệ thống nhận diện cảm xúc từ ảnh khuôn mặt bằng cách kết hợp các thuật toán học máy truyền thống và mạng nơ-ron tích chập (CNN), sử dụng các tập dữ liệu phổ biến như **RAF-DB**, **CK+** và **FER2013**.

---

##  Tập Dữ Liệu Sử Dụng

- **RAF-DB**: Dùng cho cả thuật toán ML truyền thống và CNN.
- **CK+**: Dùng cho mô hình CNN.
- **FER2013**: Dùng cho mô hình CNN.

---

##  Phương Pháp

### 1. Học máy truyền thống (trên tập RAF-DB)
- **Tiền xử lý**: Tăng cường dữ liệu (Data Augmentation), trích xuất đặc trưng,... (HOG)
- **Thuật toán sử dụng**:
  - SVM (Support Vector Machine)
  - Random Forest
  - KNN (k-Nearest Neighbors)
  - Logistic Regression
  - Decision tree

### 2. Mạng nơ-ron tích chập (CNN)
- **Kiến trúc**: CNN 4 lớp
- **Thư viện**: TensorFlow / Keras 
- **Tiền xử lý**:
  - Resize ảnh
  - Lật ảnh
  - Tăng cường dữ liệu nhẹ (Data Augmentation)
- **Hàm mất mát**: Categorical Crossentropy
- **Tối ưu hóa**: Adam

---

## 😊 Nhãn Cảm Xúc

Mô hình phân loại các cảm xúc cơ bản sau (có thể thay đổi theo tập dữ liệu):
- 😲 Ngạc nhiên (Surprise)
- 😨 Sợ hãi (Fear)
- 😖 Ghê tởm (Disgust)
- 😄 Vui (Happy)
- 😢 Buồn (Sad)
- 😠 Tức giận (Angry)
- 😐 Bình thường / Trung tính (Neutral)


---

##  Kết Quả

| Loại mô hình         | Tập dữ liệu | Độ chính xác |
|----------------------|-------------|--------------|
| SVM (HOG)            | RAF-DB      | ~75%       |
| Random Forest        | RAF-DB      | ~63%       |
| Logistic Regression  | RAF-DB      | ~65%       |
| KNN                  | RAF-DB      | ~68%       |
| CNN                  | FER2013     | ~65%       |
| CNN                  | CK+         | ~99.6%       |
| CNN                  | RAF-DB      | ~83.37%       |


