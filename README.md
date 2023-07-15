# Ứng dụng tô màu cho ảnh đen trắng
## 1. Giới thiệu
Đây là một chương trình sử dụng mạng nơ-ron nhân tạo cho bài toán học có giám sát cho ứng dụng tô màu cho ảnh đen trăng
## 2. Các thư viện sử dụng
- **OpenCV**: Cung cấp các công cụ thao tác với ảnh
- **Numpy**: Hỗ trợ cho việc tính toán các mảng nhiều chiều
- **Tensorflow**: Hỗ trọ xây dựng các mô hình mạng
- **Keras**: Hỗ trợ xây dựng CNN
- **Matplotlib**: Vẽ đồ thị, trực quan hóa các kết quả đánh giá
## 3. Cài đặt và sử dụng
### Bước 1: Chuẩn bị
Cài đặt các thư viện trong file `setup.txt`  
`pip install [tên thư viện]`  
Tạo các folder data_raw, data_processed, models
### Bước 2: Đưa tập dữ liệu ảnh vào folder data_raw
### Bước 3: Tiền xử lý dữ liệu
Chạy file: ``python preprocess.py`` , đầu vào và các nhãn được lưu tại folder data_processed
### Bước 4: Huấn luyện mô hình
Chạy file: ``python train_model.py`` , mô hình được lưu tại folder models
### Bước 5: Tô màu cho ảnh
Chạy file: ``python test_model.py``   
Ảnh đầu vào lưu tại thư mục cùng cấp và đặt tên là `input.png`  
Kết quả được lưu tại `output.png`
