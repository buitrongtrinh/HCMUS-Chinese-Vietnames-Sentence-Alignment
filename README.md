
# **Final Project**
---
# **Danh sách sinh viên thực hiện**

1. **Bùi Trọng Trịnh** - 22120390
2. **Nguyễn Lê Phúc Thắng** - 22120332
3. **abc**
4. **abc**
---

# Hướng dẫn chạy dự án model

## 1. Cài đặt dự án

### 1.1 Cài đặt các công cụ cần thiết

#### Python
- Đảm bảo Python đã được cài đặt trên máy tính. 
- Nếu chưa, tải Python tại [Python.org](https://www.python.org/). (yêu cầu version >= 3.8)


### 1.2 Clone dự án từ GitHub
1. Mở terminal hoặc Command Prompt.
2. Điều hướng đến đường dẫn muốn chạy dự án:
- Thực hiện lệnh sau để tải mã nguồn về máy tính:
  ```bash
  [git clone https://github.com/buitrongtrinh/Final-Project.git](https://github.com/thanhvinh-htnbt/Sentence-Alignment-Model.git)
  ```

### 1.3 Chuyển vào thư mục dự án
- Di chuyển vào thư mục chứa mã nguồn:
  ```bash
  cd Final-Project/
  ```

### 1.4 Cài đặt môi trường và thư viện

1. **Tạo môi trường ảo**:
   - Dùng lệnh sau để tạo một môi trường ảo:
     ```bash
     py -version -m venv project_venv
     ```
     Ví dụ:
     ```bash
     py -3.10 -m venv project_venv
     ```

2. **Kích hoạt môi trường ảo**:
   - Kích hoạt môi trường ảo trên Windows:
     ```bash
     .\project_venv\Scripts\activate
     ```

3. **Cập nhật công cụ quản lý gói pip**:
   - Cập nhật phiên bản mới nhất của pip:
     ```bash
     python.exe -m pip install --upgrade pip
     ```

4. **Cài đặt các thư viện cần thiết**:
   - Cài đặt các thư viện từ file `requirements.txt`:
     ```bash
     python.exe -m pip install -r requirements.txt
     ```
5. **Mở Jupyter Notebook**:
  - Chạy lệnh để khởi động Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
  - Lưu ý: Trình duyệt sẽ tự động mở giao diện Jupyter Notebook. Nếu không, bạn sẽ nhận được một URL trên terminal, hãy copy và dán URL đó vào trình duyệt.
---

## 2. Chạy file Notebook

1. Đảm bảo các file `.ipynb` (file Notebook) nằm trong thư mục dự án.
2. Nhấn vào file cần mở trong giao diện Jupyter để bắt đầu.
3. Cách chạy dự án:
  * Nếu muốn chạy tất cả cell:
    * Trên thanh công cụ nhấn **Run** chọn **Run All Cells** để chạy toàn bộ Cell.
  * Nếu muốn chạy tuần từ:
    * Nhấn **Shift + Enter** để chạy từng cell tuần tự.
6. Nếu dự án cần nhập dữ liệu, hãy đảm bảo dữ liệu đã được đặt trong thư mục đúng.

---

## 3. Lưu ý

- Kiểm tra các thông báo lỗi trong quá trình chạy và khắc phục nếu có.
- Khi hoàn tất, nhấn **"Quit"** trên giao diện Jupyter hoặc dừng server bằng tổ hợp phím:
  ```bash
  Ctrl + C
  ```
- Nếu gặp lỗi hoặc cần thêm thông tin, hãy kiểm tra file `README.md` trong dự án hoặc liên hệ với tác giả.
---

