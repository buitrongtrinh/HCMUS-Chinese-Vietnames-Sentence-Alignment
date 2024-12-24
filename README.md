
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
  git clone https://github.com/thanhvinh-htnbt/Sentence-Alignment-Model.git
  ```

### 1.3 Chuyển vào thư mục dự án
- Di chuyển vào thư mục chứa mã nguồn:
  ```bash
  cd Sentence-Alignment-Model/
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

5. **Cài đặt thư viện pytorch**:
   - Tải pytorch tại [pytorch.org](https://pytorch.org/)
   - Chọn version phù hợp với cấu hình máy
     Ví dụ: 
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
6. **Mở VSCode**:
  - Chạy lệnh để khởi động VSCode:
   ```bash
   code .
   ```
---

## 2. Chạy dự án:
Đối với các case:
- Data từ thầy:
  ```bash
  python main.py --viet_file data/viet.txt --hoa_file data/hoa.txt --threshold 0.7
  ```
- 4,000 dòng:
  ```bash
  python main.py --viet_file data/4000bible-uedin.vi-zh.vi --hoa_file data/4000shuffled_bible-uedin.vi-zh.zh --threshold 0.7
  ```
- 8,000 dòng:
  ```bash
  python main.py --viet_file data/8000bible-uedin.vi-zh.vi --hoa_file data/8000shuffled_bible-uedin.vi-zh.zh --threshold 0.7
- 30,000 dòng:
  ```bash
  python main.py --viet_file data/30000bible-uedin.vi-zh.vi --hoa_file data/30000shuffled_bible-uedin.vi-zh.zh --threshold 0.7
  ```
- 50,000 dòng:
  ```bash
  python main.py --viet_file data/50000bible-uedin.vi-zh.vi --hoa_file data/50000shuffled_bible-uedin.vi-zh.zh --threshold 0.7
  ```
---

## 3. Lưu ý

- Kiểm tra các thông báo lỗi trong quá trình chạy và khắc phục nếu có.
- Nếu gặp lỗi hoặc cần thêm thông tin, hãy kiểm tra file `README.md` trong dự án hoặc liên hệ với tác giả.
---

