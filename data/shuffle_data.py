import random

def shuffle_file_lines(input_file, output_file, row):
    try:
        # Đọc tất cả các dòng từ tệp đầu vào
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()[:row]

        # Xáo trộn các dòng
        random.shuffle(lines)

        # Ghi các dòng đã xáo trộn vào tệp đầu ra
        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        print(f"Các dòng đã được xáo trộn và lưu vào '{output_file}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        
def cut_file(input_file, output_file, row):
    try: 
        # Đọc tất cả các dòng từ tệp đầu vào
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()[:row]
        # Ghi các dòng đã xáo trộn vào tệp đầu ra
        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(lines)
            print(f"Các dòng đã được xáo trộn và lưu vào '{output_file}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Tên tệp đầu vào và đầu ra
row = 30000
input_file = "data/bible-uedin.vi-zh.zh"  # Thay thế bằng đường dẫn tệp của bạn
output_file = f"data/{row}shuffled_bible-uedin.vi-zh.zh"

# Gọi hàm để xáo trộn
shuffle_file_lines(input_file, output_file, row)

cut_file("data/bible-uedin.vi-zh.vi", f"data/{row}bible-uedin.vi-zh.vi", row)