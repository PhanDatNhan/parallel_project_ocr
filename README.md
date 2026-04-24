# Parallel OCR Project

Dự án triển khai hệ thống OCR sử dụng thư viện Tesseract với 4 phương pháp xử lý:

- Tuần tự: Xử lý hết ảnh này đến ảnh kia.
- Song song bằng Shared Memory: Chia thành các process(worker) để xử lý nhiều ảnh cùng lúc.
- Song song bằng MPI: Sử dụng giao thức truyền thông điệp để xử lý ảnh tuần tự trên nhiều máy tính khác nhau.
- Song song kết hợp Shared Memory và MPI: Sử dụng MPI và song song nhiều process(worker) trên nhiều máy tính khác nhau.

## Tính Năng

- Đo thời gian xử lý từng ảnh khai OCR.
- Tính CER (Character Error Rate), WER (Word Error Rate) từng ảnh và trung bình tất cả ảnh.
- Xuất kết quả ra file TXT (tóm tắt) và CSV (dữ liệu thô cho vẽ đồ thị tùy chỉnh).
- Hỗ trợ dataset lớn (10-4000 ảnh).

## Yêu Cầu Hệ Thống

- Python 3.12
- Thư viện:

```
mpi4py==4.1.1
opencv-python==4.10.0.84
pytesseract==0.3.13
python-Levenshtein==0.27.3
numpy>=1.26.2
pandas>=2.0.0
```

- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Microsoft MPI: https://www.microsoft.com/en-us/download/details.aspx?id=57467

## Cài Đặt

1. Cài đặt Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

2. Cài đặt MS-MPI: https://www.microsoft.com/en-us/download/details.aspx?id=57467

3. Tạo virtual environment:

```
python -m venv ocr_env
ocr_env\Scripts\activate
```

4. Cài thư viện:

```
pip install -r requirements.txt
```

## Cách tạo dữ liệu tự động

- Trong file `dataConfig.py` tùy chỉnh các thông số để cho ra ảnh phù hợp
- Tạo ảnh tự động `py dataGenerator.py `

## Cách Chạy

### Chọn input

- Trong file `dataConfig.py` thay đổi đường dẫn đến IMAGE_FOLDER và LABEL_FOLDER

### Song song với shared memory

- Tuần tự `py ocrBaseline.py`
- Song song `py ocrParallel.py`

### Song song vối MPI

- Baseline: `mpiexec -hosts 2 IP1 IP2 -n 2 python ocrMPI_Baseline.py`
- Shared: `mpiexec -hosts 2 IP1 IP2 -n 8 python ocrMPI_Shared.py`

  Hoặc chia process theo hostfile

- Shared: `mpiexec -machinefile hostfile.txt -n 8 python ocrMPI_Shared.py`

## Cấu Trúc Thư Mục

```
parallel_project/
├── dataset/           # Ảnh và nhãn
├── result/            # Kết quả TXT/CSV
├── ocrBaseline.py     # OCR tuần tự
├── ocrParallel.py     # OCR song song 1 máy
├── ocrMPI_Baseline.py # OCR phân tán tuần tự
├── ocrMPI_Shared.py   # OCR phân tán song song
├── utils.py           # Hàm tính metrics
├── dataConfig.py      # Cấu hình dataset
├── cpuCount.py        # Kiểm tra CPU
├── hostfile.txt       # Cấu hình MPI
├── requirements.txt   # Thư viện
└── README.md
```

## Lưu ý

- Cả 2 máy phải có cài đặt giống nhau để tránh lỗi
