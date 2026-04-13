# Parallel OCR Project

Dự án triển khai hệ thống OCR sử dụng Tesseract với 4 phương pháp xử lý để so sánh hiệu suất: tuần tự, song song trên 1 máy, và phân tán trên nhiều máy qua MPI.

## Tính Năng

- Tính CER (Character Error Rate) và WER (Word Error Rate) cho từng ảnh và trung bình.
- Xuất kết quả ra file TXT (tóm tắt) và CSV (dữ liệu thô cho vẽ đồ thị tùy chỉnh).
- Hỗ trợ dataset lớn (10-4000 ảnh).

## Yêu Cầu Hệ Thống

- Python 3.12
- Thư viện: opencv-python, pytesseract, python-levenshtein, mpi4py (cho MPI)
- Tesseract OCR
- Microsoft MPI (cho Windows MPI)

## Cài Đặt

1. Tạo virtual environment:

```
python -m venv ocr_env
ocr_env\Scripts\activate
```

2. Cài thư viện:

```
pip install -r requirements.txt
```

3. Cài đặt Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
4. Cài đặt MS-MPI: https://www.microsoft.com/en-us/download/details.aspx?id=57467

## Cách Chạy

### Chọn input

- Trong file `dataConfig` thay đổi đường dẫn đến IMAGE_FOLDER và LABEL_FOLDER

### Song song với shared memory

- Tuần tự: `python ocrBaseline.py`
- Song song: `python ocrParallel.py`

### Song song vối MPI

- Baseline: `mpiexec -hosts 2 IP1 IP2 -n 2 python ocrMPI_Baseline.py`
- Shared: `mpiexec -hosts 2 IP1 IP2 -n 8 python ocrMPI_Shared.py`

  Hoặc chia process theo

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
├── CODE_GUIDE.md      # Giải thích code
├── requirements.txt   # Thư viện
└── README.md
```

## Lưu ý

- Cả 2 máy phải có cài đặt giống nhau để tránh lỗi
