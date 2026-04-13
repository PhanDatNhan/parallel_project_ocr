import os
import time
import pytesseract
import cv2
from utils import calculate_cer, calculate_wer
import dataConfig


# ==============================
# OCR 1 ảnh
# ==============================
def ocr_single_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text


# ==============================
# Load ground truth
# ==============================
def load_ground_truth(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# Chạy OCR tuần tự
# ==============================
def run_sequential_ocr():
    image_files = sorted(os.listdir(dataConfig.IMAGE_FOLDER))

    times = []
    cers = []
    wers = []
    total_cer = 0
    total_wer = 0
    total_time_start = time.perf_counter()

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(dataConfig.IMAGE_FOLDER, image_file)
        label_path = os.path.join(
            dataConfig.LABEL_FOLDER,
            image_file.replace(".png", ".txt")
        )

        # OCR
        start = time.perf_counter()
        predicted_text = ocr_single_image(image_path)
        end = time.perf_counter()

        # Ground truth
        ground_truth = load_ground_truth(label_path)

        # CER và WER
        cer = calculate_cer(ground_truth, predicted_text)
        wer = calculate_wer(ground_truth, predicted_text)
        total_cer += cer
        total_wer += wer
        cers.append(cer)
        wers.append(wer)
        times.append(end - start)

        print(
            f"Ảnh {image_file} - Process 0: Thời gian {end-start:.3f}s, CER {cer:.4f}, WER {wer:.4f}")

    total_time_end = time.perf_counter()

    avg_cer = total_cer / len(image_files)
    avg_wer = total_wer / len(image_files)
    total_time = total_time_end - total_time_start

    print("\n===== KẾT QUẢ BASELINE =====")
    print(f"Tổng thời gian: {total_time:.2f} giây")
    print(f"Trung bình CER: {avg_cer:.4f}")
    print(f"Trung bình WER: {avg_wer:.4f}")

    # Lưu kết quả vào file txt
    with open(os.path.join('result', 'baseline_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Tổng thời gian: {total_time:.2f} giây\n")
        f.write(f"Trung bình CER: {avg_cer:.4f}\n")
        f.write(f"Trung bình WER: {avg_wer:.4f}\n")
        f.write("Chi tiết từng ảnh:\n")
        for i, (c, w, t) in enumerate(zip(cers, wers, times)):
            f.write(
                f"Ảnh {i+1}: Thời gian {t:.3f}s, CER {c:.4f}, WER {w:.4f}\n")

    # Lưu kết quả vào file CSV
    import csv
    with open(os.path.join('result', 'baseline_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Time', 'CER', 'WER'])
        for i, (c, w, t) in enumerate(zip(cers, wers, times)):
            writer.writerow([f'img_{i+1:04d}.png', t, c, w])

    return total_time, avg_cer, avg_wer


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    run_sequential_ocr()
