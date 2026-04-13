import os
import time
import cv2
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import calculate_cer, calculate_wer
import dataConfig
import csv

# Tối ưu threading và I/O
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Config tối ưu cho 8GB RAM + SSD
MAX_WORKERS = 4
BATCH_SIZE = 20
image_cache = {}


# ==============================
# Pre-load images vào cache
# ==============================
def preload_images(image_files):
    global image_cache
    print(f"Caching {len(image_files)} images vào RAM...")
    for image_file in image_files:
        image_path = os.path.join(dataConfig.IMAGE_FOLDER, image_file)
        image_cache[image_file] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"✓ Cached {len(image_cache)} images\n")


# ==============================
# Chia batch
# ==============================
def chunkify(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ==============================
# Hàm xử lý batch
# ==============================
def process_batch(batch_files):
    results = []
    for image_file in batch_files:
        label_path = os.path.join(
            dataConfig.LABEL_FOLDER,
            image_file.replace(".png", ".txt")
        )

        start = time.perf_counter()

        # Lấy từ cache RAM
        img = image_cache[image_file]

        predicted_text = pytesseract.image_to_string(img)

        with open(label_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()

        cer = calculate_cer(ground_truth, predicted_text)
        wer = calculate_wer(ground_truth, predicted_text)

        elapsed = time.perf_counter() - start

        results.append((image_file, cer, wer, elapsed))

    return results


# ==============================
# Chạy song song tối ưu
# ==============================
def run_parallel_ocr(num_processes=None):

    if num_processes is None:
        num_processes = MAX_WORKERS

    image_files = sorted(os.listdir(dataConfig.IMAGE_FOLDER))

    # Pre-load images
    preload_images(image_files)

    batches = list(chunkify(image_files, BATCH_SIZE))

    print(f"Config: Workers={num_processes}, Batch Size={BATCH_SIZE}")
    print(f"Total: {len(image_files)} images, {len(batches)} batches\n")

    start_time = time.perf_counter()

    results = []

    # ThreadPoolExecutor - tối ưu cho I/O-bound
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]

        for future in as_completed(futures):
            results.extend(future.result())

    end_time = time.perf_counter()

    # Metrics
    _image_files, cers, wers, times = zip(*results)

    total_time = end_time - start_time
    avg_cer = sum(cers) / len(cers)
    avg_wer = sum(wers) / len(wers)
    avg_time = sum(times) / len(times)

    print("\n===== KẾT QUẢ SONG SONG (OPTIMIZED) =====")
    print(f"Tổng thời gian: {total_time:.2f} giây")
    print(f"Thời gian/ảnh: {avg_time:.3f}s")
    print(f"Throughput: {len(image_files)/total_time:.1f} images/sec")
    print(f"Trung bình CER: {avg_cer:.4f}")
    print(f"Trung bình WER: {avg_wer:.4f}")

    # Lưu TXT
    os.makedirs('result', exist_ok=True)
    with open(os.path.join('result', 'parallel_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Tổng thời gian: {total_time:.2f} giây\n")
        f.write(f"Thời gian/ảnh: {avg_time:.3f}s\n")
        f.write(f"Trung bình CER: {avg_cer:.4f}\n")
        f.write(f"Trung bình WER: {avg_wer:.4f}\n")
        f.write("Chi tiết từng ảnh:\n")
        for img, c, w, t in results:
            f.write(f"{img}: {t:.3f}s, CER {c:.4f}, WER {w:.4f}\n")

    # Lưu CSV
    with open(os.path.join('result', 'parallel_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Time', 'CER', 'WER'])
        for img, c, w, t in results:
            writer.writerow([img, t, c, w])

    print(f"\n✓ Saved results\n")

    return total_time, avg_cer, avg_wer


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    run_parallel_ocr()
