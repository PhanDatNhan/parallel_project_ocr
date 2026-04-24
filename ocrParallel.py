import os
import time
import cv2
import pytesseract
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from utils import calculate_cer, calculate_wer
import dataConfig
import csv

# =========================
# CONFIG
# =========================
MAX_WORKERS = 8     # 2: 2 workers, 4: 4 workers, 8: 8 workers
BATCH_SIZE = 20     # 10:2, 1000:20, 2000:30, 4000:50


# Tắt multithreading của Tesseract
os.environ["OMP_THREAD_LIMIT"] = "1"


# =========================
# Worker xử lý batch
# =========================
def process_batch(batch_files):
    results = []

    for image_file in batch_files:
        image_path = os.path.join(dataConfig.IMAGE_FOLDER, image_file)
        label_path = os.path.join(
            dataConfig.LABEL_FOLDER,
            image_file.replace(".png", ".txt")
        )

        start = time.perf_counter()

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        predicted_text = pytesseract.image_to_string(img)

        with open(label_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()

        cer = calculate_cer(ground_truth, predicted_text)
        wer = calculate_wer(ground_truth, predicted_text)

        elapsed = time.perf_counter() - start

        results.append((image_file, cer, wer, elapsed))

    return results


# =========================
# Helper chia batch
# =========================
def chunkify(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# =========================
# MAIN PARALLEL ENGINE
# =========================
def run_ocr_production():
    image_files = sorted(os.listdir(dataConfig.IMAGE_FOLDER))

    batches = list(chunkify(image_files, BATCH_SIZE))

    print(f"Workers: {MAX_WORKERS}, Batch size: {BATCH_SIZE}")

    start_total = time.perf_counter()

    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]

        for future in as_completed(futures):
            results.extend(future.result())

    total_time = time.perf_counter() - start_total

    # =========================
    # Metrics
    # =========================
    total_images = len(results)
    avg_cer = sum(r[1] for r in results) / total_images
    avg_wer = sum(r[2] for r in results) / total_images
    avg_time = sum(r[3] for r in results) / total_images

    print("\n===== PRODUCTION RESULT =====")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time/image: {avg_time:.3f}s")
    print(f"Avg CER: {avg_cer:.4f}")
    print(f"Avg WER: {avg_wer:.4f}")

    # =========================
    # Save CSV
    # =========================
    os.makedirs("result", exist_ok=True)

    with open("result/parallel_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Time", "CER", "WER"])

        for img, cer, wer, t in results:
            writer.writerow([img, t, cer, wer])

    return results


if __name__ == "__main__":
    run_ocr_production()
