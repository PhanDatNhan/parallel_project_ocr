from mpi4py import MPI
import time
import os
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

# Config batch processing
BATCH_SIZE = 5
MAX_THREADS = 2  # Threads cho MPI process (ít hơn để tránh contention)

# Cache toàn cục
image_cache = {}


# ==============================
# Pre-load images (rank 0 pre-load, broadcast cache)
# ==============================
def preload_images(image_files, rank, comm):
    global image_cache
    if rank == 0:
        print(f"[Rank 0] Caching {len(image_files)} images vào RAM...")
        start = time.perf_counter()
        for img_file in image_files:
            path = os.path.join(dataConfig.IMAGE_FOLDER, img_file)
            image_cache[img_file] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elapsed = time.perf_counter() - start
        print(f"[Rank 0] ✓ Cached trong {elapsed:.2f}s\n")

    # Broadcast cache to all ranks
    image_cache = comm.bcast(image_cache, root=0)


# ==============================
# Chia batch
# ==============================
def chunkify(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ==============================
# Xử lý batch (ThreadPoolExecutor cho song song trên rank)
# ==============================
def process_batch(batch_files, rank):
    results = []
    for image_file in batch_files:
        label_path = os.path.join(
            dataConfig.LABEL_FOLDER,
            image_file.replace(".png", ".txt")
        )

        start = time.perf_counter()

        # Lấy ảnh từ cache RAM
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
# Chạy MPI Shared (phân tán + song parallel)
# ==============================
def run_mpi_shared():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running MPI Shared with {size} processes")
        image_files = sorted(os.listdir(dataConfig.IMAGE_FOLDER))
    else:
        image_files = None

    # Broadcast image files to all ranks
    image_files = comm.bcast(image_files, root=0)

    # Pre-load images vào cache
    preload_images(image_files, rank, comm)

    # Chia đều ảnh cho các rank
    chunk_size = len(image_files) // size
    remainder = len(image_files) % size
    start_idx = rank * chunk_size + min(rank, remainder)
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
    local_images = image_files[start_idx:end_idx]

    print(
        f"[Rank {rank}] Processing {len(local_images)} images (batch={BATCH_SIZE}, threads={MAX_THREADS})")

    # Xử lý local images với batch + ThreadPoolExecutor
    batches = list(chunkify(local_images, BATCH_SIZE))
    local_results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_batch, batch, rank)
                   for batch in batches]

        for future in as_completed(futures):
            batch_results = future.result()
            for img_file, cer, wer, elapsed in batch_results:
                local_results.append((img_file, (cer, elapsed, wer)))

    # Gather results to rank 0
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Tổng hợp kết quả
        all_cers = []
        all_times = []
        all_wers = []
        image_order = []
        rank_for_image = []

        for r, rank_results in enumerate(all_results):
            for img, (cer, time_taken, wer) in rank_results:
                all_cers.append(cer)
                all_times.append(time_taken)
                all_wers.append(wer)
                image_order.append(img)
                rank_for_image.append(r)

        # Sắp xếp theo thứ tự ảnh gốc
        sorted_indices = sorted(range(len(image_order)),
                                key=lambda i: image_files.index(image_order[i]))
        all_cers = [all_cers[i] for i in sorted_indices]
        all_times = [all_times[i] for i in sorted_indices]
        all_wers = [all_wers[i] for i in sorted_indices]
        rank_for_image = [rank_for_image[i] for i in sorted_indices]

        colors = ['blue' if r == 0 else 'red' for r in rank_for_image]

        total_time = max(comm.allreduce(time.perf_counter() -
                         time.perf_counter(), op=MPI.MAX))  # Approximate
        avg_cer = sum(all_cers) / len(all_cers)
        avg_wer = sum(all_wers) / len(all_wers)

        print("\n===== KẾT QUẢ MPI SHARED =====")
        print(f"Tổng thời gian: {total_time:.2f} giây")
        print(f"Trung bình CER: {avg_cer:.4f}")
        print(f"Trung bình WER: {avg_wer:.4f}")

        # Lưu kết quả vào file txt
        with open(os.path.join('result', 'mpi_shared_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Tổng thời gian: {total_time:.2f} giây\n")
            f.write(f"Trung bình CER: {avg_cer:.4f}\n")
            f.write(f"Trung bình WER: {avg_wer:.4f}\n")
            f.write("Chi tiết từng ảnh:\n")
            for i, (t, c, w, r) in enumerate(zip(all_times, all_cers, all_wers, rank_for_image)):
                f.write(
                    f"Ảnh {i+1} (Máy {r}): Thời gian {t:.3f}s, CER {c:.4f}, WER {w:.4f}\n")

        # Lưu kết quả vào file CSV
        with open(os.path.join('result', 'mpi_shared_results.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Rank', 'Time', 'CER', 'WER'])
            for i, (t, c, w, r) in enumerate(zip(all_times, all_cers, all_wers, rank_for_image)):
                writer.writerow([f'img_{i+1:04d}.png', r, t, c, w])


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    run_mpi_shared()
