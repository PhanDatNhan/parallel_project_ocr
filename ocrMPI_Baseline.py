from mpi4py import MPI
import time
import os
import cv2
import pytesseract
from utils import calculate_cer, calculate_wer
import dataConfig

# Tối ưu threading và I/O
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Cache toàn cục
image_cache = {}


# ==============================
# Pre-load all images vào RAM (rank 0 pre-load, sau đó broadcast)
# ==============================
def preload_images(image_files, rank):
    global image_cache
    if rank == 0:
        print(f"[Rank 0] Caching {len(image_files)} images vào RAM...")
        start = time.perf_counter()
        for img_file in image_files:
            path = os.path.join(dataConfig.IMAGE_FOLDER, img_file)
            image_cache[img_file] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elapsed = time.perf_counter() - start
        print(f"[Rank 0] ✓ Cached trong {elapsed:.2f}s\n")

    # Broadcast cache (data dict) to all ranks
    image_cache = MPI.COMM_WORLD.bcast(image_cache, root=0)


# ==============================
# Hàm OCR 1 ảnh từ cache
# ==============================
def ocr_single_image_cached(image_file):
    img = image_cache[image_file]
    text = pytesseract.image_to_string(img)
    return text


# ==============================
# Load ground truth
# ==============================
def load_ground_truth(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# Chạy OCR tuần tự trên mỗi rank (tối ưu I/O)
# ==============================
def run_mpi_baseline():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running MPI Baseline with {size} processes")
        image_files = sorted(os.listdir(dataConfig.IMAGE_FOLDER))
    else:
        image_files = None

    # Broadcast image files to all ranks
    image_files = comm.bcast(image_files, root=0)

    # Pre-load images vào cache trước khi xử lý
    preload_images(image_files, rank)

    # Chia đều ảnh cho các rank
    chunk_size = len(image_files) // size
    remainder = len(image_files) % size
    start_idx = rank * chunk_size + min(rank, remainder)
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
    local_images = image_files[start_idx:end_idx]

    print(f"Rank {rank}: processing {len(local_images)} images sequentially")

    # Xử lý local images tuần tự
    local_times = []
    local_cers = []
    local_wers = []
    total_cer = 0
    total_wer = 0
    total_time_start = time.perf_counter()

    for idx, image_file in enumerate(local_images):
        label_path = os.path.join(
            dataConfig.LABEL_FOLDER,
            image_file.replace(".png", ".txt")
        )

        # OCR từ cache
        start = time.perf_counter()
        predicted_text = ocr_single_image_cached(image_file)
        end = time.perf_counter()

        # Ground truth
        ground_truth = load_ground_truth(label_path)

        # CER và WER
        cer = calculate_cer(ground_truth, predicted_text)
        wer = calculate_wer(ground_truth, predicted_text)
        total_cer += cer
        total_wer += wer
        local_cers.append(cer)
        local_wers.append(wer)
        local_times.append(end - start)

        print(
            f"Ảnh {image_file} - Máy {rank}: Process {rank}: Thời gian {end-start:.3f}s, CER {cer:.4f}")

    total_time_end = time.perf_counter()
    local_total_time = total_time_end - total_time_start
    local_avg_cer = total_cer / len(local_images) if local_images else 0
    local_avg_wer = total_wer / len(local_images) if local_images else 0

    # Gather results to rank 0
    all_times = comm.gather(local_times, root=0)
    all_cers = comm.gather(local_cers, root=0)
    all_wers = comm.gather(local_wers, root=0)
    all_total_times = comm.gather(local_total_time, root=0)
    all_avg_cers = comm.gather(local_avg_cer, root=0)
    all_avg_wers = comm.gather(local_avg_wer, root=0)

    if rank == 0:
        # Tổng hợp kết quả
        combined_times = []
        combined_cers = []
        combined_wers = []
        rank_for_image = []
        for r in range(size):
            rank_times = all_times[r]
            rank_cers = all_cers[r]
            rank_wers = all_wers[r]
            combined_times.extend(rank_times)
            combined_cers.extend(rank_cers)
            combined_wers.extend(rank_wers)
            rank_for_image.extend([r] * len(rank_times))

        # Sắp xếp theo thứ tự ảnh gốc
        image_order = []
        for r in range(size):
            chunk_size_r = len(image_files) // size
            remainder_r = len(image_files) % size
            start = r * chunk_size_r + min(r, remainder_r)
            end = start + chunk_size_r + (1 if r < remainder_r else 0)
            image_order.extend(image_files[start:end])

        sorted_indices = sorted(range(len(image_order)),
                                key=lambda i: image_files.index(image_order[i]))
        combined_times = [combined_times[i] for i in sorted_indices]
        combined_cers = [combined_cers[i] for i in sorted_indices]
        combined_wers = [combined_wers[i] for i in sorted_indices]
        rank_for_image = [rank_for_image[i] for i in sorted_indices]

        colors = ['blue' if r == 0 else 'red' for r in rank_for_image]

        # Thời gian tổng là max của các local times
        overall_total_time = max(all_total_times)
        overall_avg_cer = sum(combined_cers) / len(combined_cers)
        overall_avg_wer = sum(combined_wers) / len(combined_wers)

        print("\n===== KẾT QUẢ MPI BASELINE =====")
        print(f"Tổng thời gian: {overall_total_time:.2f} giây")
        print(f"Trung bình CER: {overall_avg_cer:.4f}")
        print(f"Trung bình WER: {overall_avg_wer:.4f}")

        # Lưu kết quả vào file txt
        with open(os.path.join('result', 'mpi_baseline_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Tổng thời gian: {overall_total_time:.2f} giây\n")
            f.write(f"Trung bình CER: {overall_avg_cer:.4f}\n")
            f.write(f"Trung bình WER: {overall_avg_wer:.4f}\n")
            f.write("Chi tiết từng ảnh:\n")
            for i, (t, c, w, r) in enumerate(zip(combined_times, combined_cers, combined_wers, rank_for_image)):
                f.write(
                    f"Ảnh {i+1} (Máy {r}): Thời gian {t:.3f}s, CER {c:.4f}, WER {w:.4f}\n")

        # Lưu kết quả vào file CSV
        import csv
        with open(os.path.join('result', 'mpi_baseline_results.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Rank', 'Time', 'CER', 'WER'])
            for i, (t, c, w, r) in enumerate(zip(combined_times, combined_cers, combined_wers, rank_for_image)):
                writer.writerow([f'img_{i+1:04d}.png', r, t, c, w])

    else:
        # Các rank khác không làm gì
        pass


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    run_mpi_baseline()
