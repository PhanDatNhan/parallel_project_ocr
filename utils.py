import Levenshtein


# ==============================
# Hàm tính tỷ lệ lỗi ký tự (CER)
# ==============================
def calculate_cer(ground_truth, prediction):
    # Tính khoảng cách Levenshtein giữa 2 chuỗi ký tự
    distance = Levenshtein.distance(ground_truth, prediction)
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    return distance / len(ground_truth)


# ==============================
# Hàm tính tỷ lệ lỗi từ (WER)
# ==============================
def calculate_wer(ground_truth, prediction):
    # Tách chuỗi thành danh sách các từ
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    # Levenshtein.distance cũng hoạt động với list các chuỗi (list of strings)
    # Nó sẽ coi mỗi "từ" là một đơn vị tương đương với 1 "ký tự"
    distance = Levenshtein.distance(gt_words, pred_words)
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    return distance / len(gt_words)
