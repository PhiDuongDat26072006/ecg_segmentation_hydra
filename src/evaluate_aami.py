"""
Đánh giá phân đoạn ECG theo chuẩn AAMI — đúng phương pháp của bài báo.

Post-processing (Section 3.7):
  1. Trích xuất phân đoạn liên tiếp từ output model
  2. Lọc nhiễu: loại bỏ đoạn < 40ms (20 mẫu ở 500Hz)
  3. Xác định ranh giới: giữ P và T dài nhất giữa mỗi cặp QRS

Evaluation (Section 4.1):
  - Chuẩn AAMI: tolerance 150ms (75 mẫu ở 500Hz)
  - Đánh giá theo từng rhythm (NSR, ST, BBB, AVB1, AFIB, AFL, VT)
  - Loại bỏ P wave cho AFIB, AFL, VT (bài báo hiển thị "-")
  - Matching: duyệt PREDICTIONS trước, tìm ground truth gần nhất

Usage:
  # Đầy đủ — per-rhythm (cần đường dẫn LUDB):
  python evaluate_aami.py --predictions predictions.npz --data_dir /path/to/ludb/data

  # Đơn giản — dùng cls_true để tách AFIB/AFL:
  python evaluate_aami.py --predictions predictions.npz
"""

import numpy as np
import os
import argparse
from collections import OrderedDict

N_LEADS = 12
BOUNDARY_TYPES = ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset']

# Bài báo: P wave KHÔNG đánh giá cho các rhythm này (hiển thị "-")
P_WAVE_EXCLUDED = {'AFIB', 'AFL', 'VT'}

# Thứ tự hiển thị rhythm (giống bài báo)
RHYTHM_ORDER = ['NSR', 'ST', 'BBB', 'AVB1', 'AFIB', 'AFL', 'VT']


# ============================================================
# Post-processing (Paper Section 3.7)
# ============================================================

def extract_segments(labels):
    """Bước 1: Trích xuất phân đoạn thô từ chuỗi nhãn."""
    segments = []
    if len(labels) == 0:
        return segments

    current_label = labels[0]
    onset = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append({
                'label': current_label,
                'onset': onset,
                'offset': i - 1,
                'length': i - onset
            })
            current_label = labels[i]
            onset = i
    segments.append({
        'label': current_label,
        'onset': onset,
        'offset': len(labels) - 1,
        'length': len(labels) - onset
    })
    return segments


def noise_reduction(segments, min_length=20, baseline_label=3):
    """Bước 2: Lọc nhiễu < 40ms (20 mẫu tại 500Hz) và gán lại nhãn."""
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg['length'] < min_length:
                left_seg = new_segments[-1] if len(new_segments) > 0 else None
                right_seg = segments[i + 1] if i + 1 < len(segments) else None

                left_label = left_seg['label'] if left_seg else None
                right_label = right_seg['label'] if right_seg else None

                if left_seg and right_seg and left_label == right_label:
                    # Gluing: Nếu 2 bên cùng nhãn -> Gộp cả 3 đoạn
                    left_seg = new_segments.pop()
                    merged = {
                        'label': left_label,
                        'onset': left_seg['onset'],
                        'offset': right_seg['offset'],
                        'length': left_seg['length'] + seg['length'] + right_seg['length']
                    }
                    new_segments.append(merged)
                    i += 2
                    changed = True
                    continue
                else:
                    # Xóa bỏ: Biến thành Baseline (3)
                    if seg['label'] != baseline_label:
                        seg['label'] = baseline_label
                        changed = True
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
            i += 1

        # Hợp nhất các đoạn kề nhau có cùng nhãn
        merged_segments = []
        if len(new_segments) > 0:
            current = new_segments[0]
            for j in range(1, len(new_segments)):
                next_seg = new_segments[j]
                if current['label'] == next_seg['label']:
                    current['offset'] = next_seg['offset']
                    current['length'] += next_seg['length']
                else:
                    merged_segments.append(current)
                    current = next_seg
            merged_segments.append(current)
        segments = merged_segments

    return segments


def boundary_determination(segments, p_label=0, qrs_label=1, t_label=2, baseline_label=3):
    """Bước 3: Định vị ranh giới, giữ lại P và T dài nhất giữa 2 QRS."""
    qrs_indices = [i for i, seg in enumerate(segments) if seg['label'] == qrs_label]

    if not qrs_indices:
        return segments

    final_segments = []
    intervals = []
    start_idx = 0
    for qrs_idx in qrs_indices:
        intervals.append(segments[start_idx:qrs_idx])
        intervals.append([segments[qrs_idx]])
        start_idx = qrs_idx + 1
    intervals.append(segments[start_idx:])

    for interval in intervals:
        if not interval:
            continue
        if interval[0]['label'] == qrs_label:
            final_segments.append(interval[0])
            continue

        p_waves = [seg for seg in interval if seg['label'] == p_label]
        t_waves = [seg for seg in interval if seg['label'] == t_label]

        longest_p = max(p_waves, key=lambda x: x['length']) if p_waves else None
        longest_t = max(t_waves, key=lambda x: x['length']) if t_waves else None

        for seg in interval:
            if seg['label'] == p_label and seg != longest_p:
                seg['label'] = baseline_label
            elif seg['label'] == t_label and seg != longest_t:
                seg['label'] = baseline_label
            final_segments.append(seg)

    # Hợp nhất lại
    merged_segments = []
    if len(final_segments) > 0:
        current = final_segments[0]
        for j in range(1, len(final_segments)):
            next_seg = final_segments[j]
            if current['label'] == next_seg['label']:
                current['offset'] = next_seg['offset']
                current['length'] += next_seg['length']
            else:
                merged_segments.append(current)
                current = next_seg
        merged_segments.append(current)

    return merged_segments


def extract_boundaries(segments):
    """Lấy danh sách các điểm Onset/Offset cho từng nhãn."""
    boundaries = {
        'P_onset': [], 'P_offset': [],
        'QRS_onset': [], 'QRS_offset': [],
        'T_onset': [], 'T_offset': []
    }
    for seg in segments:
        if seg['label'] == 0:
            boundaries['P_onset'].append(seg['onset'])
            boundaries['P_offset'].append(seg['offset'])
        elif seg['label'] == 1:
            boundaries['QRS_onset'].append(seg['onset'])
            boundaries['QRS_offset'].append(seg['offset'])
        elif seg['label'] == 2:
            boundaries['T_onset'].append(seg['onset'])
            boundaries['T_offset'].append(seg['offset'])
    return boundaries


# ============================================================
# Evaluation Matching (Paper Section 4.1)
# ============================================================

def evaluate_aami_single_type(pred_b, true_b, tolerance=75):
    """
    So khớp AAMI: duyệt qua từng PREDICTION, tìm ground truth gần nhất.

    Paper: "we examine for each predicted point whether the prediction
    correctly detects a point in the ground truth annotation"
    """
    tp = 0
    errors = []
    matched_gt = set()

    for pb in pred_b:
        closest_idx = None
        min_dist = float('inf')
        for i, tb in enumerate(true_b):
            if i in matched_gt:
                continue
            dist = abs(pb - tb)
            if dist <= tolerance and dist < min_dist:
                min_dist = dist
                closest_idx = i

        if closest_idx is not None:
            tp += 1
            errors.append(pb - true_b[closest_idx])  # error = pred - GT
            matched_gt.add(closest_idx)

    fp = len(pred_b) - tp   # predictions không match
    fn = len(true_b) - tp   # ground truths không match

    return tp, fp, fn, errors


# ============================================================
# Rhythm Labeling từ LUDB
# ============================================================

def _extract_field(comment):
    """Trích xuất giá trị từ chuỗi 'Key: Value' hoặc 'Key: Value.'"""
    if ': ' in comment:
        value = comment.split(': ', 1)[1]
        return value.rstrip('.')
    return comment


def _categorize(rhythm, diagnosis):
    """Phân loại bản ghi LUDB vào 7 nhóm rhythm của bài báo."""
    r = rhythm.lower()
    d = diagnosis.lower()

    if 'atrial fibrillation' in r:
        return 'AFIB'
    if 'atrial flutter' in r:
        return 'AFL'
    if 'sinus tachycardia' in r:
        return 'ST'
    # VT thường nằm trong rhythm hoặc diagnosis
    if 'ventricular tachycardia' in r or 'ventricular tachycardia' in d:
        return 'VT'
    # BBB: bundle branch block (left hoặc right)
    if any(x in d for x in ['bundle branch block', 'lbbb', 'rbbb',
                              'left bundle', 'right bundle',
                              'incomplete right bundle', 'incomplete left bundle']):
        return 'BBB'
    # AVB1: AV block degree 1
    if any(x in d for x in ['av block', 'atrioventricular block', '1 degree',
                              '1st degree', 'first degree']):
        return 'AVB1'
    return 'NSR'


def get_test_rhythm_labels(data_dir, n_test_records):
    """
    Đọc bản ghi LUDB từ data_dir, lấy n_test_records bản ghi cuối cùng làm test set.
    Trả về: list[str] — rhythm category cho mỗi bản ghi test.
    """
    import wfdb

    hea_files = sorted([p for p in os.listdir(data_dir) if p.endswith('.hea')])

    if n_test_records >= len(hea_files):
        test_files = hea_files
    else:
        test_files = hea_files[-n_test_records:]

    rhythms = []
    print(f"\n{'='*60}")
    print(f"Đọc nhãn rhythm từ {len(test_files)} bản ghi test trong LUDB")
    print(f"{'='*60}")

    for f in test_files:
        record_path = os.path.abspath(os.path.join(data_dir, f))[:-4]
        record = wfdb.rdrecord(record_path)
        comments = record.__dict__['comments']

        # Trích xuất rhythm và diagnosis từ comments
        rhythm_raw = ''
        diag_raw = ''
        for c in comments:
            c_lower = c.lower()
            if c_lower.startswith('rhythm'):
                rhythm_raw = _extract_field(c)
            elif c_lower.startswith('diagnos'):
                diag_raw = _extract_field(c)

        category = _categorize(rhythm_raw, diag_raw)
        rhythms.append(category)
        print(f"  {f}: rhythm='{rhythm_raw}', diagnos='{diag_raw}' -> {category}")

    print(f"{'='*60}")

    # Thống kê
    from collections import Counter
    counts = Counter(rhythms)
    print(f"\nPhân bố rhythm trong test set:")
    for r in RHYTHM_ORDER:
        if r in counts:
            print(f"  {r}: {counts[r]} bản ghi ({counts[r] * N_LEADS} tín hiệu)")

    return rhythms


# ============================================================
# Dice Score & Accuracy
# ============================================================

def calculate_accuracy_and_dice(seg_true, seg_pred, cls_true, cls_pred):
    """Tính Accuracy tổng thể và Dice Score cho từng class."""
    cls_acc = np.mean(cls_pred == cls_true)
    seg_acc = np.mean(seg_pred == seg_true)

    print("\n" + "=" * 50)
    print("PIXEL-WISE ACCURACY & DICE SCORE")
    print("=" * 50)
    print(f"Classification Accuracy    : {cls_acc:.4f}")
    print(f"Segmentation Pixel Accuracy: {seg_acc:.4f}")
    print("-" * 50)

    class_names = ['P', 'QRS', 'T', 'Baseline']
    for c in range(4):
        p_c = (seg_pred == c)
        t_c = (seg_true == c)
        intersection = np.sum(p_c & t_c)
        union = np.sum(p_c) + np.sum(t_c)
        dice = 2.0 * intersection / union if union > 0 else 1.0
        print(f'Dice Score ({class_names[c]:>8s}): {dice:.4f}')
    print("=" * 50 + "\n")


# ============================================================
# Per-rhythm Evaluation
# ============================================================

def evaluate_per_rhythm(seg_true_all, seg_pred_all, lead_rhythms, tolerance=75):
    """
    Đánh giá AAMI theo từng rhythm — đúng phương pháp bài báo.

    Args:
        seg_true_all: (N, L) — nhãn phân đoạn ground truth
        seg_pred_all: (N, L) — nhãn phân đoạn dự đoán
        lead_rhythms: list[str] — rhythm category cho mỗi tín hiệu (lead)
        tolerance: int — ngưỡng AAMI tính bằng số mẫu

    Returns:
        results: dict[rhythm -> dict[btype -> metrics]]
    """
    all_rhythms = sorted(set(lead_rhythms), key=lambda x: RHYTHM_ORDER.index(x)
                         if x in RHYTHM_ORDER else len(RHYTHM_ORDER))

    results = OrderedDict()

    for rhythm in all_rhythms:
        indices = [i for i, r in enumerate(lead_rhythms) if r == rhythm]
        rhythm_metrics = OrderedDict()

        for btype in BOUNDARY_TYPES:
            # Bỏ qua P wave cho các rhythm mà bài báo hiển thị "-"
            if btype.startswith('P_') and rhythm in P_WAVE_EXCLUDED:
                rhythm_metrics[btype] = None
                continue

            total_tp, total_fp, total_fn = 0, 0, 0
            all_errors = []

            for i in indices:
                # Ground truth boundaries (không cần post-processing)
                t_segs = extract_segments(seg_true_all[i])
                t_b = extract_boundaries(t_segs)

                # Predicted boundaries (với post-processing theo Section 3.7)
                p_segs = extract_segments(seg_pred_all[i])
                p_segs = noise_reduction(p_segs, min_length=20)
                p_segs = boundary_determination(p_segs)
                p_b = extract_boundaries(p_segs)

                tp, fp, fn, errors = evaluate_aami_single_type(
                    p_b[btype], t_b[btype], tolerance
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_errors.extend(errors)

            # Tính F1 (micro-average trong cùng 1 rhythm)
            denom = 2 * total_tp + total_fp + total_fn
            f1 = 2 * total_tp / denom if denom > 0 else 0.0

            # Tần số 500Hz -> 1 sample = 2ms
            mean_err = np.mean(all_errors) * 2 if all_errors else 0.0
            std_err = np.std(all_errors) * 2 if all_errors else 0.0

            rhythm_metrics[btype] = {
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
                'f1': f1, 'mean_err': mean_err, 'std_err': std_err,
                'n_signals': len(indices)
            }

        results[rhythm] = rhythm_metrics

    return results


# ============================================================
# Báo cáo kết quả
# ============================================================

def print_f1_table(results):
    """In bảng F1-scores theo từng rhythm — giống format bài báo."""
    W = 105
    print("\n" + "=" * W)
    print("F1-SCORES (%) PER RHYTHM — Paper Table Format")
    print("=" * W)

    # Header
    header = f"{'Rhythm':<10}"
    for btype in BOUNDARY_TYPES:
        name = btype.replace('_', ' ')
        header += f" | {name:>12}"
    print(header)
    print("-" * W)

    # Per-rhythm rows
    for rhythm, metrics in results.items():
        row = f"{rhythm:<10}"
        for btype in BOUNDARY_TYPES:
            m = metrics[btype]
            if m is None:
                row += f" | {'   -':>12}"
            else:
                row += f" | {m['f1'] * 100:>11.2f}%"
        print(row)

    print("-" * W)

    # Macro-average row (trung bình F1 theo rhythm, bỏ qua None)
    row_macro = f"{'All(macro)':<10}"
    for btype in BOUNDARY_TYPES:
        f1_values = [results[r][btype]['f1'] for r in results
                     if results[r][btype] is not None]
        if f1_values:
            avg = np.mean(f1_values) * 100
            row_macro += f" | {avg:>11.2f}%"
        else:
            row_macro += f" | {'   -':>12}"
    print(row_macro)

    # Micro-average row (gộp TP/FP/FN qua tất cả rhythm, bỏ qua excluded)
    row_micro = f"{'All(micro)':<10}"
    for btype in BOUNDARY_TYPES:
        total_tp = sum(results[r][btype]['tp'] for r in results
                       if results[r][btype] is not None)
        total_fp = sum(results[r][btype]['fp'] for r in results
                       if results[r][btype] is not None)
        total_fn = sum(results[r][btype]['fn'] for r in results
                       if results[r][btype] is not None)
        denom = 2 * total_tp + total_fp + total_fn
        f1 = 2 * total_tp / denom if denom > 0 else 0.0
        row_micro += f" | {f1 * 100:>11.2f}%"
    print(row_micro)

    print("=" * W)


def print_detailed_table(results):
    """In bảng chi tiết TP/FP/FN/F1/Error cho từng rhythm và boundary."""
    W = 110
    print("\n" + "=" * W)
    print("DETAILED AAMI 150ms EVALUATION (Tolerance: 75 samples)")
    print("=" * W)

    for rhythm, metrics in results.items():
        n_sig = None
        print(f"\n--- Rhythm: {rhythm} ---")
        header = (f"{'Boundary':<15} | {'TP':<6} | {'FP':<6} | {'FN':<6} | "
                  f"{'F1-Score':<10} | {'Mean Err(ms)':<14} | {'Std Err(ms)':<14}")
        print(header)
        print("-" * W)

def predict_from_pth(pth_path, data_dir):
    import sys
    import torch
    import numpy as np

    # Cần nạp file model.py và datareader.py từ thư mục gốc ecg-segmentation-main
    old_code_dir = r"C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\NCKH\Điện tim\ecg-MI-classification-code\ecg-segmentation\ecg-segmentation-main"
    if old_code_dir not in sys.path:
        sys.path.insert(0, old_code_dir)

    import model as old_model
    from datareader import load_ludb_tensors
    from torch.utils.data import TensorDataset, DataLoader
    import os

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục LUDB data: {data_dir}. Cần thư mục này để load test set!")

    ludb_files = [os.path.abspath(os.path.join(data_dir, p))[:-4] for p in os.listdir(data_dir) if p.endswith('.hea')]
    n_ludb_train = 180
    ludb_files_test = ludb_files[n_ludb_train:]

    print(f"Đang nạp tập test LUDB ({len(ludb_files_test)} bản ghi)... (khoảng 15-30 giây)")
    X_test, y_seg_test, y_cls_test = load_ludb_tensors(ludb_files_test)
    test_loader = DataLoader(TensorDataset(X_test, y_seg_test, y_cls_test), batch_size=16, shuffle=False)

    print(f"Đang khởi tạo mô hình gốc từ {pth_path}...")
    checkpoint = torch.load(pth_path, map_location='cpu')
    n_channels = checkpoint.get('n_channels', 32)

    net = old_model.ECGUNet3pCGM(n_channels=n_channels)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()

    all_seg_pred = []
    all_cls_pred = []

    print(f"Đang chạy Inference bằng {device}...")
    with torch.no_grad():
        for batch_idx, (data, _, _) in enumerate(test_loader):
            data = data.to(device)
            seg_output, cls_output = net(data)

            seg_pred = torch.argmax(seg_output, dim=1).cpu().numpy()
            cls_pred = torch.argmax(cls_output, dim=1).cpu().numpy()

            all_seg_pred.append(seg_pred)
            all_cls_pred.append(cls_pred)

    seg_pred_all = np.concatenate(all_seg_pred, axis=0)
    cls_pred_all = np.concatenate(all_cls_pred, axis=0)

    seg_true_all = torch.argmax(y_seg_test, dim=1).numpy()
    cls_true_all = y_cls_test.numpy()

    return seg_true_all, seg_pred_all, cls_true_all, cls_pred_all


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Đánh giá phân đoạn ECG theo chuẩn AAMI — phương pháp bài báo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--predictions', type=str, default='predictions.npz',
                        help='Đường dẫn file predictions.npz HOẶC final_model.pth')
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\NCKH\Điện tim\ecg-MI-classification-code\ecg-segmentation\segmentation_data\lobachevsky-university-electrocardiography-database-1.0.1\data',
                        help='Đường dẫn LUDB data (để lấy nhãn rhythm per-record). ')
    parser.add_argument('--n_train', type=int, default=180,
                        help='Số bản ghi dùng để train (để xác định test set)')
    parser.add_argument('--tolerance', type=int, default=75,
                        help='Ngưỡng AAMI tính bằng số mẫu (75 = 150ms ở 500Hz)')
    args = parser.parse_args()

    # ---- Nạp dữ liệu ----
    if not os.path.exists(args.predictions):
        print(f"Không tìm thấy file {args.predictions}!")
        return

    if args.predictions.endswith('.pth'):
        print(f"\nPhát hiện file .pth, tự động đọc mô hình cũ và suy luận trên test set...")
        seg_true_all, seg_pred_all, cls_true_all, cls_pred_all = predict_from_pth(args.predictions, args.data_dir)
    elif args.predictions.endswith('.npz'):
        print(f"Đang nạp {args.predictions}...")
        data = np.load(args.predictions)
        seg_true_all = data['seg_true']
        seg_pred_all = data['seg_pred']
        cls_true_all = data.get('cls_true', None)
        cls_pred_all = data.get('cls_pred', None)
    else:
        print("Định dạng file không được hỗ trợ. Vui lòng cung cấp file .npz hoặc .pth!")
        return

    n_signals = len(seg_true_all)
    n_records = n_signals // N_LEADS
    print(f"Số tín hiệu: {n_signals} ({n_records} bản ghi x {N_LEADS} chuyển đạo)")

    # ---- Dice Score & Accuracy ----
    if cls_true_all is not None and cls_pred_all is not None:
        calculate_accuracy_and_dice(seg_true_all, seg_pred_all, cls_true_all, cls_pred_all)

    # ---- Xác định nhãn rhythm cho mỗi tín hiệu ----
    if args.data_dir is not None:
        # Đầy đủ: đọc từ LUDB để lấy 7 nhóm rhythm
        record_rhythms = get_test_rhythm_labels(args.data_dir, n_records)
        # Mở rộng: mỗi bản ghi có 12 chuyển đạo
        lead_rhythms = [r for r in record_rhythms for _ in range(N_LEADS)]
    else:
        # Fallback: dùng cls_true (0=non-AFIB, 1=AFIB/AFL)
        print("\n⚠ Không có --data_dir: dùng cls_true để phân AFIB/AFL.")
        print("  Để đánh giá đầy đủ per-rhythm, hãy cung cấp --data_dir.\n")

        if cls_true_all is not None:
            lead_rhythms = []
            for i in range(n_signals):
                if cls_true_all[i] == 1:
                    lead_rhythms.append('AFIB')  # Gộp AFIB + AFL
                else:
                    lead_rhythms.append('NSR')   # Gộp tất cả non-AFIB/AFL
        else:
            # Không có rhythm info -> đánh giá tất cả như 1 nhóm
            lead_rhythms = ['ALL'] * n_signals

    assert len(lead_rhythms) == n_signals, \
        f"Số nhãn rhythm ({len(lead_rhythms)}) != số tín hiệu ({n_signals})"

    # ---- Đánh giá per-rhythm ----
    print("\nĐang đánh giá theo từng rhythm...")
    results = evaluate_per_rhythm(seg_true_all, seg_pred_all, lead_rhythms, args.tolerance)

    # ---- In báo cáo ----
    print_f1_table(results)
    print_detailed_table(results)

    print("\nEND")


if __name__ == "__main__":
    main()
