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
  python evaluate_aami.py --predictions predictions.npz --data_dir /path/to/ludb/data
  python evaluate_aami.py --predictions final_model.pth --data_dir /path/to/ludb/data
  python evaluate_aami.py --predictions model.ckpt --data_dir /path/to/ludb/data
"""

import numpy as np
import os
import sys
import argparse
import torch
import hydra
from omegaconf import OmegaConf
from collections import OrderedDict, Counter
from models.components.post_processing import *
from data.components.datareader import load_ludb_tensors
# ---- Hằng số ----
N_LEADS = 12
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
EVAL_LEAD_INDICES = [0, 1, 2, 3, 4, 5, 6 , 7, 8, 9, 10, 11]  # Chỉ đánh giá Lead I và Lead II
BOUNDARY_TYPES = ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset']
P_WAVE_EXCLUDED = {'AFIB', 'AFL', 'VT'}
RHYTHM_ORDER = ['NSR', 'ST', 'BBB', 'AVB1', 'AFIB', 'AFL', 'VT']
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ============================================================
# Evaluation Matching (Paper Section 4.1)
# ============================================================

def evaluate_aami_single_type(pred_b, true_b, tolerance=75):
    """So khớp AAMI: duyệt qua từng PREDICTION, tìm ground truth gần nhất."""
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
            errors.append(pb - true_b[closest_idx])
            matched_gt.add(closest_idx)

    fp = len(pred_b) - tp
    fn = len(true_b) - tp
    return tp, fp, fn, errors


# ============================================================
# Rhythm Labeling từ LUDB
# ============================================================

def _extract_field(comment):
    """Trích xuất giá trị từ chuỗi 'Key: Value'."""
    if ': ' in comment:
        return comment.split(': ', 1)[1].rstrip('.')
    return comment


def _categorize(rhythm, diagnosis):
    """Phân loại bản ghi LUDB vào 7 nhóm rhythm của bài báo."""
    r, d = rhythm.lower(), diagnosis.lower()

    if 'atrial fibrillation' in r:
        return 'AFIB'
    if 'atrial flutter' in r:
        return 'AFL'
    if 'sinus tachycardia' in r:
        return 'ST'
    if 'ventricular tachycardia' in r or 'ventricular tachycardia' in d:
        return 'VT'
    if any(x in d for x in ['bundle branch block', 'lbbb', 'rbbb',
                              'left bundle', 'right bundle',
                              'incomplete right bundle', 'incomplete left bundle']):
        return 'BBB'
    if any(x in d for x in ['av block', 'atrioventricular block', '1 degree',
                              '1st degree', 'first degree']):
        return 'AVB1'
    return 'NSR'


def get_test_rhythm_labels(data_dir, n_test_records):
    """Đọc bản ghi LUDB, lấy n_test_records cuối cùng làm test set."""
    import wfdb

    hea_files = sorted([p for p in os.listdir(data_dir) if p.endswith('.hea')])
    test_files = hea_files[-n_test_records:] if n_test_records < len(hea_files) else hea_files

    rhythms = []
    print(f"\n{'=' * 60}")
    print(f"Đọc nhãn rhythm từ {len(test_files)} bản ghi test trong LUDB")
    print(f"{'=' * 60}")

    for f in test_files:
        record_path = os.path.abspath(os.path.join(data_dir, f))[:-4]
        record = wfdb.rdrecord(record_path)
        comments = record.__dict__['comments']

        rhythm_raw, diag_raw = '', ''
        for c in comments:
            c_lower = c.lower()
            if c_lower.startswith('rhythm'):
                rhythm_raw = _extract_field(c)
            elif c_lower.startswith('diagnos'):
                diag_raw = _extract_field(c)

        category = _categorize(rhythm_raw, diag_raw)
        rhythms.append(category)
        print(f"  {f}: rhythm='{rhythm_raw}', diagnos='{diag_raw}' -> {category}")

    print(f"{'=' * 60}")

    counts = Counter(rhythms)
    print(f"\nPhân bố rhythm trong test set:")
    for r in RHYTHM_ORDER:
        if r in counts:
            print(f"  {r}: {counts[r]} bản ghi ({counts[r] * len(EVAL_LEAD_INDICES)} tín hiệu)")

    return rhythms


# ============================================================
# Dice Score & Accuracy
# ============================================================

def calculate_accuracy_and_dice(seg_true, seg_pred, cls_true, cls_pred):
    """Tính Accuracy tổng thể và Dice Score cho từng class.

    Chỉ tính trên vùng annotated (bỏ baseline đầu/cuối bản ghi).
    """
    # Trim từng signal theo vùng annotated
    trimmed_true = []
    trimmed_pred = []
    for i in range(len(seg_true)):
        ann_start, ann_end = find_annotated_range(seg_true[i])
        trimmed_true.append(seg_true[i][ann_start:ann_end + 1])
        trimmed_pred.append(seg_pred[i][ann_start:ann_end + 1])
    seg_true_trimmed = np.concatenate(trimmed_true)
    seg_pred_trimmed = np.concatenate(trimmed_pred)

    cls_acc = np.mean(cls_pred == cls_true)
    seg_acc = np.mean(seg_pred_trimmed == seg_true_trimmed)

    print("\n" + "=" * 50)
    print("PIXEL-WISE ACCURACY & DICE SCORE")
    print("=" * 50)
    print(f"Classification Accuracy    : {cls_acc:.4f}")
    print(f"Segmentation Pixel Accuracy: {seg_acc:.4f}")
    print("-" * 50)

    class_names = ['P', 'QRS', 'T', 'Baseline']
    for c in range(4):
        p_c = (seg_pred_trimmed == c)
        t_c = (seg_true_trimmed == c)
        intersection = np.sum(p_c & t_c)
        union = np.sum(p_c) + np.sum(t_c)
        dice = 2.0 * intersection / union if union > 0 else 1.0
        print(f'Dice Score ({class_names[c]:>8s}): {dice:.4f}')
    print("=" * 50 + "\n")


# ============================================================
# Per-rhythm Evaluation
# ============================================================
# ============================================================
# Annotated-range Detection
# ============================================================

def find_annotated_range(labels, baseline_label=3):
    """Tìm vùng mà bác sĩ thực sự gán nhãn (bỏ qua baseline ở đầu/cuối).

    LUDB: bác sĩ thường không gán nhãn ~1000 mẫu đầu và cuối bản ghi,
    vùng đó mặc định thành baseline. Model dự đoán P/QRS/T ở đó
    → FP giả nếu không trim.

    Returns:
        (start, end): chỉ số mẫu đầu và cuối của vùng annotated (inclusive).
                      Nếu toàn baseline → (0, len-1).
    """
    non_baseline = np.where(labels != baseline_label)[0]
    if len(non_baseline) == 0:
        return 0, len(labels) - 1
    return int(non_baseline[0]), int(non_baseline[-1])

def extract_boundaries(segments, BOUNDARY_TYPES):
    """Lấy danh sách các điểm Onset/Offset cho từng nhãn."""
    label_map = {0: 'P', 1: 'QRS', 2: 'T'}
    boundaries = {bt: [] for bt in BOUNDARY_TYPES}
    for seg in segments:
        prefix = label_map.get(seg['label'])
        if prefix:
            boundaries[f'{prefix}_onset'].append(seg['onset'])
            boundaries[f'{prefix}_offset'].append(seg['offset'])
    return boundaries

def evaluate_per_rhythm(seg_true_all, seg_pred_all, lead_rhythms, tolerance=75):
    """Đánh giá AAMI theo từng rhythm — đúng phương pháp bài báo."""
    all_rhythms = sorted(set(lead_rhythms), key=lambda x: RHYTHM_ORDER.index(x)
                         if x in RHYTHM_ORDER else len(RHYTHM_ORDER))
    results = OrderedDict()

    for rhythm in all_rhythms:
        indices = [i for i, r in enumerate(lead_rhythms) if r == rhythm]
        rhythm_metrics = OrderedDict()

        for btype in BOUNDARY_TYPES:
            if btype.startswith('P_') and rhythm in P_WAVE_EXCLUDED:
                rhythm_metrics[btype] = None
                continue

            total_tp, total_fp, total_fn = 0, 0, 0
            all_errors = []

            for i in indices:
                gt_labels = seg_true_all[i]
                pred_labels = seg_pred_all[i]

                # Trim theo vùng annotated của GT (bỏ đầu/cuối không gán nhãn)
                ann_start, ann_end = find_annotated_range(gt_labels)
                gt_trimmed = gt_labels[ann_start:ann_end + 1]
                pred_trimmed = pred_labels[ann_start:ann_end + 1]

                # Ground truth (không cần post-processing)
                t_b = extract_boundaries(extract_segments(gt_trimmed), BOUNDARY_TYPES)

                # Predicted (với post-processing theo Section 3.7)
                p_segs = post_processing(pred_trimmed)

                p_b = extract_boundaries(p_segs, BOUNDARY_TYPES)

                tp, fp, fn, errors = evaluate_aami_single_type(
                    p_b[btype], t_b[btype], tolerance
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_errors.extend(errors)

            denom = 2 * total_tp + total_fp + total_fn
            f1 = 2 * total_tp / denom if denom > 0 else 0.0
            mean_err = np.mean(all_errors) * 2 if all_errors else 0.0   # 500Hz → 2ms/sample
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

    header = f"{'Rhythm':<10}"
    for btype in BOUNDARY_TYPES:
        header += f" | {btype.replace('_', ' '):>12}"
    print(header)
    print("-" * W)

    for rhythm, metrics in results.items():
        row = f"{rhythm:<10}"
        for btype in BOUNDARY_TYPES:
            m = metrics[btype]
            row += f" | {'   -':>12}" if m is None else f" | {m['f1'] * 100:>11.2f}%"
        print(row)

    print("-" * W)

    # Macro-average
    row = f"{'All(macro)':<10}"
    for btype in BOUNDARY_TYPES:
        vals = [results[r][btype]['f1'] for r in results if results[r][btype] is not None]
        row += f" | {np.mean(vals) * 100:>11.2f}%" if vals else f" | {'   -':>12}"
    print(row)

    # Micro-average
    row = f"{'All(micro)':<10}"
    for btype in BOUNDARY_TYPES:
        tp = sum(results[r][btype]['tp'] for r in results if results[r][btype] is not None)
        fp = sum(results[r][btype]['fp'] for r in results if results[r][btype] is not None)
        fn = sum(results[r][btype]['fn'] for r in results if results[r][btype] is not None)
        denom = 2 * tp + fp + fn
        row += f" | {2 * tp / denom * 100 if denom > 0 else 0:>11.2f}%"
    print(row)

    print("=" * W)


def print_detailed_table(results):
    """In bảng chi tiết TP/FP/FN/F1/Error cho từng rhythm và boundary."""
    W = 110
    print("\n" + "=" * W)
    print("DETAILED AAMI 150ms EVALUATION (Tolerance: 75 samples)")
    print("=" * W)

    for rhythm, metrics in results.items():
        print(f"\n--- Rhythm: {rhythm} ---")
        header = (f"{'Boundary':<15} | {'TP':<6} | {'FP':<6} | {'FN':<6} | "
                  f"{'F1-Score':<10} | {'Mean Err(ms)':<14} | {'Std Err(ms)':<14}")
        print(header)
        print("-" * W)
        for btype in BOUNDARY_TYPES:
            m = metrics[btype]
            if m is None:
                print(f"{btype:<15} | {'-':<6} | {'-':<6} | {'-':<6} | "
                      f"{'-':<10} | {'-':<14} | {'-':<14}")
            else:
                print(f"{btype:<15} | {m['tp']:<6} | {m['fp']:<6} | {m['fn']:<6} | "
                      f"{m['f1'] * 100:.2f}%{'':4s} | {m['mean_err']:<14.2f} | {m['std_err']:<14.2f}")
        print("-" * W)


# ============================================================
# Model Loading & Inference Helpers
# ============================================================

def _load_test_data(data_dir, n_train=180):
    """Nạp test set từ LUDB (dùng chung cho cả .pth và .ckpt)."""
    from torch.utils.data import TensorDataset, DataLoader

    # Thêm project root vào sys.path nếu chưa có
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục LUDB data: {data_dir}")

    ludb_files = sorted([
        os.path.abspath(os.path.join(data_dir, p))[:-4]
        for p in os.listdir(data_dir) if p.endswith('.hea')
    ])
    ludb_files_test = ludb_files[n_train:]

    print(f"Đang nạp tập test LUDB ({len(ludb_files_test)} bản ghi)...")
    X_test, y_seg_test, y_cls_test = load_ludb_tensors(ludb_files_test)
    loader = DataLoader(TensorDataset(X_test, y_seg_test, y_cls_test), batch_size=16, shuffle=False)

    seg_true = torch.argmax(y_seg_test, dim=1).numpy()
    cls_true = y_cls_test.numpy()

    return loader, seg_true, cls_true

def _run_inference(net, loader):
    """Chạy inference trên DataLoader, trả về seg_pred và cls_pred."""
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()

    all_seg, all_cls = [], []
    print(f"Đang chạy Inference bằng {device}...")
    with torch.no_grad():
        for data, _, _ in loader:
            data = data.to(device)
            seg_out, cls_out = net(data)
            all_seg.append(torch.argmax(seg_out, dim=1).cpu().numpy())
            all_cls.append(torch.argmax(cls_out, dim=1).cpu().numpy())

    return np.concatenate(all_seg, axis=0), np.concatenate(all_cls, axis=0)


def build_ecg_unet3p_cgm(mask=True):
    """Khởi tạo mạng bằng cách đọc cấu hình từ file YAML thông qua Hydra."""
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'model', 'ECGUnet3pCGM.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    # Nạp file config YAML
    cfg = OmegaConf.load(config_path)

    # Dùng Hydra instantiate để tự động khởi tạo toàn bộ Class từ định nghĩa yaml
    net = hydra.utils.instantiate(cfg.net)

    # Ghi đè tham số mask nếu cần
    net.mask = mask
    return net

def predict_from_pth(pth_path, data_dir, n_train):
    """Nạp mô hình cũ (.pth) và chạy inference."""
    loader, seg_true, cls_true = _load_test_data(data_dir, n_train)

    print(f"Đang khởi tạo mô hình cũ từ {pth_path}...")
    checkpoint = torch.load(pth_path, map_location='cpu')

    net = build_ecg_unet3p_cgm()
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(state_dict)

    seg_pred, cls_pred = _run_inference(net, loader)
    return seg_true, seg_pred, cls_true, cls_pred


def predict_from_ckpt(ckpt_path, data_dir, n_train):
    """Nạp mô hình Lightning (.ckpt) và chạy inference."""
    loader, seg_true, cls_true = _load_test_data(data_dir, n_train)

    print(f"Đang đọc checkpoint từ {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    net = build_ecg_unet3p_cgm()

    # Bỏ prefix 'net.' nếu lưu từ LightningModule
    state_dict = checkpoint['state_dict']
    cleaned = {(k[4:] if k.startswith('net.') else k): v for k, v in state_dict.items()}
    net.load_state_dict(cleaned)

    seg_pred, cls_pred = _run_inference(net, loader)
    return seg_true, seg_pred, cls_true, cls_pred


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Đánh giá phân đoạn ECG theo chuẩn AAMI.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--predictions', type=str, default='final_model_180.pth',
                        help='Đường dẫn file .npz, .pth hoặc .ckpt')
    parser.add_argument('--data_dir', type=str,
                        default=(r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice'
                                 r'\NCKH\Điện tim\ecg-MI-classification-code\ecg-segmentation'
                                 r'\segmentation_data\lobachevsky-university-electrocardiography-'
                                 r'database-1.0.1\data'),
                        help='Đường dẫn LUDB data')
    parser.add_argument('--n_train', type=int, default=180,
                        help='Số bản ghi dùng để train (để xác định test set)')
    parser.add_argument('--tolerance', type=int, default=75,
                        help='Ngưỡng AAMI tính bằng số mẫu (75 = 150ms ở 500Hz)')
    args = parser.parse_args()

    # ---- Nạp dữ liệu ----
    if not os.path.exists(args.predictions):
        print(f"Không tìm thấy file {args.predictions}!")
        return

    ext = os.path.splitext(args.predictions)[1].lower()
    if ext == '.ckpt':
        print(f"\nPhát hiện file .ckpt → suy luận trên test set...")
        seg_true_all, seg_pred_all, cls_true_all, cls_pred_all = predict_from_ckpt(args.predictions, args.data_dir, n_train = args.n_train)
    elif ext == '.pth':
        print(f"\nPhát hiện file .pth → suy luận trên test set...")
        seg_true_all, seg_pred_all, cls_true_all, cls_pred_all = predict_from_pth(args.predictions, args.data_dir, n_train = args.n_train)
    elif ext == '.npz':
        print(f"Đang nạp {args.predictions}...")
        data = np.load(args.predictions)
        seg_true_all = data['seg_true']
        seg_pred_all = data['seg_pred']
        cls_true_all = data.get('cls_true', None)
        cls_pred_all = data.get('cls_pred', None)
    else:
        print("Định dạng file không được hỗ trợ! Hãy dùng .npz, .pth hoặc .ckpt.")
        return

    n_total = len(seg_true_all)
    n_records = n_total // N_LEADS
    eval_leads = EVAL_LEAD_INDICES
    eval_lead_names = [LEAD_NAMES[i] for i in eval_leads]
    print(f"Tổng tín hiệu: {n_total} ({n_records} bản ghi x {N_LEADS} chuyển đạo)")
    print(f"→ Chỉ đánh giá trên Lead: {', '.join(eval_lead_names)}")

    # ---- Lọc chỉ giữ Lead I và II ----
    keep_indices = []
    for rec in range(n_records):
        for lead_idx in eval_leads:
            keep_indices.append(rec * N_LEADS + lead_idx)
    keep_indices = np.array(keep_indices)

    seg_true_all = seg_true_all[keep_indices]
    seg_pred_all = seg_pred_all[keep_indices]
    if cls_true_all is not None:
        cls_true_all = cls_true_all[keep_indices]
    if cls_pred_all is not None:
        cls_pred_all = cls_pred_all[keep_indices]

    n_signals = len(seg_true_all)
    print(f"Số tín hiệu sau lọc: {n_signals} ({n_records} bản ghi x {len(eval_leads)} chuyển đạo)")

    # ---- Dice Score & Accuracy ----
    if cls_true_all is not None and cls_pred_all is not None:
        calculate_accuracy_and_dice(seg_true_all, seg_pred_all, cls_true_all, cls_pred_all)

    # ---- Xác định nhãn rhythm ----
    if args.data_dir is not None:
        record_rhythms = get_test_rhythm_labels(args.data_dir, n_records)
        lead_rhythms = [r for r in record_rhythms for _ in range(len(eval_leads))]
    else:
        print("\n⚠ Không có --data_dir: dùng cls_true để phân AFIB/AFL.")
        if cls_true_all is not None:
            lead_rhythms = ['AFIB' if c == 1 else 'NSR' for c in cls_true_all]
        else:
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
