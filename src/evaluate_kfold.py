import os
import argparse
import numpy as np
import torch
import wfdb
from collections import OrderedDict
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import các hàm cần thiết từ module đã có
from evaluate_aami import (
    EVAL_LEAD_INDICES, N_LEADS, BOUNDARY_TYPES, RHYTHM_ORDER, P_WAVE_EXCLUDED,
    _extract_field, _categorize, evaluate_per_rhythm, build_ecg_unet3p_cgm, _run_inference
)
from data.ECGseg_Datamodule import ECGseg_DataModule

def get_rhythm_labels_for_files(test_files):
    """Đọc nhãn rhythm từ một danh sách các file .hea cụ thể."""
    rhythms = []
    for f in test_files:
        record_path = f[:-4]
        record = wfdb.rdrecord(record_path)
        comments = record.__dict__['comments']
        
        # Ghép tất cả các dòng comment lại thành 1 chuỗi để tìm keyword bệnh lý
        all_comments_text = " ".join(comments).lower()
        rhythms.append(_categorize(all_comments_text, all_comments_text))
    return rhythms

def _get_fold_test_info(data_dir, fold_idx):
    """Lấy DataModule, loader, ground truth và rhythm labels cho 1 fold."""
    dm = ECGseg_DataModule(
        use_sampler=False, data_dir=data_dir, batch_size=16,
        pin_memory=False, dataset_name="ludb", fold=fold_idx
    )
    dm.setup()
    loader = dm.test_dataloader()
    seg_true_all = torch.argmax(dm.test_dataset.tensors[1], dim=1).numpy()

    n_ludb_test = 66
    ludb_files = [os.path.abspath(os.path.join(data_dir, p)) for p in sorted(os.listdir(data_dir)) if p.endswith('.hea')]
    test_start_idx = 1 + fold_idx * n_ludb_test
    test_end_idx = 1 + (fold_idx + 1) * n_ludb_test
    ludb_files_test = ludb_files[test_start_idx:test_end_idx]
    record_rhythms = get_rhythm_labels_for_files(ludb_files_test)

    keep_indices = []
    n_records = len(ludb_files_test)
    for rec in range(n_records):
        for lead_idx in EVAL_LEAD_INDICES:
            keep_indices.append(rec * N_LEADS + lead_idx)
    keep_indices = np.array(keep_indices)

    seg_true_filtered = seg_true_all[keep_indices]
    lead_rhythms = [r for r in record_rhythms for _ in range(len(EVAL_LEAD_INDICES))]
    return loader, seg_true_filtered, lead_rhythms, keep_indices

def _load_model(pth_path, mask=True, strict=True):
    """Tải mô hình từ file .pth hoặc .ckpt."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(pth_path, map_location=device)
    net = build_ecg_unet3p_cgm(mask=mask)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    cleaned = {(k[4:] if k.startswith('net.') else k): v for k, v in state_dict.items()}
    net.load_state_dict(cleaned, strict=strict)
    return net

def predict_and_evaluate_fold(pth_path, data_dir, fold_idx, tolerance):
    """Chạy suy luận và đánh giá cho 1 fold (mô hình CÓ CGM, mask=True)."""
    print(f"\n[{'='*20} XỬ LÝ FOLD {fold_idx} {'='*20}]")
    print(f"File mô hình (có CGM): {pth_path}")

    loader, seg_true_filtered, lead_rhythms, keep_indices = _get_fold_test_info(data_dir, fold_idx)
    net = _load_model(pth_path, mask=True)
    seg_pred_all, _ = _run_inference(net, loader)
    seg_pred_filtered = seg_pred_all[keep_indices]

    results = evaluate_per_rhythm(seg_true_filtered, seg_pred_filtered, lead_rhythms, tolerance)
    return results

def predict_and_evaluate_fold_no_cls(pth_path, data_dir, fold_idx, tolerance):
    """Chạy suy luận và đánh giá cho 1 fold (mô hình KHÔNG có CGM, mask=False)."""
    print(f"\n[{'='*20} XỬ LÝ FOLD {fold_idx} (NO CGM) {'='*20}]")
    print(f"File mô hình (không CGM): {pth_path}")

    loader, seg_true_filtered, lead_rhythms, keep_indices = _get_fold_test_info(data_dir, fold_idx)
    net = _load_model(pth_path, mask=False, strict=False)
    seg_pred_all, _ = _run_inference(net, loader)
    seg_pred_filtered = seg_pred_all[keep_indices]

    results = evaluate_per_rhythm(seg_true_filtered, seg_pred_filtered, lead_rhythms, tolerance)
    return results

def aggregate_kfold_results(fold_results_list):
    """Tính trung bình và độ lệch chuẩn của các metrics qua K folds."""
    agg_results = {'Overall': {btype: {'se':[], 'ppv':[], 'f1':[], 'm':[], 'std':[]} for btype in BOUNDARY_TYPES}}
    
    # Tính F1-score trung bình per rhythm (cho Table 3)
    per_rhythm_f1 = {r: {btype: [] for btype in BOUNDARY_TYPES} for r in RHYTHM_ORDER + ['All']}
    
    for results in fold_results_list:
        # Cập nhật kết quả Overall cho từng fold
        fold_overall_metrics = {btype: {'tp':0, 'fp':0, 'fn':0, 'errs_mean':[], 'errs_std':[]} for btype in BOUNDARY_TYPES}
        
        # Biến phụ để tính 'All' micro/macro trong 1 fold
        all_f1_fold = {btype: [] for btype in BOUNDARY_TYPES}

        for rhythm, metrics in results.items():
            for btype in BOUNDARY_TYPES:
                m = metrics[btype]
                if m is not None:
                    fold_overall_metrics[btype]['tp'] += m['tp']
                    fold_overall_metrics[btype]['fp'] += m['fp']
                    fold_overall_metrics[btype]['fn'] += m['fn']
                    # Lưu lại tp, mean_err, std_err để tính trung bình có trọng số (weighted)
                    if m['tp'] > 0:
                        fold_overall_metrics[btype]['errs_mean'].append((m['tp'], m['mean_err']))
                        fold_overall_metrics[btype]['errs_std'].append((m['tp'], m['std_err']))
                    
                    per_rhythm_f1[rhythm][btype].append(m['f1'])
                    all_f1_fold[btype].append(m['f1'])
                    
        # F1 'All' macro cho fold này
        for btype in BOUNDARY_TYPES:
            if len(all_f1_fold[btype]) > 0:
                per_rhythm_f1['All'][btype].append(np.mean(all_f1_fold[btype]))
                
        # Tính Se, PPV, F1 tổng cộng của fold này, rồi lưu vào mảng agg_results
        for btype in BOUNDARY_TYPES:
            om = fold_overall_metrics[btype]
            tp, fp, fn = om['tp'], om['fp'], om['fn']
            se = tp / (tp + fn) if (tp + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            # Tính gộp Mean và Std Error (Pooled Mean and Variance)
            if tp > 0 and len(om['errs_mean']) > 0:
                mean_m = sum(t * mean_val for t, mean_val in om['errs_mean']) / tp
                # E[X^2] = var + mean^2
                e_x2 = sum(t * (s**2 + mean_val**2) for (t, mean_val), (_, s) in zip(om['errs_mean'], om['errs_std'])) / tp
                mean_std = np.sqrt(max(0, e_x2 - mean_m**2))
            else:
                mean_m = 0
                mean_std = 0
            
            agg_results['Overall'][btype]['se'].append(se)
            agg_results['Overall'][btype]['ppv'].append(ppv)
            agg_results['Overall'][btype]['f1'].append(f1)
            agg_results['Overall'][btype]['m'].append(mean_m)
            agg_results['Overall'][btype]['std'].append(mean_std)

    # Tính Mean ± Std
    final_agg = {'Overall': {}}
    for btype in BOUNDARY_TYPES:
        final_agg['Overall'][btype] = {
            'se_mean': np.mean(agg_results['Overall'][btype]['se']) * 100,
            'se_std': np.std(agg_results['Overall'][btype]['se']) * 100,
            'ppv_mean': np.mean(agg_results['Overall'][btype]['ppv']) * 100,
            'ppv_std': np.std(agg_results['Overall'][btype]['ppv']) * 100,
            'f1_mean': np.mean(agg_results['Overall'][btype]['f1']) * 100,
            'f1_std': np.std(agg_results['Overall'][btype]['f1']) * 100,
            'm_mean': np.mean(agg_results['Overall'][btype]['m']),
            'm_std': np.mean(agg_results['Overall'][btype]['std']) # Lấy trung bình độ lệch chuẩn của các fold
        }
        
    # Tính Mean F1 per rhythm
    final_rhythm_f1 = {}
    for r in RHYTHM_ORDER + ['All']:
        final_rhythm_f1[r] = {}
        for btype in BOUNDARY_TYPES:
            vals = per_rhythm_f1[r][btype]
            if len(vals) > 0:
                final_rhythm_f1[r][btype] = np.mean(vals) * 100
            else:
                final_rhythm_f1[r][btype] = None
                
    return final_agg, final_rhythm_f1

def count_afib_afl_fp_from_predictions(seg_true_filtered, seg_pred_filtered, lead_rhythms):
    """Đếm số FP sóng P và số QRS beats trên các bản ghi AFIB/AFL."""
    from evaluate_aami import post_processing, find_annotated_range, extract_boundaries, extract_segments
    
    P_BTYPES = ['P_onset', 'P_offset']
    fp_counts = {'AFIB': {b: 0 for b in P_BTYPES}, 'AFL': {b: 0 for b in P_BTYPES}}
    beat_counts = {'AFIB': 0, 'AFL': 0, 'ALL': 0}  # Đếm số QRS beats
    
    for i, rhythm in enumerate(lead_rhythms):
        gt_labels = seg_true_filtered[i]
        ann_start, ann_end = find_annotated_range(gt_labels)
        gt_trimmed = gt_labels[ann_start:ann_end + 1]
        
        # Đếm số QRS beats từ ground truth cho TẤT CẢ rhythms
        gt_b = extract_boundaries(extract_segments(gt_trimmed), BOUNDARY_TYPES)
        beat_counts['ALL'] += len(gt_b['QRS_onset'])
        
        if rhythm not in ('AFIB', 'AFL'):
            continue
        
        beat_counts[rhythm] += len(gt_b['QRS_onset'])
        
        pred_labels = seg_pred_filtered[i]
        pred_trimmed = pred_labels[ann_start:ann_end + 1]
        
        p_segs = post_processing(pred_trimmed)
        p_b = extract_boundaries(p_segs, BOUNDARY_TYPES)
        
        for btype in P_BTYPES:
            fp_counts[rhythm][btype] += len(p_b[btype])
    
    return fp_counts, beat_counts

def predict_and_evaluate_fold_full(pth_path, data_dir, fold_idx, tolerance, mask=True, strict=True):
    """Chạy suy luận và đánh giá cho 1 fold. Trả về:
    - results: kết quả AAMI per rhythm
    - afib_afl_fp: số FP sóng P trên AFIB/AFL
    - beat_counts: số QRS beats per rhythm
    """
    cgm_label = "có CGM" if mask else "không CGM"
    print(f"\n[{'='*20} XỬ LÝ FOLD {fold_idx} ({cgm_label}) {'='*20}]")
    print(f"File mô hình ({cgm_label}): {pth_path}")
    
    loader, seg_true_filtered, lead_rhythms, keep_indices = _get_fold_test_info(data_dir, fold_idx)
    net = _load_model(pth_path, mask=mask, strict=strict)
    seg_pred_all, _ = _run_inference(net, loader)
    seg_pred_filtered = seg_pred_all[keep_indices]
    
    results = evaluate_per_rhythm(seg_true_filtered, seg_pred_filtered, lead_rhythms, tolerance)
    afib_afl_fp, beat_counts = count_afib_afl_fp_from_predictions(
        seg_true_filtered, seg_pred_filtered, lead_rhythms
    )
    
    return results, afib_afl_fp, beat_counts

def aggregate_p_wave_table5(fold_results_list, fold_fp_list, fold_beats_list):
    """Tổng hợp metrics cho Table 5 (giống bài báo [36])."""
    P_BTYPES = ['P_onset', 'P_offset']
    k = len(fold_results_list)
    
    # === FP trung bình trên AFIB/AFL ===
    avg_fp = {'AFIB': {b: 0 for b in P_BTYPES}, 'AFL': {b: 0 for b in P_BTYPES}}
    for fp_data in fold_fp_list:
        for r in ['AFIB', 'AFL']:
            for b in P_BTYPES:
                avg_fp[r][b] += fp_data[r][b]
    for r in ['AFIB', 'AFL']:
        for b in P_BTYPES:
            avg_fp[r][b] /= k
    
    # === Tổng số beats ===
    total_beats = {'AFIB': 0, 'AFL': 0, 'ALL': 0}
    for bc in fold_beats_list:
        for key in total_beats:
            total_beats[key] += bc[key]
    
    # === PPV và Se trên TOÀN BỘ test set ===
    totals = {b: {'tp': 0, 'fp': 0, 'fn': 0} for b in P_BTYPES}
    for results in fold_results_list:
        for rhythm, metrics in results.items():
            for btype in P_BTYPES:
                m = metrics[btype]
                if m is None:
                    continue
                totals[btype]['tp'] += m['tp']
                totals[btype]['fp'] += m['fp']
                totals[btype]['fn'] += m['fn']
    
    # Cộng thêm FP từ AFIB/AFL
    total_afib_afl_fp = {b: 0 for b in P_BTYPES}
    for fp_data in fold_fp_list:
        for r in ['AFIB', 'AFL']:
            for b in P_BTYPES:
                total_afib_afl_fp[b] += fp_data[r][b]
    
    ppv_se = {}
    for btype in P_BTYPES:
        tp = totals[btype]['tp']
        fp_total = totals[btype]['fp'] + total_afib_afl_fp[btype]
        fn = totals[btype]['fn']
        ppv_se[btype] = {
            'ppv': tp / (tp + fp_total) * 100 if (tp + fp_total) > 0 else 0,
            'se': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        }
    
    return avg_fp, ppv_se, total_beats

def print_table_1_cgm_ablation(fp_with, ppv_se_with, fp_without, ppv_se_without, beats):
    """In Bảng 1 giống Table 5 bài báo [36]."""
    W = 115
    print("\n" + "=" * W)
    print("TABLE 1: REDUCED FALSE P-WAVE PREDICTIONS (AVERAGED OVER K FOLDS)")
    print("=" * W)
    
    ab = beats['AFIB']; fb = beats['AFL']; tb = beats['ALL']
    print(f"{'':25} | {'AFIB (' + str(ab) + ' beats)':^21} | {'AFL (' + str(fb) + ' beats)':^21} | {'All (' + str(tb) + ' beats)':^47}")
    print(f"{'':25} | {'False Positives':^21} | {'False Positives':^21} | {'PPV (Precision)':^23} | {'Se (Recall)':^21}")
    print(f"{'':25} | {'P onset':>10} {'P offset':>10} | {'P onset':>10} {'P offset':>10} | {'P onset':>11} {'P offset':>11} | {'P onset':>10} {'P offset':>10}")
    print("-" * W)
    
    for model_name, fp_data, ppv_se in [
        ("Trained\nw/o classification", fp_without, ppv_se_without),
        ("Trained\nw/ classification", fp_with, ppv_se_with)
    ]:
        name_lines = model_name.split('\n')
        label = name_lines[0] + ' ' + name_lines[1]
        
        af_on = f"{fp_data['AFIB']['P_onset']:.2f}"
        af_off = f"{fp_data['AFIB']['P_offset']:.2f}"
        fl_on = f"{fp_data['AFL']['P_onset']:.2f}"
        fl_off = f"{fp_data['AFL']['P_offset']:.2f}"
        ppv_on = f"{ppv_se['P_onset']['ppv']:.2f}"
        ppv_off = f"{ppv_se['P_offset']['ppv']:.2f}"
        se_on = f"{ppv_se['P_onset']['se']:.2f}"
        se_off = f"{ppv_se['P_offset']['se']:.2f}"
        
        print(f"{label:<25} | {af_on:>10} {af_off:>10} | {fl_on:>10} {fl_off:>10} | {ppv_on:>11} {ppv_off:>11} | {se_on:>10} {se_off:>10}")
    
    print("=" * W)

def print_table_2_sota(final_agg):
    print("\n" + "="*90)
    print("TABLE 2: COMPARISON WITH SOTA METHODS ON LUDB")
    print("="*90)
    
    # Dữ liệu SOTA hardcode từ Table 4 của bài báo [36]
    sota_data = [
        ("Kalyakulina [4]", "Se (%)", "98.46", "98.46", "99.61", "99.61", "-", "98.03"),
        ("", "PPV (%)", "96.41", "96.41", "99.87", "99.87", "-", "98.84"),
        ("", "m ± σ (ms)", "-2.7±10.2", "0.4±11.4", "-8.1±7.7", "3.8±8.8", "-", "5.7±15.5"),
        ("Sereda [29]", "Se (%)", "95.20", "95.39", "99.51", "99.50", "97.95", "97.56"),
        ("", "PPV (%)", "82.66", "82.59", "98.17", "97.96", "94.81", "94.96"),
        ("", "m ± σ (ms)", "2.7±21.9", "-7.4±28.6", "2.6±12.4", "-1.7±14.1", "8.4±28.2", "-3.1±28.2"),
        ("Moskalenko [8]", "Se (%)", "98.61", "98.59", "99.99", "99.99", "99.32", "99.40"),
        ("", "PPV (%)", "95.61", "95.59", "99.99", "99.99", "99.02", "99.10"),
        ("", "m ± σ (ms)", "-4.1±20.4", "3.7±19.6", "1.8±13.0", "-0.2±11.4", "-3.6±28.0", "-4.1±35.3"),
    ]
    
    header = f"{'Method':<16} | {'Metrics':<10} | {'P onset':>9} | {'P offset':>9} | {'QRS onset':>9} | {'QRS offset':>10} | {'T onset':>9} | {'T offset':>9}"
    print(header)
    print("-" * 100)
    for row in sota_data:
        print(f"{row[0]:<16} | {row[1]:<10} | {row[2]:>9} | {row[3]:>9} | {row[4]:>9} | {row[5]:>10} | {row[6]:>9} | {row[7]:>9}")
    
    print("-" * 100)
    # Thêm kết quả của bạn
    res = final_agg['Overall']
    
    def get_fmt(metric_key):
        return [f"{res[b][metric_key]:.2f}" for b in BOUNDARY_TYPES]
        
    se_vals = get_fmt('se_mean')
    ppv_vals = get_fmt('ppv_mean')
    m_sig_vals = [f"{res[b]['m_mean']:.1f}±{res[b]['m_std']:.1f}" for b in BOUNDARY_TYPES]
    
    print(f"{'Our K-fold':<16} | {'Se (%)':<10} | {se_vals[0]:>9} | {se_vals[1]:>9} | {se_vals[2]:>9} | {se_vals[3]:>10} | {se_vals[4]:>9} | {se_vals[5]:>9}")
    print(f"{'':<16} | {'PPV (%)':<10} | {ppv_vals[0]:>9} | {ppv_vals[1]:>9} | {ppv_vals[2]:>9} | {ppv_vals[3]:>10} | {ppv_vals[4]:>9} | {ppv_vals[5]:>9}")
    print(f"{'':<16} | {'m ± σ (ms)':<10} | {m_sig_vals[0]:>9} | {m_sig_vals[1]:>9} | {m_sig_vals[2]:>9} | {m_sig_vals[3]:>10} | {m_sig_vals[4]:>9} | {m_sig_vals[5]:>9}")
    print("="*100)

def print_table_3_rhythm(final_rhythm_f1):
    print("\n" + "="*90)
    print("TABLE 3: F1-SCORES (%) PER RHYTHM (AVERAGED OVER K FOLDS)")
    print("="*90)
    
    header = f"{'Rhythm':<10}" + " | ".join([f"{b.replace('_', ' '):>10}" for b in BOUNDARY_TYPES])
    print(header)
    print("-" * 90)
    
    for r in RHYTHM_ORDER + ['All']:
        if r not in final_rhythm_f1: continue
        if r == 'VT': continue  # Bỏ qua dòng VT vì LUDB không có dữ liệu cho loại này
        row = f"{r:<10} | "
        for btype in BOUNDARY_TYPES:
            val = final_rhythm_f1[r][btype]
            if val is None:
                row += f"{'-':>10} | "
            else:
                row += f"{val:10.2f} | "
        print(row)
    print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Đánh giá và xuất 3 bảng báo cáo từ K-fold models.')
    parser.add_argument('--models', type=str, required=False, 
                        default="final_model_180_cls.pth",
                        help='Đường dẫn các file mô hình CÓ CGM, cách nhau bởi dấu phẩy')
    parser.add_argument('--models_no_cls', type=str, required=False,
                        default="final_model_no_cls_fold0.pth",
                        help='Đường dẫn các file mô hình KHÔNG CÓ CGM, cách nhau bởi dấu phẩy')
    parser.add_argument('--data_dir', type=str, required=False,
                        default=r"C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\NCKH\Điện tim\ecg-MI-classification-code\ecg-segmentation\segmentation_data\lobachevsky-university-electrocardiography-database-1.0.1\data",
                        help='Đường dẫn LUDB data')
    parser.add_argument('--tolerance', type=int, default=75,
                        help='Ngưỡng AAMI tính bằng số mẫu (75 = 150ms ở 500Hz)')
    
    args = parser.parse_args()
    
    # ===== Đánh giá mô hình CÓ CGM =====
    model_paths = [p.strip() for p in args.models.split(',')]
    print(f"Bắt đầu đánh giá K-fold ({len(model_paths)} folds) — MÔ HÌNH CÓ CGM...")
    
    fold_results_with = []
    fold_fp_with = []
    fold_beats_with = []
    for idx, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"LỖI: Không tìm thấy file {path}")
            exit(1)
        results, afib_afl_fp, beat_counts = predict_and_evaluate_fold_full(
            path, args.data_dir, fold_idx=idx, tolerance=args.tolerance, mask=True, strict=True
        )
        fold_results_with.append(results)
        fold_fp_with.append(afib_afl_fp)
        fold_beats_with.append(beat_counts)
        
    print("\nĐang tổng hợp kết quả mô hình CÓ CGM...")
    final_agg, final_rhythm_f1 = aggregate_kfold_results(fold_results_with)
    fp_with, ppv_se_with, beats_with = aggregate_p_wave_table5(fold_results_with, fold_fp_with, fold_beats_with)

    # ===== Đánh giá mô hình KHÔNG CÓ CGM =====
    model_no_cls_paths = [p.strip() for p in args.models_no_cls.split(',')]
    print(f"\nBắt đầu đánh giá K-fold ({len(model_no_cls_paths)} folds) — MÔ HÌNH KHÔNG CÓ CGM...")

    fold_results_without = []
    fold_fp_without = []
    fold_beats_without = []
    for idx, path in enumerate(model_no_cls_paths):
        if not os.path.exists(path):
            print(f"LỖI: Không tìm thấy file {path}")
            exit(1)
        results, afib_afl_fp, beat_counts = predict_and_evaluate_fold_full(
            path, args.data_dir, fold_idx=idx, tolerance=args.tolerance, mask=False, strict=False
        )
        fold_results_without.append(results)
        fold_fp_without.append(afib_afl_fp)
        fold_beats_without.append(beat_counts)

    print("\nĐang tổng hợp kết quả mô hình KHÔNG CÓ CGM...")
    fp_without, ppv_se_without, beats_without = aggregate_p_wave_table5(fold_results_without, fold_fp_without, fold_beats_without)

    # Dùng beats từ mô hình CÓ CGM (cùng data → cùng số beats)
    beats = beats_with

    # ===== In 3 bảng console =====
    print_table_1_cgm_ablation(fp_with, ppv_se_with, fp_without, ppv_se_without, beats)
    print_table_2_sota(final_agg)
    print_table_3_rhythm(final_rhythm_f1)
    
    print("\nDONE!")

