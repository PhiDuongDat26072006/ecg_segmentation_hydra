import numpy as np
import os

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
                right_seg = segments[i+1] if i + 1 < len(segments) else None
                
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
                    i += 2 # Bỏ qua đoạn right_seg vì đã gộp
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
        
        # Hợp nhất các đoạn kề nhau có cùng nhãn sau khi đã thay đổi
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
        intervals.append([segments[qrs_idx]]) # QRS segment
        start_idx = qrs_idx + 1
    intervals.append(segments[start_idx:])
    
    for interval in intervals:
        if not interval: continue
        if interval[0]['label'] == qrs_label:
            final_segments.append(interval[0])
            continue
            
        # Tìm P và T trong khoảng giữa 2 QRS
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

def evaluate_aami_single_type(true_b, pred_b, tolerance=75):
    """So khớp và tính TP, FP, FN với sai số <= 150ms (75 mẫu)."""
    tp = 0
    errors = []
    matched_pred = set()
    
    for tb in true_b:
        closest_pb = None
        min_dist = float('inf')
        for pb in pred_b:
            if pb in matched_pred: continue
            dist = abs(tb - pb)
            if dist <= tolerance and dist < min_dist:
                min_dist = dist
                closest_pb = pb
        
        if closest_pb is not None:
            tp += 1
            # Khoảng cách thực tế: Dự đoán - Ground Truth
            errors.append(closest_pb - tb) 
            matched_pred.add(closest_pb)
            
    fn = len(true_b) - tp
    fp = len(pred_b) - tp
    
    return tp, fp, fn, errors

def calculate_accuracy_and_dice(seg_true, seg_pred, cls_true, cls_pred):
    """Tính toán Accuracy tổng thể và Dice Score cho từng class (giống hệt code gốc)."""
    cls_acc = np.mean(cls_pred == cls_true)
    seg_acc = np.mean(seg_pred == seg_true)
    
    print("\n" + "="*50)
    print("PIXEL-WISE ACCURACY & DICE SCORE")
    print("="*50)
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
    print("="*50 + "\n")

def main():
    save_path = 'predictions1.npz'
    if not os.path.exists(save_path):
        print(f"Không tìm thấy file {save_path}! Vui lòng train mô hình trước.")
        return
        
    print("Đang nạp file predictions.npz và tiến hành hậu xử lý...")
    data = np.load(save_path)
    seg_true_all = data['seg_true']
    seg_pred_all = data['seg_pred']
    
    # Lấy thêm nhãn classification nếu có (để tính accuracy như file gốc)
    if 'cls_true' in data and 'cls_pred' in data:
        cls_true_all = data['cls_true']
        cls_pred_all = data['cls_pred']
        calculate_accuracy_and_dice(seg_true_all, seg_pred_all, cls_true_all, cls_pred_all)
    
    metrics = {k: {'tp': 0, 'fp': 0, 'fn': 0, 'errors': []} for k in ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset']}
    
    # Duyệt qua từng bản ghi điện tim
    for i in range(len(seg_true_all)):
        print(f"BẢN GHI THỨ {i}")
        # 1. Trích xuất ranh giới Ground Truth (Bác sĩ)
        t_segs = extract_segments(seg_true_all[i])
        t_b = extract_boundaries(t_segs)
        # 2. Hậu xử lý kết quả Dự đoán của AI
        p_segs = extract_segments(seg_pred_all[i])
        p_segs = noise_reduction(p_segs, min_length=20) # Bỏ nhiễu < 40ms
        p_segs = boundary_determination(p_segs)         # Xác định P và T chính
        p_b = extract_boundaries(p_segs)
        # 3. So khớp chuẩn AAMI 150ms
        for k in metrics.keys():
            print(f"{k}")
            tp, fp, fn, errors = evaluate_aami_single_type(t_b[k], p_b[k], tolerance=75)
            metrics[k]['tp'] += tp
            metrics[k]['fp'] += fp
            metrics[k]['fn'] += fn
            metrics[k]['errors'].extend(errors)
            
    # In báo cáo
    print("\n" + "="*95)
    print("AAMI 150ms EVALUATION RESULTS (Tolerance: 75 samples)")
    print("="*95)
    print(f"{'Boundary':<15} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'F1-Score':<10} | {'Mean Err (ms)':<15} | {'Std Err (ms)':<15}")
    print("-" * 95)
    for k in metrics:
        tp = metrics[k]['tp']
        fp = metrics[k]['fp']
        fn = metrics[k]['fn']
        errors = metrics[k]['errors']
        
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        
        # Tần số 500Hz -> 1 sample = 2ms
        mean_err = np.mean(errors) * 2 if errors else 0
        std_err = np.std(errors) * 2 if errors else 0
        
        print(f"{k:<15} | {tp:<6} | {fp:<6} | {fn:<6} | {f1:<10.4f} | {mean_err:<15.2f} | {std_err:<15.2f}")
    print("="*95)

    print("END")

if __name__ == "__main__":
    main()
