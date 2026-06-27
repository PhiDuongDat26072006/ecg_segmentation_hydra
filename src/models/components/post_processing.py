def _merge_adjacent(segments):
    """Hợp nhất các segment liên tiếp có cùng nhãn."""
    if not segments:
        return []
    merged = []
    current = segments[0]
    for nxt in segments[1:]:
        if current['label'] == nxt['label']:
            current['offset'] = nxt['offset']
            current['length'] += nxt['length']
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged

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
                    # Xóa bỏ: Biến thành Baseline
                    if seg['label'] != baseline_label:
                        seg['label'] = baseline_label
                        changed = True
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
            i += 1

        # Hợp nhất các đoạn kề nhau có cùng nhãn
        segments = _merge_adjacent(new_segments)

    return segments

def boundary_determination(segments, p_label=0, qrs_label=1, t_label=2, baseline_label=3):
    """Bước 3: Giữ lại P và T dài nhất giữa mỗi cặp QRS, gộp lại."""
    qrs_indices = [i for i, seg in enumerate(segments) if seg['label'] == qrs_label]

    if not qrs_indices:
        return segments

    # Chia thành các interval: [trước QRS₁] [QRS₁] [giữa] [QRS₂] ... [sau QRS cuối]
    intervals = []
    start = 0
    for qi in qrs_indices:
        intervals.append(segments[start:qi])    # đoạn trước/giữa
        intervals.append([segments[qi]])        # QRS riêng
        start = qi + 1
    intervals.append(segments[start:])          # đoạn sau QRS cuối

    # Xử lý từng interval
    final_segments = []
    for interval in intervals:
        if not interval:
            continue

        # QRS → giữ nguyên
        if interval[0]['label'] == qrs_label:
            final_segments.append(interval[0])
            continue

        # Tìm P và T dài nhất trong interval
        p_waves = [s for s in interval if s['label'] == p_label]
        t_waves = [s for s in interval if s['label'] == t_label]
        longest_p = max(p_waves, key=lambda x: x['length']) if p_waves else None
        longest_t = max(t_waves, key=lambda x: x['length']) if t_waves else None

        # Đánh P/T không dài nhất thành baseline
        for seg in interval:
            if seg['label'] == p_label and seg is not longest_p:
                seg['label'] = baseline_label
            elif seg['label'] == t_label and seg is not longest_t:
                seg['label'] = baseline_label
            final_segments.append(seg)

    return _merge_adjacent(final_segments)

def post_processing(pred_trimmed):
    p_segs = extract_segments(pred_trimmed)
    p_segs = noise_reduction(p_segs, min_length=20)
    p_segs = boundary_determination(p_segs)
    return p_segs


