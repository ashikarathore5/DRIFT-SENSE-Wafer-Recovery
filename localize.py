import os
import sys
import time
import argparse
import numpy as np
import cv2
import pandas as pd

def find_subpixel_drift(ref_img, search_img, crop_size=400, scale_range=(0.98, 1.02), num_scales=5):
    """
    Applied Materials Compliant Localizer (1:1 Wafer Alignment):
    1. Extracts high-confidence central feature crop from Reference Image.
    2. Multi-scale search around 1.0x (0.98 to 1.02) to handle micro-zoom variations.
    3. Finds peak correlation match in Search Image.
    4. 10x Bicubic local surface interpolation for sub-pixel accuracy (< 0.1px precision).
    """
    H_ref, W_ref = ref_img.shape[:2]
    H_src, W_src = search_img.shape[:2]
    
    # 1. Extract Central Template from Reference Image
    cx_ref, cy_ref = W_ref // 2, H_ref // 2
    half_crop = crop_size // 2
    
    x1 = max(0, cx_ref - half_crop)
    y1 = max(0, cy_ref - half_crop)
    x2 = min(W_ref, cx_ref + half_crop)
    y2 = min(H_ref, cy_ref + half_crop)
    
    ref_crop = ref_img[y1:y2, x1:x2]
    h_crop, w_crop = ref_crop.shape[:2]

    best_val = -1.0
    best_loc = None
    best_w, best_h = w_crop, h_crop

    # 2. Multi-Scale Pyramid around 1.0x scale
    scales = np.linspace(scale_range[0], scale_range[1], num_scales)
    
    for scale in scales:
        if scale == 1.0:
            template = ref_crop
        else:
            template = cv2.resize(ref_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        ht, wt = template.shape[:2]
        if ht >= H_src or wt >= W_src:
            continue
            
        res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_w, best_h = wt, ht
            best_res = res

    if best_loc is None:
        return W_src / 2.0, H_src / 2.0

    x, y = best_loc

    # 3. Sub-Pixel Refinement via 10x Local Surface Interpolation
    pad = 2
    y_min, y_max = max(0, y - pad), min(best_res.shape[0], y + pad + 1)
    x_min, x_max = max(0, x - pad), min(best_res.shape[1], x + pad + 1)
    roi = best_res[y_min:y_max, x_min:x_max]

    sub_x, sub_y = 0.0, 0.0
    if roi.shape[0] == 2*pad + 1 and roi.shape[1] == 2*pad + 1:
        zoom_factor = 10
        roi_upsampled = cv2.resize(roi, (roi.shape[1]*zoom_factor, roi.shape[0]*zoom_factor), interpolation=cv2.INTER_CUBIC)
        _, _, _, max_sub_loc = cv2.minMaxLoc(roi_upsampled)
        sub_x = (max_sub_loc[0] - pad * zoom_factor) / float(zoom_factor)
        sub_y = (max_sub_loc[1] - pad * zoom_factor) / float(zoom_factor)

    # Calculate final center coordinate in search image space
    final_pred_x = float(x) + sub_x + (best_w / 2.0)
    final_pred_y = float(y) + sub_y + (best_h / 2.0)

    return final_pred_x, final_pred_y

def resolve_file_path(base_dir, rel_path):
    rel_path = str(rel_path).replace('\\', '/').strip()
    p1 = os.path.join(base_dir, rel_path)
    if os.path.exists(p1): return p1
    parent_dir = os.path.dirname(os.path.normpath(base_dir))
    p2 = os.path.join(parent_dir, rel_path)
    if os.path.exists(p2): return p2
    folder_name = os.path.basename(os.path.normpath(base_dir))
    if rel_path.startswith(f"{folder_name}/"):
        p3 = os.path.join(base_dir, rel_path[len(folder_name)+1:])
        if os.path.exists(p3): return p3
    return p1

def find_column(df, possible_names):
    cols_clean = {str(c).strip().lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in cols_clean:
            return cols_clean[name.lower()]
    return None

def evaluate(data_dir="dataset/val", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(data_dir), "labels.csv")

    df = pd.read_csv(csv_path)
    
    ref_col = find_column(df, ["ref_path", "reference_path", "ref_image", "ref", "reference"])
    src_col = find_column(df, ["search_path", "search_image", "src_path", "search", "source"])
    x_col = find_column(df, ["true_x", "center_x", "x", "drift_x", "shift_x", "target_x", "dx"])
    y_col = find_column(df, ["true_y", "center_y", "y", "drift_y", "shift_y", "target_y", "dy"])

    has_ground_truth = (x_col is not None) and (y_col is not None)

    print("\n═══════════════════════════════════════════════════════")
    print(" 🚀 DRIFT-SENSE EVALUATION REPORT (APPLIED MATERIALS SPEC)")
    print("═══════════════════════════════════════════════════════")

    errors = []
    latencies = []
    results = []

    for idx, row in df.iterrows():
        ref_p = resolve_file_path(data_dir, row[ref_col])
        src_p = resolve_file_path(data_dir, row[src_col])
        
        ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        src_img = cv2.imread(src_p, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or src_img is None:
            continue

        t0 = time.perf_counter()
        pred_x, pred_y = find_subpixel_drift(ref_img, src_img)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        res_entry = {
            "ref_path": row[ref_col],
            "search_path": row[src_col],
            "pred_x": pred_x,
            "pred_y": pred_y,
            "latency_ms": latency_ms
        }

        if has_ground_truth:
            true_x, true_y = float(row[x_col]), float(row[y_col])
            err = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            errors.append(err)
            res_entry["true_x"] = true_x
            res_entry["true_y"] = true_y
            res_entry["error_px"] = err

        results.append(res_entry)

    total_pairs = len(results)

    if total_pairs == 0:
        print("❌ No images could be loaded. Please check folder paths!")
        return

    print(f" Total Pairs Evaluated : {total_pairs}")
    print("-------------------------------------------------------")

    if has_ground_truth and len(errors) > 0:
        errors = np.array(errors)
        print(f" Mean Error           : {np.mean(errors):.3f} px")
        print(f" Median Error         : {np.median(errors):.3f} px")
        print(f" 95th Percentile      : {np.percentile(errors, 95):.3f} px")
        print("-------------------------------------------------------")
        print(f" Pass @ 5px           : {np.mean(errors <= 5.0)*100:.1f}%")
        print(f" Pass @ 4px           : {np.mean(errors <= 4.0)*100:.1f}%")
        print(f" Pass @ 2px           : {np.mean(errors <= 2.0)*100:.1f}%")
        print(f" Pass @ 1px           : {np.mean(errors <= 1.0)*100:.1f}%")
        print("-------------------------------------------------------")

    print(f" Mean Latency         : {np.mean(latencies):.2f} ms/pair")
    print("═══════════════════════════════════════════════════════\n")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, "predictions_manifest.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/val")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    evaluate(args.data_dir, args.out_dir)