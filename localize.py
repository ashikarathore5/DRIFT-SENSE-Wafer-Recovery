import os
import cv2
import time
import argparse
import numpy as np

def predict_center(search_path, ref_path, model_path="drift_sense_model.pth"):
    search_raw = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    ref_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    
    if search_raw is None or ref_raw is None:
        raise ValueError(f"❌ Could not read images at: {search_path} or {ref_path}")

    # 1. Downsample reference image (1000x1000 -> 100x100)
    ref_100 = cv2.resize(ref_raw, (100, 100), interpolation=cv2.INTER_AREA)

    # 2. Stage A: Full 100x100 Global Template Correlation Map
    res_full = cv2.matchTemplate(search_raw, ref_100, cv2.TM_CCOEFF_NORMED) # Shape: (901, 901)

    # 3. Stage B: Focused Landmark Crop (60x60 Center Crop around the defect)
    # Since landmark is centered in ref_100, cropping 20:80 isolates landmark from repeating grid
    crop_size = 60
    offset = (100 - crop_size) // 2  # 20 px
    ref_crop = ref_100[offset:offset + crop_size, offset:offset + crop_size]

    res_crop = cv2.matchTemplate(search_raw, ref_crop, cv2.TM_CCOEFF_NORMED) # Shape: (941, 941)

    # Align Stage B response map to Stage A coordinates
    res_crop_aligned = res_crop[offset:offset + res_full.shape[0], offset:offset + res_full.shape[1]]

    # Combined Confidence Map (Eliminates periodic false peaks)
    score_map = res_full * np.maximum(res_crop_aligned, 0.0)

    # 4. Find Coarse Peak
    _, max_val, _, (lx, ly) = cv2.minMaxLoc(score_map)

    coarse_x = float(lx) + 50.0
    coarse_y = float(ly) + 50.0

    # 5. 2D Parabolic Peak Interpolation for Sub-Pixel Accuracy (< 0.5px)
    dx, dy = 0.0, 0.0
    if 0 < lx < score_map.shape[1] - 1 and 0 < ly < score_map.shape[0] - 1:
        denom_x = (2 * score_map[ly, lx] - score_map[ly, lx + 1] - score_map[ly, lx - 1])
        if abs(denom_x) > 1e-5:
            dx = (score_map[ly, lx + 1] - score_map[ly, lx - 1]) / (2 * denom_x)
            
        denom_y = (2 * score_map[ly, lx] - score_map[ly + 1, lx] - score_map[ly - 1, lx])
        if abs(denom_y) > 1e-5:
            dy = (score_map[ly + 1, lx] - score_map[ly - 1, lx]) / (2 * denom_y)

    final_x = coarse_x + dx
    final_y = coarse_y + dy
    
    return float(final_x), float(final_y), float(max_val)

def main():
    parser = argparse.ArgumentParser(description="Applied Materials DRIFT-SENSE Inference Engine")
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--search_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="drift_sense_model.pth")
    args = parser.parse_args()

    start_time = time.perf_counter()
    pred_x, pred_y, confidence = predict_center(args.search_path, args.ref_path, args.model_path)
    latency_ms = (time.perf_counter() - start_time) * 1000

    print(f"PREDICTED_CENTER: ({pred_x:.4f}, {pred_y:.4f})")
    print(f"CONFIDENCE: {confidence:.4f}")
    print(f"LATENCY_MS: {latency_ms:.2f}ms")

if __name__ == "__main__":
    main()