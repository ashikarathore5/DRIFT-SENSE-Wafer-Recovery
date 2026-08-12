import os
import time
import argparse
import pandas as pd
import numpy as np

# Import localization inference engine
try:
    from inference import predict_center
except ImportError:
    raise ImportError("❌ Could not import 'predict_center' from 'inference.py'. Make sure inference.py is in the root directory!")

def run_official_evaluation(csv_path="dataset/val/labels.csv", output_manifest="results/evaluation_manifest.csv"):
    if not os.path.exists(csv_path):
        # Fallback to train CSV if val CSV doesn't exist
        csv_path = "dataset/train/labels.csv" if os.path.exists("dataset/train/labels.csv") else "data/ground_truth.csv"
        
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find ground truth CSV at '{csv_path}'. Run generate_dataset.py first!")
        return

    df = pd.read_csv(csv_path)
    os.makedirs("results", exist_ok=True)
    
    # Standardize column names (handles both center_x/gt_x)
    x_col = 'center_x' if 'center_x' in df.columns else 'gt_x'
    y_col = 'center_y' if 'center_y' in df.columns else 'gt_y'
    
    errors = []
    latencies = []
    pred_x_list = []
    pred_y_list = []
    
    print(f"📊 Running Official AMAT Evaluation on {len(df)} samples from '{csv_path}'...\n")
    
    # Determine base directory OS-agnostically
    csv_dir = os.path.dirname(csv_path)
    root_dir = os.path.dirname(csv_dir) if csv_dir.endswith(('val', 'train')) else ""

    for idx, row in df.iterrows():
        search_rel = row['search_path']
        ref_rel = row['ref_path']

        search_p = os.path.normpath(os.path.join(root_dir, search_rel)) if root_dir and not os.path.exists(search_rel) else os.path.normpath(search_rel)
        ref_p = os.path.normpath(os.path.join(root_dir, ref_rel)) if root_dir and not os.path.exists(ref_rel) else os.path.normpath(ref_rel)

        t0 = time.perf_counter()
        pred_x, pred_y, conf = predict_center(search_p, ref_p)
        t1 = time.perf_counter()
        
        # Euclidean Error Distance (AMAT Benchmark Formula)
        gt_x, gt_y = float(row[x_col]), float(row[y_col])
        err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        
        errors.append(err)
        latencies.append((t1 - t0) * 1000.0)
        pred_x_list.append(round(float(pred_x), 4))
        pred_y_list.append(round(float(pred_y), 4))

    errors = np.array(errors)
    
    # Calculate Required PDF Metrics
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    worst_err = np.max(errors)
    
    pass_5px = np.mean(errors <= 5.0) * 100.0
    pass_4px = np.mean(errors <= 4.0) * 100.0
    pass_2px = np.mean(errors <= 2.0) * 100.0
    pass_1px = np.mean(errors <= 1.0) * 100.0
    sub_pixel_pass = np.mean(errors <= 0.5) * 100.0

    # Output manifest generation
    df['pred_x'] = pred_x_list
    df['pred_y'] = pred_y_list
    df['euclidean_error_px'] = np.round(errors, 4)
    df['latency_ms'] = np.round(latencies, 2)
    df.to_csv(output_manifest, index=False)

    print("==================================================")
    print("      APPLIED MATERIALS DRIFT-SENSE REPORT        ")
    print("==================================================")
    print(f"🎯 Mean Error (MAE):          {mean_err:.4f} px")
    print(f"🎯 Median Error:              {median_err:.4f} px")
    print(f"⚠️ Worst-Case Error:          {worst_err:.4f} px")
    print(f"⚡ Mean Latency:              {np.mean(latencies):.2f} ms")
    print("--------------------------------------------------")
    print(f"✅ Pass Rate @ 5-pixel:       {pass_5px:.1f}%")
    print(f"✅ Pass Rate @ 4-pixel:       {pass_4px:.1f}%")
    print(f"✅ Pass Rate @ 2-pixel:       {pass_2px:.1f}%")
    print(f"✅ Pass Rate @ 1-pixel:       {pass_1px:.1f}%")
    print(f"🔬 Sub-Pixel (<0.5px) Accuracy: {sub_pixel_pass:.1f}%")
    print("==================================================")
    print(f"📁 Evaluation manifest saved to '{output_manifest}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="dataset/val/labels.csv")
    args = parser.parse_args()
    run_official_evaluation(args.csv)
    