import argparse
import os
import cv2
from localize import match_drift


def main():
    parser = argparse.ArgumentParser(description="DRIFT-SENSE Single Pair Inference")
    parser.add_argument(
        "--ref_path", type=str, default="dataset/val/ref/ref_0000.png"
    )
    parser.add_argument(
        "--search_path", type=str, default="dataset/val/search/search_0000.png"
    )
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.ref_path) or not os.path.exists(
        args.search_path
    ):
        print("❌ Image paths not found! Generating dataset sample...")
        os.system("python generate_dataset.py")

    ref_img = cv2.imread(args.ref_path)
    search_img = cv2.imread(args.search_path)

    pred_x, pred_y = match_drift(ref_img, search_img)
    print(f"\n🎯 DRIFT LOCALIZATION RESULT:")
    print(f"   Reference Image : {args.ref_path}")
    print(f"   Search Image    : {args.search_path}")
    print(f"   Predicted Drift Center: ({pred_x:.3f}, {pred_y:.3f})\n")


if __name__ == "__main__":
    main()