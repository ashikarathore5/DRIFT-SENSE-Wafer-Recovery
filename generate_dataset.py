import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def draw_wafer_pattern(width=1000, height=1000, shift_x=0.0, shift_y=0.0, seed=0):
    img = np.zeros((height, width), dtype=np.float32)
    rng = np.random.RandomState(seed)
    
    # Asymmetric distinctive SEM alignment features
    cx, cy = int(500 + shift_x), int(500 + shift_y)
    cv2.line(img, (cx - 150, cy), (cx + 150, cy), 220, 5)
    cv2.line(img, (cx, cy - 150), (cx, cy + 150), 220, 5)
    cv2.circle(img, (cx, cy), 40, 255, -1)
    cv2.circle(img, (cx, cy), 80, 180, 3)

    # Unique corner markers
    cv2.rectangle(img, (int(150 + shift_x), int(150 + shift_y)), (int(250 + shift_x), int(250 + shift_y)), 200, -1)
    cv2.rectangle(img, (int(700 + shift_x), int(200 + shift_y)), (int(850 + shift_x), int(230 + shift_y)), 210, -1)
    cv2.circle(img, (int(200 + shift_x), int(800 + shift_y)), 50, 190, -1)

    # Unique random chip die features
    for _ in range(20):
        rx = int(rng.randint(50, 900) + shift_x)
        ry = int(rng.randint(50, 900) + shift_y)
        rw, rh = rng.randint(30, 100), rng.randint(30, 100)
        val = int(rng.randint(100, 240))
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), val, -1)

    # Gaussian noise
    noise = rng.normal(0, 10, (height, width)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

def create_dataset(base_dir="dataset", num_train=1000, num_val=100):
    for split, count in [("train", num_train), ("val", num_val)]:
        split_dir = os.path.join(base_dir, split)
        ref_dir = os.path.join(split_dir, "ref")
        search_dir = os.path.join(split_dir, "search")
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(search_dir, exist_ok=True)

        records = []
        for i in tqdm(range(count), desc=f"generating {split}"):
            dx = float(np.random.uniform(-120.0, 120.0))
            dy = float(np.random.uniform(-120.0, 120.0))

            ref_img = draw_wafer_pattern(1000, 1000, shift_x=0.0, shift_y=0.0, seed=i+1000)
            search_img = draw_wafer_pattern(1000, 1000, shift_x=dx, shift_y=dy, seed=i+1000)

            ref_rel = f"{split}/ref/ref_{i:04d}.png"
            search_rel = f"{split}/search/search_{i:04d}.png"

            cv2.imwrite(os.path.join(base_dir, ref_rel), ref_img)
            cv2.imwrite(os.path.join(base_dir, search_rel), search_img)

            records.append({
                "id": i,
                "ref_path": ref_rel,
                "search_path": search_rel,
                "center_x": 500.0 + dx,
                "center_y": 500.0 + dy,
                "dx": dx,
                "dy": dy
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(split_dir, "labels.csv"), index=False)

if __name__ == "__main__":
    create_dataset()