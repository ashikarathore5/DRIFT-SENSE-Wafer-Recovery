"""
generate_dataset.py (Upgraded Photorealistic SEM Engine)
========================================================
Generates realistic SEM images matching actual chip design physics:
- Secondary Electron Edge Blooming
- Electron Beam PSF Blur
- Precise Chip Layouts (DRAM, FinFET, Via Array)
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None):
        total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        for i, item in enumerate(iterable):
            yield item

IMG_SIZE = 1000
REF_MAG = 100
SEARCH_MAG = 10
SCALE_RATIO = REF_MAG / SEARCH_MAG
SEARCH_FOV = float(IMG_SIZE)
REF_FOV = SEARCH_FOV / SCALE_RATIO
EDGE_MARGIN = 70

PATTERN_TYPES = ["dram_grid", "finfet", "via_array"]
LANDMARK_BY_PATTERN = {
    "dram_grid": ["cut_wordline", "missing_contact"],
    "finfet": ["gate_break"],
    "via_array": ["missing_contact"],
}

_SEARCH_XX, _SEARCH_YY = None, None
_SHADE_GX, _SHADE_GY = None, None

@dataclass
class PatternParams:
    kind: str
    period_x: float
    period_y: float
    width_x: float
    width_y: float
    dot_radius: float = 0.0

def coord_grid(center: Tuple[float, float], fov: float, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    half = fov / 2.0
    step = fov / out_size
    xs = np.linspace(center[0] - half, center[0] + half, out_size, endpoint=False) + step / 2.0
    ys = np.linspace(center[1] - half, center[1] + half, out_size, endpoint=False) + step / 2.0
    xx, yy = np.meshgrid(xs, ys)
    return xx.astype(np.float32), yy.astype(np.float32)

def _init_module_level_grids():
    global _SEARCH_XX, _SEARCH_YY, _SHADE_GX, _SHADE_GY
    _SEARCH_XX, _SEARCH_YY = coord_grid((SEARCH_FOV / 2, SEARCH_FOV / 2), SEARCH_FOV, IMG_SIZE)
    _SHADE_GX, _SHADE_GY = np.meshgrid(
        np.linspace(0, 1, IMG_SIZE, dtype=np.float32),
        np.linspace(0, 1, IMG_SIZE, dtype=np.float32),
    )

def random_pattern_params(rng: np.random.Generator) -> PatternParams:
    kind = rng.choice(PATTERN_TYPES)
    if kind == "dram_grid":
        return PatternParams(kind, rng.uniform(20, 30), rng.uniform(24, 36),
                             rng.uniform(4.0, 7.0), rng.uniform(5.0, 8.0))
    if kind == "finfet":
        return PatternParams(kind, rng.uniform(12, 18), rng.uniform(32, 48),
                             rng.uniform(3.0, 5.0), rng.uniform(10.0, 16.0))
    return PatternParams(kind, rng.uniform(22, 34), rng.uniform(22, 34),
                          0.0, 0.0, dot_radius=rng.uniform(5.0, 8.0))

def render_pattern_cad(xx: np.ndarray, yy: np.ndarray, p: PatternParams) -> np.ndarray:
    """Renders clean CAD geometry mask before SEM physical effects."""
    if p.kind == "dram_grid":
        cad = np.zeros(xx.shape, dtype=np.float32)
        vert = (xx % p.period_x) < p.width_x
        horz = (yy % p.period_y) < p.width_y
        cad[vert] = 0.5
        cad[horz] = 0.7
        cad[vert & horz] = 1.0
        return cad

    if p.kind == "finfet":
        cad = np.zeros(xx.shape, dtype=np.float32)
        fins = (xx % p.period_x) < p.width_x
        gates = (yy % p.period_y) < p.width_y
        cad[fins] = 0.4
        cad[gates] = 0.7
        cad[fins & gates] = 1.0
        return cad

    cad = np.zeros(xx.shape, dtype=np.float32)
    gx = np.round(xx / p.period_x) * p.period_x
    gy = np.round(yy / p.period_y) * p.period_y
    dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
    cad[dist < p.dot_radius] = 1.0
    return cad

def snap_to_feature(raw_center: Tuple[float, float], landmark_type: str, p: PatternParams) -> Tuple[float, float]:
    x0, y0 = raw_center
    if landmark_type in ("cut_wordline", "gate_break"):
        y0 = round(y0 / p.period_y) * p.period_y + p.width_y / 2.0
    elif landmark_type == "missing_contact":
        if p.kind == "via_array":
            x0 = round(x0 / p.period_x) * p.period_x
            y0 = round(y0 / p.period_y) * p.period_y
        else:
            x0 = round(x0 / p.period_x) * p.period_x + p.width_x / 2.0
            y0 = round(y0 / p.period_y) * p.period_y + p.width_y / 2.0
    x0 = float(np.clip(x0, EDGE_MARGIN, SEARCH_FOV - EDGE_MARGIN))
    y0 = float(np.clip(y0, EDGE_MARGIN, SEARCH_FOV - EDGE_MARGIN))
    return x0, y0

def inject_landmark(img: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                    center: Tuple[float, float], landmark_type: str,
                    p: PatternParams, rng: np.random.Generator) -> np.ndarray:
    cx, cy = center
    if landmark_type == "missing_contact":
        radius = (p.dot_radius if p.kind == "via_array" else max(p.width_x, p.width_y) * 1.2) + 2.0
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img[dist < radius] = 0.0
    elif landmark_type in ("cut_wordline", "gate_break"):
        run_gap = p.width_y * 2.2 + rng.uniform(2.0, 4.0)
        thick = p.width_y * 1.6
        mask = (np.abs(xx - cx) < run_gap / 2.0) & (np.abs(yy - cy) < thick / 2.0)
        img[mask] = 0.0
    return img

def apply_sem_physics(cad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies realistic SEM physics: Beam Blur + Edge Blooming Effect."""
    # 1. Secondary Electron Edge Blooming (Gradient magnitude)
    gx = cv2.Sobel(cad, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cad, cv2.CV_32F, 0, 1, ksize=3)
    edge_bloom = np.sqrt(gx**2 + gy**2)
    
    # 2. Blend Base Materials + Edge Highlights
    sem = 0.15 + 0.5 * cad + 0.35 * edge_bloom
    
    # 3. Electron Beam Blur (PSF)
    sem = cv2.GaussianBlur(sem, (5, 5), 0.8)
    return sem

def add_shading(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(0.08, 0.18)
    grad = np.cos(angle) * (_SHADE_GX - 0.5) + np.sin(angle) * (_SHADE_GY - 0.5)
    grad = grad / (np.abs(grad).max() + 1e-6)
    return img * (1.0 + strength * grad)

def add_sem_noise(img: np.ndarray, rng: np.random.Generator,
                  poisson_peak=(30, 70), gaussian_sigma=(0.015, 0.03)) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    peak = rng.uniform(*poisson_peak)
    noisy = rng.poisson(img * peak) / peak
    sigma = rng.uniform(*gaussian_sigma)
    noisy = noisy + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0.0, 1.0)

def generate_pair(rng: np.random.Generator):
    params = random_pattern_params(rng)
    landmark_type = rng.choice(LANDMARK_BY_PATTERN[params.kind])
    raw_center = (rng.uniform(EDGE_MARGIN, SEARCH_FOV - EDGE_MARGIN),
                  rng.uniform(EDGE_MARGIN, SEARCH_FOV - EDGE_MARGIN))
    center = snap_to_feature(raw_center, landmark_type, params)

    # Render Search CAD -> Inject Defect -> Apply SEM Physics
    search_cad = render_pattern_cad(_SEARCH_XX, _SEARCH_YY, params)
    search_cad = inject_landmark(search_cad, _SEARCH_XX, _SEARCH_YY, center, landmark_type, params, rng)
    search_sem = apply_sem_physics(search_cad, rng)
    search_img = add_sem_noise(add_shading(search_sem, rng), rng)

    # Render Ref CAD -> Inject Defect -> Apply SEM Physics
    xx_r, yy_r = coord_grid(center, REF_FOV, IMG_SIZE)
    ref_cad = render_pattern_cad(xx_r, yy_r, params)
    ref_cad = inject_landmark(ref_cad, xx_r, yy_r, center, landmark_type, params, rng)
    ref_sem = apply_sem_physics(ref_cad, rng)
    ref_img = add_sem_noise(add_shading(ref_sem, rng), rng)

    search_u8 = (search_img * 255).astype(np.uint8)
    ref_u8 = (ref_img * 255).astype(np.uint8)
    return search_u8, ref_u8, center, params.kind, landmark_type

def _init_worker():
    cv2.setNumThreads(1)
    _init_module_level_grids()

def _worker_generate(args):
    i, seed, split, out_root_str = args
    out_root = Path(out_root_str)
    rng = np.random.default_rng(seed)
    search_img, ref_img, center, pattern, landmark = generate_pair(rng)

    fname = f"{i:06d}.png"
    search_path = f"{split}/search/{fname}"
    ref_path = f"{split}/reference/{fname}"

    cv2.imwrite(str(out_root / search_path), search_img)
    cv2.imwrite(str(out_root / ref_path), ref_img)

    return {
        "id": i,
        "search_path": search_path,
        "ref_path": ref_path,
        "center_x": center[0],
        "center_y": center[1],
        "pattern": pattern,
        "landmark": landmark,
    }

def generate_split(split: str, n_samples: int, seed_base: int, out_root: Path):
    search_dir = out_root / split / "search"
    ref_dir = out_root / split / "reference"
    search_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(i, seed_base + i, split, str(out_root)) for i in range(n_samples)]
    rows = []
    t0 = time.time()

    num_workers = os.cpu_count() or 1
    chunksize = max(1, n_samples // (num_workers * 4))
    with multiprocessing.Pool(processes=num_workers, initializer=_init_worker) as pool:
        for row in tqdm(pool.imap_unordered(_worker_generate, tasks, chunksize=chunksize),
                         desc=f"generating {split}", total=n_samples):
            rows.append(row)

    rows.sort(key=lambda x: x["id"])

    csv_path = out_root / split / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{split}] wrote {n_samples} pairs -> {csv_path} ({time.time() - t0:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense synthetic dataset generator")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _init_module_level_grids()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    generate_split("train", args.train_samples, args.seed * 1000, out_root)
    generate_split("val", args.val_samples, args.seed * 1000 + 900_000, out_root)

    meta = {
        "img_size": IMG_SIZE,
        "ref_mag": REF_MAG,
        "search_mag": SEARCH_MAG,
        "scale_ratio": SCALE_RATIO,
        "pattern_types": PATTERN_TYPES,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "seed": args.seed,
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_root / 'meta.json'}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()