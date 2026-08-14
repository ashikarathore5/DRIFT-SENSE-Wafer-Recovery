import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2

# --------------------------------------------------------------------------
# Fast Lightweight Siamese CNN
# --------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(32, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class DriftSenseCNN(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.backbone = FeatureExtractor(embed_dim)

    def forward(self, search, template):
        f_search = self.backbone(search)       # (B, 16, 64, 64)
        f_template = self.backbone(template)   # (B, 16, 64, 64)
        
        f_template = F.adaptive_avg_pool2d(f_template, (12, 12))
        f_search_norm = F.normalize(f_search, p=2, dim=1)
        f_template_norm = F.normalize(f_template, p=2, dim=1)
        
        B, C, H, W = f_search_norm.shape
        f_search_grouped = f_search_norm.view(1, B * C, H, W)
        
        resp = F.conv2d(f_search_grouped, f_template_norm, groups=B)
        resp = resp.permute(1, 0, 2, 3)
        return resp

# --------------------------------------------------------------------------
# Fast CPU Dataset (Resizes on the fly)
# --------------------------------------------------------------------------
class WaferDatasetFast(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        csv_dir = os.path.dirname(csv_path)
        self.root_dir = os.path.dirname(csv_dir) if csv_dir.endswith(('train', 'val')) else csv_dir
        self.x_col = 'center_x' if 'center_x' in self.df.columns else 'gt_x'
        self.y_col = 'center_y' if 'center_y' in self.df.columns else 'gt_y'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s_rel, r_rel = row['search_path'], row['ref_path']

        s_path = os.path.join(self.root_dir, s_rel) if not os.path.isabs(s_rel) else s_rel
        r_path = os.path.join(self.root_dir, r_rel) if not os.path.isabs(r_rel) else r_rel

        search_img = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)

        if search_img is None or ref_img is None:
            raise FileNotFoundError(f"Cannot read: {s_path} or {r_path}")

        # Downsample to 256x256 for blazing fast CPU computation
        search_small = cv2.resize(search_img, (256, 256), interpolation=cv2.INTER_AREA)
        ref_small = cv2.resize(ref_img, (256, 256), interpolation=cv2.INTER_AREA)

        s_tensor = torch.from_numpy(search_small).float().unsqueeze(0) / 255.0
        r_tensor = torch.from_numpy(ref_small).float().unsqueeze(0) / 255.0

        gt_x, gt_y = float(row[self.x_col]), float(row[self.y_col])
        tx = int(np.clip((gt_x / 1000.0) * 53, 0, 52))
        ty = int(np.clip((gt_y / 1000.0) * 53, 0, 52))

        y_grid, x_grid = np.ogrid[:53, :53]
        dist_sq = (x_grid - tx)**2 + (y_grid - ty)**2
        heatmap = np.exp(-dist_sq / (2 * (1.5**2))).astype(np.float32)

        return s_tensor, r_tensor, torch.from_numpy(heatmap).unsqueeze(0)

# --------------------------------------------------------------------------
# Fast Training Loop (3 Epochs)
# --------------------------------------------------------------------------
def train_model(train_csv="dataset/train/labels.csv", epochs=3, batch_size=32, lr=0.003):
    if not os.path.exists(train_csv):
        print(f"❌ Error: {train_csv} not found.")
        return

    device = torch.device("cpu")
    print(f"🚀 Acceleration Device: {device} (Fast CPU Mode)")

    dataset = WaferDatasetFast(train_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DriftSenseCNN(embed_dim=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"📦 Training on {len(dataset)} pairs...")

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0

        for search, ref, target_heatmaps in dataloader:
            optimizer.zero_grad()
            resp = model(search, ref)
            loss = criterion(resp, target_heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - t0
        avg_loss = running_loss / len(dataloader)
        print(f"[{epoch+1:02d}/{epochs:02d}] Train Loss: {avg_loss:.4f} | Epoch Time: {epoch_time:.1f}s")

    torch.save(model.state_dict(), "drift_sense_model.pth")
    print("--------------------------------------------------")
    print("✅ Fast Training Complete! Model saved to 'drift_sense_model.pth'")
    print("--------------------------------------------------")

if __name__ == "__main__":
    train_model()
    