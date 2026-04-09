"""
dataset_swiss.py
================
PyTorch Dataset for the SWISS Competition Forms point clouds.

Prerequisite: run convert_3dm.py first to produce the .npz files.

Label scheme
------------
Each .npz file encodes three metadata arrays:

  letter      int64 [1]   – class label 0-25 (A=0 … Z=25)
  repetitions int64 [1]   – letter repetitions 1-4 (A / AA / AAA / AAAA)
  num         int64 [1]   – variant number within the group (1, 2, 3 …)

The default label used for training is `letter` (26 classes).
To train a "complexity" classifier instead pass label_key="repetitions".

Sample format (what __getitem__ returns)
-----------------------------------------
  coord        float32 [N, 3]  – normalised XYZ, unit sphere
  feat         float32 [N, 6]  – cat([coord, normal])  ← PTv3 input
  label        int64   [N]     – per-point class label (broadcast from scene)
  scene_label  int              – scalar scene-level label
  filename     str              – file stem (e.g. 'AAAA3')

Collate function
----------------
swiss_collate_fn merges a list of per-scene dicts into the batched format
expected by PTv3 (all points concatenated, with a `batch` tensor).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Human-readable class names ───────────────────────────────────────────────
LETTER_NAMES = [chr(ord("A") + i) for i in range(26)]  # ['A', 'B', …, 'Z']


# ── Dataset ──────────────────────────────────────────────────────────────────

class SwissDataset(Dataset):
    """
    Loads pre-converted .npz point clouds for the SWISS Competition Forms.

    Parameters
    ----------
    data_dir     : path to the directory of .npz files (from convert_3dm.py)
    split        : 'train', 'val', or 'all'
    val_fraction : fraction of files held out for validation (default 0.2)
    seed         : random seed for the train/val split (default 42)
    label_key    : 'letter' | 'repetitions'  (which metadata field to use as label)
    num_points   : if set, sub-sample / over-sample to exactly this many points
    augment      : if True, apply random jitter + random flip during training
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "all",
        val_fraction: float = 0.2,
        seed: int = 42,
        label_key: str = "letter",
        num_points: int = None,
        augment: bool = False,
    ):
        self.data_dir  = Path(data_dir)
        self.label_key = label_key
        self.num_points = num_points
        self.augment   = augment and (split == "train")

        all_files = sorted(self.data_dir.glob("*.npz"))
        if not all_files:
            raise FileNotFoundError(
                f"No .npz files found in '{data_dir}'.\n"
                "Run  python convert_3dm.py  first."
            )

        # Deterministic stratified-ish split (shuffle by seed, take first n_val)
        rng = np.random.default_rng(seed)
        shuffled = [all_files[i] for i in rng.permutation(len(all_files))]
        n_val    = max(1, int(len(all_files) * val_fraction))

        if split == "train":
            self.files = shuffled[n_val:]
        elif split == "val":
            self.files = shuffled[:n_val]
        else:
            self.files = shuffled

        # Derive num_classes from label_key
        if label_key == "letter":
            self.num_classes = 26
            self.class_names = LETTER_NAMES
        elif label_key == "repetitions":
            self.num_classes = 4
            self.class_names = ["single", "double", "triple", "quadruple"]
        else:
            raise ValueError(f"Unknown label_key '{label_key}'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        coord  = data["coord"].astype(np.float32)   # [N, 3]
        normal = data["normal"].astype(np.float32)  # [N, 3]

        if self.label_key == "letter":
            label_scalar = int(data["letter"][0])
        else:
            label_scalar = int(data["repetitions"][0]) - 1  # 0-indexed

        # Optional resampling
        if self.num_points is not None:
            N = coord.shape[0]
            if N >= self.num_points:
                idx_pts = np.random.choice(N, self.num_points, replace=False)
            else:
                idx_pts = np.random.choice(N, self.num_points, replace=True)
            coord  = coord[idx_pts]
            normal = normal[idx_pts]

        # Optional augmentation (training only)
        if self.augment:
            coord, normal = _augment(coord, normal)

        coord  = torch.from_numpy(coord)
        normal = torch.from_numpy(normal)
        feat   = torch.cat([coord, normal], dim=-1)  # [N, 6]

        N      = coord.shape[0]
        labels = torch.full((N,), label_scalar, dtype=torch.long)

        return {
            "coord":       coord,         # [N, 3]
            "feat":        feat,          # [N, 6]
            "label":       labels,        # [N]
            "scene_label": label_scalar,  # scalar int
            "filename":    self.files[idx].stem,
        }

    # ── Convenience ──────────────────────────────────────────────────────────

    def class_name(self, label: int) -> str:
        if 0 <= label < len(self.class_names):
            return self.class_names[label]
        return str(label)


# ── Augmentation ─────────────────────────────────────────────────────────────

def _augment(coord: np.ndarray, normal: np.ndarray):
    """
    Training augmentation:
      - Gaussian jitter (σ=0.01, clipped to ±0.05)
      - Random horizontal flip (50 % chance each axis)
      - Random Z-axis rotation (uniform 0–2π): invariance to plan-view azimuth
      - Anisotropic scale [0.85, 1.15]: invariance to letterform stretch
    """
    jitter = np.clip(np.random.normal(0, 0.01, coord.shape), -0.05, 0.05)
    coord  = coord + jitter.astype(np.float32)

    # Random flip on X and Y axes
    for axis in (0, 1):
        if np.random.random() < 0.5:
            coord[:, axis]  = -coord[:, axis]
            normal[:, axis] = -normal[:, axis]

    # Z-axis rotation: rotate XY plane, preserve Z
    theta = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(theta), np.sin(theta)
    R_xy  = np.array([[c, -s], [s, c]], dtype=np.float32)
    coord[:, :2]  = coord[:, :2]  @ R_xy.T
    normal[:, :2] = normal[:, :2] @ R_xy.T

    # Anisotropic scale per axis [0.85, 1.15] — renormalise normals afterward
    scale  = np.random.uniform(0.85, 1.15, size=(1, 3)).astype(np.float32)
    coord  = coord * scale
    normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)

    return coord, normal


# ── Collate function ─────────────────────────────────────────────────────────

def swiss_collate_fn(batch):
    """
    Merge per-scene dicts into a single batched dict for PTv3.

    PTv3 expects all points concatenated along dim 0, with a `batch`
    tensor of shape [total_N] that identifies which scene each point
    belongs to.

    Per-scene scalars / strings are collected into lists.
    """
    coords      = []
    feats       = []
    labels      = []
    batch_ids   = []
    scene_labels = []
    filenames   = []

    for i, s in enumerate(batch):
        N = s["coord"].shape[0]
        coords.append(s["coord"])
        feats.append(s["feat"])
        labels.append(s["label"])
        batch_ids.append(torch.full((N,), i, dtype=torch.long))
        scene_labels.append(s["scene_label"])
        filenames.append(s["filename"])

    return {
        "coord":        torch.cat(coords),      # [total_N, 3]
        "feat":         torch.cat(feats),       # [total_N, 6]
        "label":        torch.cat(labels),      # [total_N]
        "batch":        torch.cat(batch_ids),   # [total_N]
        "grid_size":    0.05,                   # voxel size in metres
        "scene_label":  torch.tensor(scene_labels, dtype=torch.long),  # [B]
        "filename":     filenames,              # list[str]
    }


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_npy")
    parser.add_argument("--num-points", type=int, default=None)
    args = parser.parse_args()

    ds = SwissDataset(args.data_dir, split="all", num_points=args.num_points)
    print(f"Dataset: {len(ds)} scenes  |  {ds.num_classes} classes")

    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=swiss_collate_fn)
    batch  = next(iter(loader))

    print(f"coord  : {batch['coord'].shape}  {batch['coord'].dtype}")
    print(f"feat   : {batch['feat'].shape}   {batch['feat'].dtype}")
    print(f"label  : {batch['label'].shape}  {batch['label'].dtype}")
    print(f"batch  : {batch['batch'].shape}")
    print(f"scenes : {batch['scene_label'].tolist()}")
    print(f"files  : {batch['filename']}")

    # Class distribution
    from collections import Counter
    all_labels = [ds[i]["scene_label"] for i in range(len(ds))]
    counts = Counter(all_labels)
    print(f"\nClass distribution ({ds.num_classes} classes):")
    for label in sorted(counts):
        print(f"  {ds.class_name(label):12s}  ({label:2d})  {counts[label]:4d} scenes")
