"""
Training Loop for Point Transformer V3 (PTv3)
=============================================
Walks through the full pipeline:

  Dataset → DataLoader → PTv3 backbone → SegHead
      → CrossEntropyLoss → AdamW → backprop → mIoU eval

Supports two dataset modes:
  --dataset dummy   Synthetic point clouds (default, no extra files needed)
  --dataset swiss   SWISS Competition Forms (requires running convert_3dm.py first)

Runs on CPU or CUDA. On Mac, spconv falls back to the mock
defined at the top of model.py, so no GPU/CUDA required.

Usage:
    python train.py
    python train.py --dataset swiss --data-dir data_npy
    python train.py --dataset swiss --data-dir data_npy --epochs 50 --batch-size 4
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Swiss dataset (imported lazily so the dummy path has no extra dependencies)
_swiss_available = True
try:
    from dataset_swiss import SwissDataset, swiss_collate_fn
except ImportError:
    _swiss_available = False

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #

CFG = {
    # ── dummy dataset ──────────────────────────────────────────────────────
    "num_classes":      20,    # number of semantic classes (e.g. ScanNet-style)
    "num_scenes":       32,    # total dummy scenes in the dataset
    "points_per_scene": 2048,  # unique voxels per scene
    "grid_size":        0.05,  # voxel size in metres

    # ── swiss dataset ──────────────────────────────────────────────────────
    # (num_classes overridden to 26 at runtime when --dataset swiss is used)

    # ── shared training ────────────────────────────────────────────────────
    "batch_size":       2,
    "num_epochs":       5,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "val_split":        0.2,   # fraction of scenes held out for validation

    # Small model for fast CPU demo
    # (default PTv3 uses enc_channels up to 512 — too slow on CPU)
    "model_kwargs": dict(
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        enc_depths=(1, 1, 1, 2, 1),
        enc_channels=(16, 32, 64, 128, 256),
        enc_num_head=(1, 2, 4, 8, 16),
        enc_patch_size=(64, 64, 64, 64, 64),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(32, 32, 64, 128),  # dec_channels[0]=32 → head input dim
        dec_num_head=(2, 2, 4, 8),
        dec_patch_size=(64, 64, 64, 64),
        mlp_ratio=2.0,
        drop_path=0.0,
        enable_rpe=False,
        enable_flash=False,
    ),
}


# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #

class DummyPointCloudDataset(Dataset):
    """
    Synthetic point cloud dataset.

    Each scene contains `num_points` points with:
      - coord  [N, 3]  – XYZ in metres, aligned to voxel centres
      - feat   [N, 6]  – 6-D feature (e.g. XYZ + RGB normalised to [0,1])
      - label  [N]     – integer class id in [0, num_classes)

    Voxels are guaranteed to be unique per scene so that no two points
    collapse to the same grid cell after quantisation. This ensures the
    point count is preserved through sparsify() on both real and mock
    spconv backends.
    """

    def __init__(self, num_scenes, num_points, num_classes, grid_size, seed=42):
        self.num_scenes  = num_scenes
        self.num_points  = num_points
        self.num_classes = num_classes
        self.grid_size   = grid_size
        rng = np.random.default_rng(seed)
        self._data = [self._make_scene(rng) for _ in range(num_scenes)]

    def _make_scene(self, rng):
        grid_range = 64  # scene spans a 64^3 voxel grid
        # Over-sample then deduplicate to get exactly num_points unique voxels
        raw = rng.integers(0, grid_range, size=(self.num_points * 4, 3))
        _, idx = np.unique(raw, axis=0, return_index=True)
        grid = raw[idx[:self.num_points]]
        assert len(grid) >= self.num_points, (
            "Not enough unique voxels — increase grid_range or reduce num_points"
        )
        # coord: voxel centre in metres (grid_coord * grid_size + 0.5 * grid_size)
        coord = (grid.astype(np.float32) + 0.5) * self.grid_size
        feat  = rng.random((self.num_points, 6)).astype(np.float32)
        label = rng.integers(0, self.num_classes, self.num_points).astype(np.int64)
        return {
            "coord": torch.from_numpy(coord),
            "feat":  torch.from_numpy(feat),
            "label": torch.from_numpy(label),
        }

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        return self._data[idx]


def collate_fn(batch):
    """
    Merge per-scene dicts into a single batched dict.

    PTv3 expects all points concatenated along dim 0, with a `batch`
    tensor of shape [total_N] identifying which scene each point belongs
    to.  This is the standard "offset/batch" convention in Pointcept.

    See: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
    """
    coords, feats, labels, batch_ids = [], [], [], []
    for i, s in enumerate(batch):
        N = s["coord"].shape[0]
        coords.append(s["coord"])
        feats.append(s["feat"])
        labels.append(s["label"])
        batch_ids.append(torch.full((N,), i, dtype=torch.long))

    return {
        "coord":     torch.cat(coords),    # [total_N, 3]
        "feat":      torch.cat(feats),     # [total_N, 6]
        "label":     torch.cat(labels),    # [total_N]
        "batch":     torch.cat(batch_ids), # [total_N]  ← PTv3 batch index
        "grid_size": batch[0]["grid_size"] if "grid_size" in batch[0]
                     else CFG["grid_size"],
    }


# ---------------------------------------------------------------------------- #
# Segmentation head
# ---------------------------------------------------------------------------- #

class SegHead(nn.Module):
    """
    Linear segmentation head.

    Takes PTv3's per-point output features (shape [N, dec_channels[0]])
    and projects them to class logits (shape [N, num_classes]).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc   = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


# ---------------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------------- #

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Mean Intersection-over-Union (mIoU) over classes present in `target`.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    mIoU  = mean over classes where (TP+FP+FN) > 0
    """
    p = pred.cpu().numpy()
    t = target.cpu().numpy()
    ious = []
    for c in range(num_classes):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------- #
# Train / validate helpers
# ---------------------------------------------------------------------------- #

def train_one_epoch(model, head, loader, optimizer, scheduler, criterion,
                    device, epoch, num_epochs):
    model.train()
    head.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        # ── 1. Prepare input dict for PTv3 ──────────────────────────────────
        #    PTv3.forward() expects a plain dict with these keys:
        #      "coord"     – raw XYZ             [total_N, 3]
        #      "feat"      – point features       [total_N, 6]
        #      "batch"     – scene index per pt   [total_N]  (long)
        #      "grid_size" – voxel size (scalar float)
        data_dict = {
            "coord":     batch["coord"].to(device),
            "feat":      batch["feat"].to(device),
            "batch":     batch["batch"].to(device),
            "grid_size": batch["grid_size"],
        }
        labels = batch["label"].to(device)          # [total_N]

        # ── 2. PTv3 forward pass ─────────────────────────────────────────────
        #    Internally this runs:
        #      a) Point.serialization()  – assign Z-order / Hilbert codes
        #      b) Point.sparsify()       – build SparseConvTensor
        #      c) self.embedding         – initial SubMConv3d embedding
        #      d) self.enc               – 5-stage encoder (pooling + blocks)
        #      e) self.dec               – 4-stage decoder (unpooling + blocks)
        #    Returns a Point object; output.feat is [total_N, dec_channels[0]]
        output = model(data_dict)

        # ── 3. Segmentation head → logits ────────────────────────────────────
        logits = head(output.feat)                  # [total_N, num_classes]

        # ── 4. Loss ──────────────────────────────────────────────────────────
        loss = criterion(logits, labels)

        # ── 5. Backward pass ─────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()  # OneCycleLR advances per step

        total_loss += loss.item()
        elapsed = time.time() - t0
        print(
            f"  [Epoch {epoch+1}/{num_epochs}]"
            f"  step {step+1:02d}/{len(loader)}"
            f"  loss={loss.item():.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"  {elapsed:.1f}s"
        )

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, head, loader, criterion, device, num_classes):
    model.eval()
    head.eval()
    total_loss = 0.0
    all_pred, all_target = [], []

    for batch in loader:
        data_dict = {
            "coord":     batch["coord"].to(device),
            "feat":      batch["feat"].to(device),
            "batch":     batch["batch"].to(device),
            "grid_size": batch["grid_size"],
        }
        labels = batch["label"].to(device)

        output = model(data_dict)
        logits = head(output.feat)
        total_loss += criterion(logits, labels).item()

        all_pred.append(logits.argmax(dim=-1))
        all_target.append(labels)

    miou = compute_miou(
        torch.cat(all_pred),
        torch.cat(all_target),
        num_classes,
    )
    return total_loss / len(loader), miou


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="PTv3 training loop")
    parser.add_argument("--epochs",     type=int,   default=CFG["num_epochs"])
    parser.add_argument("--batch-size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument(
        "--dataset", default="dummy", choices=["dummy", "swiss"],
        help="'dummy' (synthetic, default) or 'swiss' (SWISS Competition Forms)",
    )
    parser.add_argument(
        "--data-dir", default="data_npy",
        help="Path to .npz files produced by convert_3dm.py (only used with --dataset swiss)",
    )
    parser.add_argument(
        "--label-key", default="letter", choices=["letter", "repetitions"],
        help="Which metadata field to use as class label for the swiss dataset",
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    # Note: MPS (Apple Silicon GPU) is skipped because PTv3 uses integer
    # tensor operations (e.g. torch.div with rounding_mode) that are not
    # yet fully supported on MPS. CPU is used as the fallback on Mac.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Dataset & DataLoaders ─────────────────────────────────────────────────
    if args.dataset == "swiss":
        if not _swiss_available:
            sys.exit(
                "dataset_swiss.py not found. "
                "Make sure it is in the same directory as train.py."
            )
        print(f"\n── Building SWISS Competition Forms dataset ──")
        print(f"   data dir  : {args.data_dir}")
        print(f"   label key : {args.label_key}")
        train_set = SwissDataset(
            args.data_dir, split="train",
            val_fraction=CFG["val_split"],
            label_key=args.label_key,
            augment=True,
        )
        val_set = SwissDataset(
            args.data_dir, split="val",
            val_fraction=CFG["val_split"],
            label_key=args.label_key,
        )
        num_classes = train_set.num_classes
        coll_fn     = swiss_collate_fn

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=coll_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            collate_fn=coll_fn, num_workers=0,
        )
        print(f"  {len(train_set)} train scenes / {len(val_set)} val scenes")
        print(f"  {num_classes} classes  ({train_set.label_key})")

    else:
        print("\n── Building dummy dataset ──")
        dataset = DummyPointCloudDataset(
            num_scenes=CFG["num_scenes"],
            num_points=CFG["points_per_scene"],
            num_classes=CFG["num_classes"],
            grid_size=CFG["grid_size"],
        )
        n_val   = max(1, int(len(dataset) * CFG["val_split"]))
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(0),
        )
        num_classes = CFG["num_classes"]
        coll_fn     = collate_fn

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=coll_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            collate_fn=coll_fn, num_workers=0,
        )
        print(f"  {n_train} train scenes / {n_val} val scenes")
        print(f"  {CFG['points_per_scene']} pts/scene  "
              f"× {args.batch_size} batch = "
              f"{CFG['points_per_scene'] * args.batch_size:,} pts/step")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n── Initialising PTv3 ──")
    from model import PointTransformerV3
    model = PointTransformerV3(**CFG["model_kwargs"]).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone parameters: {num_params:,}")

    head_in = CFG["model_kwargs"]["dec_channels"][0]  # = 32 with our tiny config
    head = SegHead(head_in, num_classes).to(device)
    print(f"  SegHead: {head_in} → {num_classes} classes")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    # AdamW + OneCycleLR is a strong baseline for 3-D perception tasks.
    # OneCycleLR provides a warmup phase followed by cosine annealing and is
    # advanced *per step* (not per epoch) inside train_one_epoch.
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_miou = 0.0
    history   = {"train_loss": [], "val_loss": [], "val_miou": []}

    print("\n" + "=" * 60)
    print(f"Training for {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\n{'─' * 60}")
        print(f"  EPOCH {epoch + 1} / {args.epochs}")
        print(f"{'─' * 60}")

        # ── Train ────────────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, head, train_loader, optimizer, scheduler,
            criterion, device, epoch, args.epochs,
        )

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss, val_miou = validate(
            model, head, val_loader, criterion, device, num_classes,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        print(
            f"\n  >> EPOCH {epoch+1} SUMMARY"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_mIoU={val_miou:.4f}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_miou >= best_miou:
            best_miou = val_miou
            ckpt = {
                "epoch":        epoch + 1,
                "model":        model.state_dict(),
                "head":         head.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_miou":     val_miou,
                "val_loss":     val_loss,
                # Stored so visualize_latent.py can rebuild the model exactly
                "model_kwargs": CFG["model_kwargs"],
                "num_classes":  num_classes,
                "dataset":      args.dataset,
            }
            torch.save(ckpt, "best_model.pth")
            print(f"  >> checkpoint saved  (best val mIoU = {best_miou:.4f})")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  Best val mIoU : {best_miou:.4f}")
    print("  Loss curve    :")
    for i, (tl, vl, vm) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["val_miou"])
    ):
        print(f"    epoch {i+1:02d}  train={tl:.4f}  val={vl:.4f}  mIoU={vm:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
