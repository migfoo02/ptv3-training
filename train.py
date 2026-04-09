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


class GlobalPoolHead(nn.Module):
    """
    Scene-level classification head.

    Mean-pools per-point decoder features [total_N, C] into one vector per
    scene [B, C] via scatter_mean, then projects to class logits [B, num_classes].
    This is the correct head for learning discriminative latent representations:
    the loss gradient is on the scene level, so the backbone must encode
    discriminative geometry into its per-point features rather than a constant
    scene-wide prediction.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc   = nn.Linear(in_channels, num_classes)

    def forward(
        self,
        feat: torch.Tensor,   # [total_N, C]
        batch: torch.Tensor,  # [total_N], long — scene index per point
        batch_size: int,      # B
    ) -> torch.Tensor:        # [B, num_classes]
        from torch_scatter import scatter_mean
        pooled = scatter_mean(feat, batch, dim=0, dim_size=batch_size)  # [B, C]
        return self.fc(self.norm(pooled))


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Expects L2-normalised embeddings. Pulls same-class pairs together and
    pushes different-class pairs apart on the unit hypersphere.

    Parameters
    ----------
    temperature : τ — sharpness of the softmax (default 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,  # [B, C], L2-normalised
        labels: torch.Tensor,    # [B], long
    ) -> torch.Tensor:
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)

        # Pairwise cosine similarities (unit sphere → dot product)
        sim = torch.mm(features, features.T) / self.temperature  # [B, B]

        # Positive mask: same label, different index
        lrow     = labels.unsqueeze(1)
        lcol     = labels.unsqueeze(0)
        pos_mask = (lrow == lcol).float()
        self_mask = torch.eye(B, dtype=torch.float, device=device)
        pos_mask  = pos_mask - self_mask

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Numerically stable log-sum-exp over non-self pairs
        sim_max  = sim.max(dim=1, keepdim=True).values.detach()
        exp_sim  = torch.exp(sim - sim_max) * (1 - self_mask)
        log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()


# ---------------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------------- #

def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 scene-level accuracy."""
    return (pred == target).float().mean().item()


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
                    device, epoch, num_epochs,
                    head_type="seg", supcon_criterion=None,
                    supcon_alpha=0.0):
    model.train()
    head.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        data_dict = {
            "coord":     batch["coord"].to(device),
            "feat":      batch["feat"].to(device),
            "batch":     batch["batch"].to(device),
            "grid_size": batch["grid_size"],
        }

        output = model(data_dict)

        if head_type == "global_cls":
            # ── Scene-level classification ───────────────────────────────────
            #    GlobalPoolHead mean-pools per-point features → one vector per
            #    scene → linear classifier.  Loss on scene labels [B], not [N].
            batch_size   = int(output.batch.max().item()) + 1
            logits       = head(output.feat, output.batch, batch_size)  # [B, K]
            scene_labels = batch["scene_label"].to(device)              # [B]
            loss         = criterion(logits, scene_labels)

            # Optional supervised contrastive loss (stage 2)
            if supcon_criterion is not None and supcon_alpha > 0:
                from torch_scatter import scatter_mean
                pooled = scatter_mean(
                    output.feat, output.batch, dim=0, dim_size=batch_size
                )                                                        # [B, C]
                normed = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-8)
                loss   = loss + supcon_alpha * supcon_criterion(normed, scene_labels)
        else:
            # ── Per-point segmentation (legacy / dummy dataset) ──────────────
            labels = batch["label"].to(device)                          # [total_N]
            logits = head(output.feat)                                  # [total_N, K]
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

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
def validate(model, head, loader, criterion, device, num_classes, head_type="seg"):
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

        output = model(data_dict)

        if head_type == "global_cls":
            batch_size   = int(output.batch.max().item()) + 1
            logits       = head(output.feat, output.batch, batch_size)
            scene_labels = batch["scene_label"].to(device)
            total_loss  += criterion(logits, scene_labels).item()
            all_pred.append(logits.argmax(dim=-1))
            all_target.append(scene_labels)
        else:
            labels = batch["label"].to(device)
            logits = head(output.feat)
            total_loss += criterion(logits, labels).item()
            all_pred.append(logits.argmax(dim=-1))
            all_target.append(labels)

    pred_all   = torch.cat(all_pred)
    target_all = torch.cat(all_target)
    if head_type == "global_cls":
        metric = compute_accuracy(pred_all, target_all)
    else:
        metric = compute_miou(pred_all, target_all, num_classes)
    return total_loss / len(loader), metric


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
    parser.add_argument(
        "--head-type", default="global_cls", choices=["seg", "global_cls"],
        help="'global_cls' pools per-point features to scene level (recommended); "
             "'seg' uses per-point broadcast labels (legacy)",
    )
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2],
        help="Training stage: 1=CE only, 2=CE+SupCon fine-tuning (requires --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint to resume from (required for --stage 2)",
    )
    parser.add_argument(
        "--supcon-alpha", type=float, default=0.5,
        help="Max weight for SupCon loss in stage 2 (ramped over first 10 epochs)",
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

    head_in   = CFG["model_kwargs"]["dec_channels"][0]  # 32 with small config
    head_type = args.head_type if args.dataset == "swiss" else "seg"
    if head_type == "global_cls":
        head = GlobalPoolHead(head_in, num_classes).to(device)
        print(f"  GlobalPoolHead: {head_in}→pool→{num_classes} classes")
    else:
        head = SegHead(head_in, num_classes).to(device)
        print(f"  SegHead: {head_in}→{num_classes} classes")

    # ── Load checkpoint for stage 2 ───────────────────────────────────────────
    if args.stage == 2:
        if args.checkpoint is None:
            sys.exit("--stage 2 requires --checkpoint pointing to stage-1 weights.")
        ckpt_data = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt_data["model"])
        head.load_state_dict(ckpt_data["head"])
        print(f"  Loaded stage-1 checkpoint: {args.checkpoint}  "
              f"(epoch {ckpt_data.get('epoch','?')})")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=CFG["weight_decay"],
    )
    if args.stage == 2:
        # Fine-tuning: cosine decay from current LR, no warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(train_loader),
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    if args.dataset == "swiss" and head_type == "global_cls":
        # Class-weighted CE: compensates for imbalance (U=12 vs K=35 samples)
        from collections import Counter
        counts = Counter(train_set[i]["scene_label"] for i in range(len(train_set)))
        weight_vec = torch.tensor(
            [len(train_set) / (num_classes * max(counts.get(c, 1), 1))
             for c in range(num_classes)],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_vec)
    else:
        criterion = nn.CrossEntropyLoss()

    supcon_criterion = SupConLoss(temperature=0.07).to(device) if args.stage == 2 else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_metric = 0.0
    metric_name = "acc" if head_type == "global_cls" else "mIoU"
    history     = {"train_loss": [], "val_loss": [], f"val_{metric_name}": []}

    print("\n" + "=" * 60)
    print(f"Training stage {args.stage} for {args.epochs} epochs  "
          f"(head={head_type}, metric={metric_name})")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\n{'─' * 60}")
        print(f"  EPOCH {epoch + 1} / {args.epochs}")
        print(f"{'─' * 60}")

        # SupCon alpha ramps from 0 → supcon_alpha over the first 10 epochs of stage 2
        current_alpha = (
            min(args.supcon_alpha, args.supcon_alpha * (epoch + 1) / 10)
            if args.stage == 2 else 0.0
        )

        train_loss = train_one_epoch(
            model, head, train_loader, optimizer, scheduler,
            criterion, device, epoch, args.epochs,
            head_type=head_type,
            supcon_criterion=supcon_criterion,
            supcon_alpha=current_alpha,
        )

        val_loss, val_metric = validate(
            model, head, val_loader, criterion, device, num_classes,
            head_type=head_type,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history[f"val_{metric_name}"].append(val_metric)

        print(
            f"\n  >> EPOCH {epoch+1} SUMMARY"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_{metric_name}={val_metric:.4f}"
            + (f"  supcon_α={current_alpha:.3f}" if args.stage == 2 else "")
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_metric >= best_metric:
            best_metric = val_metric
            ckpt = {
                "epoch":        epoch + 1,
                "model":        model.state_dict(),
                "head":         head.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_miou":     val_metric,   # keep key for visualize_latent.py compat
                "val_loss":     val_loss,
                "model_kwargs": CFG["model_kwargs"],
                "num_classes":  num_classes,
                "dataset":      args.dataset,
                "head_type":    head_type,
                "stage":        args.stage,
            }
            torch.save(ckpt, "best_model.pth")
            print(f"  >> checkpoint saved  (best val {metric_name} = {best_metric:.4f})")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  Best val {metric_name}: {best_metric:.4f}")
    print("  Curve:")
    for i, (tl, vl, vm) in enumerate(
        zip(history["train_loss"], history["val_loss"], history[f"val_{metric_name}"])
    ):
        print(f"    epoch {i+1:02d}  train={tl:.4f}  val={vl:.4f}  {metric_name}={vm:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
