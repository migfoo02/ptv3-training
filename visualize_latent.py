"""
visualize_latent.py
===================
Extract per-scene latent embeddings from a trained PTv3 model and plot
them in 2-D, coloured by class label.

Two embedding strategies are supported:

  --source decoder (default)
      Mean-pool the decoder's full-resolution per-point features
      (output.feat, shape [N, dec_channels[0]]) over each scene.
      Requires the model to have been trained with the decoder enabled.

  --source encoder
      Register a forward hook on model.enc to capture the encoder
      bottleneck features (coarsest resolution, shape [M, enc_channels[-1]])
      and mean-pool them per scene.  This is a more compressed, abstract
      representation of scene structure.

Two dimensionality-reduction methods are supported (tried in order):
  1. UMAP  (pip install umap-learn)    – best topology preservation
  2. t-SNE (pip install scikit-learn)  – classic
  3. PCA   (numpy only)               – always available, linear baseline

Usage
-----
# After training with the swiss dataset:
python visualize_latent.py --checkpoint best_model.pth --data-dir data_npy

# Encoder bottleneck embeddings via t-SNE:
python visualize_latent.py --checkpoint best_model.pth --data-dir data_npy \\
       --source encoder --method tsne

# Save the figure without displaying it:
python visualize_latent.py --checkpoint best_model.pth --data-dir data_npy \\
       --output latent_space.png --no-show

# Also save the raw embeddings for downstream analysis:
python visualize_latent.py --checkpoint best_model.pth --data-dir data_npy \\
       --save-embeddings embeddings.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Local imports ─────────────────────────────────────────────────────────────
from dataset_swiss import SwissDataset, swiss_collate_fn, LETTER_NAMES
from model import PointTransformerV3


# ── Model config (must match checkpoint) ─────────────────────────────────────
# These defaults match the small "CPU demo" config in train.py.
# They are overridden if the checkpoint stores model_kwargs.
DEFAULT_MODEL_KWARGS = dict(
    in_channels   = 6,
    order         = ("z", "z-trans", "hilbert", "hilbert-trans"),
    enc_depths    = (1, 1, 1, 2, 1),
    enc_channels  = (16, 32, 64, 128, 256),
    enc_num_head  = (1, 2, 4, 8, 16),
    enc_patch_size= (64, 64, 64, 64, 64),
    dec_depths    = (1, 1, 1, 1),
    dec_channels  = (32, 32, 64, 128),
    dec_num_head  = (2, 2, 4, 8),
    dec_patch_size= (64, 64, 64, 64),
    mlp_ratio     = 2.0,
    drop_path     = 0.0,
    enable_rpe    = False,
    enable_flash  = False,
)


# ── Embedding extraction ──────────────────────────────────────────────────────

def _extract_decoder(model, loader, device):
    """
    Mean-pool the decoder output features per scene.

    Returns
    -------
    embeddings  float32 ndarray  [num_scenes, dec_channels[0]]
    labels      int64   ndarray  [num_scenes]
    filenames   list[str]
    """
    model.eval()
    embeddings, labels, filenames = [], [], []

    with torch.no_grad():
        for batch in loader:
            data_dict = {
                "coord":     batch["coord"].to(device),
                "feat":      batch["feat"].to(device),
                "batch":     batch["batch"].to(device),
                "grid_size": batch["grid_size"],
            }
            output     = model(data_dict)          # Point object
            feat       = output.feat               # [total_N, C]
            batch_ids  = data_dict["batch"]        # [total_N]
            batch_size = int(batch_ids.max().item()) + 1

            for i in range(batch_size):
                mask = (batch_ids == i)
                emb  = feat[mask].mean(0).cpu().numpy()   # [C]
                embeddings.append(emb)

            labels.extend(batch["scene_label"].tolist())
            filenames.extend(batch["filename"])

    return np.array(embeddings, dtype=np.float32), np.array(labels), filenames


def _extract_encoder(model, loader, device):
    """
    Hook into model.enc to capture the encoder bottleneck (coarsest scale)
    and mean-pool per scene.

    Returns same format as _extract_decoder.
    """
    model.eval()
    embeddings, labels, filenames = [], [], []

    _enc_cache = {}

    def _hook(module, inp, out):
        # out is a Point object at the encoder bottleneck resolution
        _enc_cache["feat"]  = out.feat.detach()   # [M, enc_channels[-1]]
        _enc_cache["batch"] = out.batch.detach()  # [M]

    handle = model.enc.register_forward_hook(_hook)

    with torch.no_grad():
        for batch in loader:
            data_dict = {
                "coord":     batch["coord"].to(device),
                "feat":      batch["feat"].to(device),
                "batch":     batch["batch"].to(device),
                "grid_size": batch["grid_size"],
            }
            model(data_dict)

            feat      = _enc_cache["feat"]          # [M, C]
            batch_ids = _enc_cache["batch"]         # [M]
            batch_size = int(batch_ids.max().item()) + 1

            for i in range(batch_size):
                mask = (batch_ids == i)
                emb  = feat[mask].mean(0).cpu().numpy()
                embeddings.append(emb)

            labels.extend(batch["scene_label"].tolist())
            filenames.extend(batch["filename"])

    handle.remove()
    return np.array(embeddings, dtype=np.float32), np.array(labels), filenames


# ── Dimensionality reduction ──────────────────────────────────────────────────

def _reduce_umap(X, seed):
    import umap  # type: ignore
    reducer = umap.UMAP(n_components=2, random_state=seed, verbose=False)
    return reducer.fit_transform(X)


def _reduce_tsne(X, seed):
    from sklearn.manifold import TSNE  # type: ignore
    perplexity = min(30, max(5, len(X) // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=1000, init="pca")
    return tsne.fit_transform(X)


def _reduce_pca(X):
    """Linear PCA via SVD (numpy only, always available)."""
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return (X_c @ Vt[:2].T)


def reduce_2d(X: np.ndarray, method: str, seed: int = 0):
    """
    Reduce [N, D] embeddings to [N, 2].

    method : 'umap' | 'tsne' | 'pca' | 'auto'
             'auto' tries umap → tsne → pca in order.
    """
    if method in ("umap", "auto"):
        try:
            coords = _reduce_umap(X, seed)
            print(f"  Dimensionality reduction: UMAP")
            return coords, "UMAP"
        except ImportError:
            if method == "umap":
                sys.exit("umap-learn not installed.  pip install umap-learn")

    if method in ("tsne", "auto"):
        try:
            coords = _reduce_tsne(X, seed)
            print(f"  Dimensionality reduction: t-SNE")
            return coords, "t-SNE"
        except ImportError:
            if method == "tsne":
                sys.exit("scikit-learn not installed.  pip install scikit-learn")

    # PCA fallback (always works)
    coords = _reduce_pca(X)
    print(f"  Dimensionality reduction: PCA (numpy fallback — install umap-learn or scikit-learn for better results)")
    return coords, "PCA"


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_latent(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    filenames: list,
    method_name: str,
    source: str,
    class_names: list,
    output_path: str = None,
    show: bool = True,
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        sys.exit("matplotlib not installed.  pip install matplotlib")

    num_classes = len(class_names)
    cmap        = plt.colormaps.get_cmap("tab20").resampled(num_classes)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each class as a separate scatter series (for legend)
    unique_labels = sorted(set(labels.tolist()))
    for cls in unique_labels:
        mask  = labels == cls
        color = cmap(cls / max(num_classes - 1, 1))
        name  = class_names[cls] if cls < len(class_names) else str(cls)
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[color],
            label=name,
            alpha=0.75,
            s=40,
            edgecolors="none",
        )

    # Annotate a few points with their filename
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(filenames), size=min(20, len(filenames)), replace=False)
    for i in sample_idx:
        ax.annotate(
            filenames[i],
            xy=(coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=5,
            alpha=0.6,
        )

    ax.set_title(
        f"PTv3 Latent Space — {method_name}  ({source} features, {len(labels)} scenes)",
        fontsize=13,
    )
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    ax.legend(
        title="Letter",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        ncol=2,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Quantitative metrics ──────────────────────────────────────────────────────

def compute_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Quantitative embedding quality metrics.

    Returns dict with keys: silhouette, knn_acc, within_dist, between_dist,
    dist_ratio (within/between, lower = better), class_dist_matrix [K, K].
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import normalize

    X = normalize(embeddings, norm="l2")

    sil = silhouette_score(X, labels, metric="euclidean")

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    cv  = min(5, min(np.bincount(labels).tolist()))  # can't have more folds than min class count
    cv  = max(2, cv)
    knn_acc = float(cross_val_score(knn, X, labels, cv=cv, scoring="accuracy").mean())

    num_classes = int(labels.max()) + 1
    within_dists = []
    for c in range(num_classes):
        pts = X[labels == c]
        if len(pts) < 2:
            continue
        d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        within_dists.append(d[np.triu_indices(len(pts), k=1)].mean())
    within_dist = float(np.mean(within_dists)) if within_dists else 0.0

    dist_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            pi, pj = X[labels == i], X[labels == j]
            if len(pi) == 0 or len(pj) == 0:
                continue
            dist_matrix[i, j] = np.linalg.norm(pi[:, None] - pj[None, :], axis=-1).mean()

    between_dist = float(dist_matrix[dist_matrix > 0].mean()) if (dist_matrix > 0).any() else 1.0
    dist_ratio   = within_dist / (between_dist + 1e-8)

    return {
        "silhouette":        sil,
        "knn_acc":           knn_acc,
        "within_dist":       within_dist,
        "between_dist":      between_dist,
        "dist_ratio":        dist_ratio,
        "class_dist_matrix": dist_matrix,
    }


# ── Additional plot functions ─────────────────────────────────────────────────

def plot_repetition_scatter(
    coords_2d: np.ndarray,
    rep_labels: np.ndarray,
    method_name: str,
    output_path: str = None,
    show: bool = True,
):
    """2-D scatter coloured by repetition level (1–4)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed.  pip install matplotlib")

    rep_names = ["single", "double", "triple", "quadruple"]
    colors    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax   = plt.subplots(figsize=(10, 8))
    for r, (name, color) in enumerate(zip(rep_names, colors), start=1):
        mask = rep_labels == r
        if mask.sum() == 0:
            continue
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=color, label=f"{r}× ({name})", alpha=0.75, s=40, edgecolors="none")
    ax.legend(title="Repetitions", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(f"PTv3 Latent Space — {method_name} (coloured by repetition)")
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_distance_matrix(
    dist_matrix: np.ndarray,
    class_names: list,
    output_path: str = None,
    show: bool = True,
):
    """Heatmap of mean pairwise L2 distances between the K letter classes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed.  pip install matplotlib")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dist_matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean L2 distance (L2-normalised embeddings)")
    ax.set_title("26×26 Mean Inter-class Embedding Distance\n(darker = closer, better separation = dark diagonal blocks)")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise PTv3 latent representations of SWISS point clouds"
    )
    parser.add_argument(
        "--checkpoint", default="best_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-dir", default="data_npy",
        help="Directory of .npz files from convert_3dm.py",
    )
    parser.add_argument(
        "--split", default="all", choices=["train", "val", "all"],
        help="Which split to embed (default: all)",
    )
    parser.add_argument(
        "--source", default="encoder", choices=["decoder", "encoder"],
        help="Which features to pool: 'encoder' bottleneck (default, 256D) or 'decoder' (32D)",
    )
    parser.add_argument(
        "--method", default="auto", choices=["auto", "umap", "tsne", "pca"],
        help="Dimensionality reduction method (default: auto → umap/tsne/pca)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for saving output figures (default: current dir)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() (useful for headless environments)",
    )
    parser.add_argument(
        "--no-metrics", action="store_true",
        help="Skip silhouette / k-NN computation (for quick visual checks)",
    )
    parser.add_argument(
        "--save-embeddings", default=None,
        help="Save raw (pre-reduction) embeddings to this .npz file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.data_dir}")
    ds = SwissDataset(args.data_dir, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=swiss_collate_fn,
        num_workers=0,
    )
    print(f"  {len(ds)} scenes  |  {ds.num_classes} classes")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model_kwargs = ckpt.get("model_kwargs", DEFAULT_MODEL_KWARGS)
    model = PointTransformerV3(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Epoch {ckpt.get('epoch', '?')}  |  "
          f"val_metric={ckpt.get('val_miou', float('nan')):.4f}")

    # ── Extract embeddings ────────────────────────────────────────────────────
    print(f"\nExtracting {args.source} embeddings for {len(ds)} scenes …")
    if args.source == "encoder":
        embeddings, labels, filenames = _extract_encoder(model, loader, device)
    else:
        embeddings, labels, filenames = _extract_decoder(model, loader, device)

    print(f"  Raw embedding shape: {embeddings.shape}")

    if args.save_embeddings:
        np.savez_compressed(
            args.save_embeddings,
            embeddings = embeddings,
            labels     = labels,
            filenames  = np.array(filenames),
        )
        print(f"  Embeddings saved → {args.save_embeddings}")

    # ── Quantitative metrics ──────────────────────────────────────────────────
    if not args.no_metrics:
        print("\nComputing metrics …")
        m = compute_metrics(embeddings, labels)
        print(
            f"\n── Embedding Quality Metrics ─────────────────────────\n"
            f"  Silhouette score    : {m['silhouette']:+.4f}  (higher = better, max 1)\n"
            f"  5-NN accuracy (CV)  : {m['knn_acc']*100:.1f} %\n"
            f"  Within-class dist   : {m['within_dist']:.4f}\n"
            f"  Between-class dist  : {m['between_dist']:.4f}\n"
            f"  Within/Between ratio: {m['dist_ratio']:.4f}  (lower = better)\n"
            f"─────────────────────────────────────────────────────"
        )

    # ── Dimensionality reduction ──────────────────────────────────────────────
    print(f"\nReducing to 2D with method='{args.method}' …")
    coords_2d, method_name = reduce_2d(embeddings, args.method, seed=args.seed)

    # ── Load repetition labels from .npz files ────────────────────────────────
    data_path = Path(args.data_dir)
    rep_labels = np.array([
        int(np.load(data_path / f"{fn}.npz")["repetitions"][0])
        for fn in filenames
    ])

    # ── Plots ─────────────────────────────────────────────────────────────────
    show = not args.no_show
    mn   = method_name.lower()

    print("\nPlotting …")
    plot_latent(
        coords_2d   = coords_2d,
        labels      = labels,
        filenames   = filenames,
        method_name = method_name,
        source      = args.source,
        class_names = ds.class_names,
        output_path = str(out_dir / f"latent_letter_{mn}.png"),
        show        = show,
    )
    plot_repetition_scatter(
        coords_2d   = coords_2d,
        rep_labels  = rep_labels,
        method_name = method_name,
        output_path = str(out_dir / f"latent_repetition_{mn}.png"),
        show        = show,
    )
    if not args.no_metrics:
        plot_distance_matrix(
            dist_matrix = m["class_dist_matrix"],
            class_names = ds.class_names,
            output_path = str(out_dir / "latent_distmatrix.png"),
            show        = show,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
