"""
convert_3dm.py
==============
Convert Rhino .3dm files → compressed numpy point clouds (.npz).

For each file the script:
  1. Loads all geometry objects via rhino3dm
  2. Collects Mesh objects directly; tessellates Brep objects into meshes
  3. Samples `--num-points` points uniformly (area-weighted) from triangle surfaces
  4. Computes per-point face normals
  5. Normalises the point cloud to unit sphere centred at the origin
  6. Saves {stem}.npz with arrays:
       coord       float32 [N, 3]  – normalised XYZ
       normal      float32 [N, 3]  – surface normals
       letter      int64   [1]     – class label 0-25 (A=0, B=1, …, Z=25)
       repetitions int64   [1]     – number of letter repetitions (1-4)
       num         int64   [1]     – variant number within the group

Filename convention expected by the parser:
    <LETTER repeated R times><variant number>
    e.g.  A1, AA1, AAA1, AAAA1  →  letter=A, reps=1..4, num=1

Requirements:
    pip install rhino3dm numpy

Usage:
    python convert_3dm.py
    python convert_3dm.py --input "SWISS Competition Forms" --output data_npy
    python convert_3dm.py --num-points 4096 --overwrite
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import rhino3dm
except ImportError:
    sys.exit(
        "rhino3dm is required for .3dm conversion.\n"
        "Install it with:  pip install rhino3dm"
    )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_stem(stem: str):
    """
    Extract (base_letter, repetitions, variant_number) from a stem like 'AAAA3'.

    Returns
    -------
    base   : str  single uppercase letter, e.g. 'A'
    reps   : int  number of times the letter appears, e.g. 4
    num    : int  trailing variant number, e.g. 3
    """
    m = re.match(r'^([A-Z])(\1*)(\d+)$', stem.upper())
    if m:
        base  = m.group(1)
        extra = m.group(2)          # '' for reps=1, 'AAA' for reps=4
        reps  = 1 + len(extra)
        num   = int(m.group(3))
        return base, reps, num
    # Fallback: take first char as letter, rest as number
    for i, ch in enumerate(stem):
        if ch.isdigit():
            letter = stem[:i].upper()
            letter = letter[0] if letter else 'A'
            reps   = len(stem[:i]) if stem[:i] else 1
            num    = int(stem[i:]) if stem[i:].isdigit() else 0
            return letter, reps, num
    return stem[0].upper(), 1, 0


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

def _rhino_mesh_to_arrays(mesh):
    """
    Convert a rhino3dm.Mesh to (verts float64[V,3], normals float32[V,3], faces int64[F,3]).

    rhino3dm 8.x exposes Vertices / Faces / Normals as iterables, not indexable lists,
    so we use list() to collect them.  Face tuples are (i0,i1,i2,i3); a quad has
    i2≠i3 and is split into two triangles.
    """
    verts_raw   = list(mesh.Vertices)   # Point3f objects with .X .Y .Z
    faces_raw   = list(mesh.Faces)      # tuples (i0, i1, i2, i3)
    normals_raw = list(mesh.Normals)    # Vector3f objects (may be empty)

    if not verts_raw or not faces_raw:
        return None, None, None

    verts = np.array([[v.X, v.Y, v.Z] for v in verts_raw], dtype=np.float64)

    if len(normals_raw) == len(verts_raw):
        normals = np.array([[n.X, n.Y, n.Z] for n in normals_raw], dtype=np.float32)
    else:
        normals = None   # will be computed from face normals after sampling

    face_list = []
    for fc in faces_raw:
        i0, i1, i2, i3 = fc
        face_list.append([i0, i1, i2])
        if i2 != i3:            # quad → two triangles
            face_list.append([i0, i2, i3])

    if not face_list:
        return None, None, None

    faces = np.array(face_list, dtype=np.int64)
    return verts, normals, faces


def _collect_meshes_from_file(path: str):
    """
    Return (vertices, per_vertex_normals, faces) numpy arrays.

    Dispatch:
      rhino3dm.Mesh  → converted directly
      rhino3dm.Brep  → each BrepFace carries a cached render mesh accessed via
                       face.GetMesh(MeshType.Any); faces are iterated from geo.Faces
      rhino3dm.Extrusion → converted to Brep first, then same path
    """
    model = rhino3dm.File3dm.Read(path)
    if model is None:
        raise RuntimeError(f"rhino3dm could not open: {path}")

    all_verts   = []
    all_faces   = []
    vert_offset = 0

    for obj in model.Objects:
        geo = obj.Geometry
        sub_meshes = []   # list of rhino3dm.Mesh

        if isinstance(geo, rhino3dm.Mesh):
            sub_meshes = [geo]

        elif isinstance(geo, rhino3dm.Brep):
            # BrepFace.GetMesh() returns the cached display/render mesh for each face
            for brep_face in geo.Faces:
                try:
                    m = brep_face.GetMesh(rhino3dm.MeshType.Any)
                    if m is not None:
                        sub_meshes.append(m)
                except Exception:
                    pass

        elif isinstance(geo, rhino3dm.Extrusion):
            # Convert Extrusion → Brep, then process faces
            try:
                brep = geo.ToBrep(False)
                if brep is not None:
                    for brep_face in brep.Faces:
                        m = brep_face.GetMesh(rhino3dm.MeshType.Any)
                        if m is not None:
                            sub_meshes.append(m)
            except Exception:
                pass

        for mesh in sub_meshes:
            verts, _, faces = _rhino_mesh_to_arrays(mesh)
            if verts is None:
                continue

            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += len(verts)

    if not all_verts:
        return None, None

    return (
        np.concatenate(all_verts, axis=0).astype(np.float64),
        np.concatenate(all_faces, axis=0).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def _sample_surface(verts, faces, n_points, rng):
    """
    Area-weighted uniform sampling on a triangle mesh.

    Returns
    -------
    pts     float32 [N, 3]
    normals float32 [N, 3]
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    edge1  = v1 - v0
    edge2  = v2 - v0
    cross  = np.cross(edge1, edge2)                       # [F, 3]
    areas  = 0.5 * np.linalg.norm(cross, axis=1)         # [F]
    total  = areas.sum()

    if total < 1e-12:
        # Degenerate mesh – fall back to random vertex sampling
        idx     = rng.choice(len(verts), size=n_points, replace=len(verts) < n_points)
        normals = np.zeros((n_points, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        return verts[idx].astype(np.float32), normals

    probs   = areas / total
    tri_idx = rng.choice(len(faces), size=n_points, p=probs)

    # Uniform barycentric coordinates
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    flip = r1 + r2 > 1.0
    r1[flip] = 1.0 - r1[flip]
    r2[flip] = 1.0 - r2[flip]
    r3 = 1.0 - r1 - r2

    pts = (
        r1[:, None] * v0[tri_idx]
        + r2[:, None] * v1[tri_idx]
        + r3[:, None] * v2[tri_idx]
    ).astype(np.float32)

    # Face normals (unit-length)
    fn  = cross[tri_idx].astype(np.float32)
    mag = np.linalg.norm(fn, axis=1, keepdims=True)
    mag = np.where(mag < 1e-12, 1.0, mag)
    normals = (fn / mag).astype(np.float32)

    return pts, normals


def _normalise(pts):
    """Centre at origin and scale to unit sphere."""
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-12:
        pts = pts / scale
    return pts


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_file(src: Path, dst: Path, n_points: int, rng, overwrite: bool) -> bool:
    """Convert a single .3dm file. Returns True on success."""
    if dst.exists() and not overwrite:
        print(f"  [skip] {src.name}  (already exists, use --overwrite)")
        return True

    try:
        verts, faces = _collect_meshes_from_file(str(src))
    except Exception as exc:
        print(f"  [err]  {src.name}: failed to load — {exc}")
        return False

    if verts is None or len(faces) == 0:
        print(f"  [warn] {src.name}: no usable mesh geometry found")
        return False

    pts, normals = _sample_surface(verts, faces, n_points, rng)
    pts = _normalise(pts)

    base, reps, num = parse_stem(src.stem)
    label = ord(base) - ord('A')   # 0-25

    np.savez_compressed(
        dst,
        coord       = pts,                         # [N, 3] float32
        normal      = normals,                     # [N, 3] float32
        letter      = np.array([label],  dtype=np.int64),
        repetitions = np.array([reps],   dtype=np.int64),
        num         = np.array([num],    dtype=np.int64),
    )
    print(
        f"  [ok]   {src.name:20s}  "
        f"letter={base}({label})  reps={reps}  "
        f"pts={n_points}  faces={len(faces)}"
    )
    return True


def convert_all(input_dir: str, output_dir: str, n_points: int, overwrite: bool):
    src_dir = Path(input_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.3dm"))
    if not files:
        print(f"No .3dm files found in: {src_dir}")
        return

    rng  = np.random.default_rng(42)
    ok   = 0
    fail = []

    print(f"\nConverting {len(files)} .3dm files  →  {dst_dir}/")
    print(f"Points per cloud: {n_points}\n")

    for src in files:
        dst = dst_dir / (src.stem + ".npz")
        if convert_file(src, dst, n_points, rng, overwrite):
            ok += 1
        else:
            fail.append(src.name)

    print(f"\n{'─'*50}")
    print(f"Done:  {ok}/{len(files)} converted")
    if fail:
        print(f"Failed ({len(fail)}):")
        for f in fail:
            print(f"  {f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .3dm files to numpy point cloud archives (.npz)"
    )
    parser.add_argument(
        "--input",  default="SWISS Competition Forms",
        help="Directory containing .3dm files (default: 'SWISS Competition Forms')",
    )
    parser.add_argument(
        "--output", default="data_npy",
        help="Output directory for .npz files (default: data_npy)",
    )
    parser.add_argument(
        "--num-points", type=int, default=2048,
        help="Number of points to sample per model (default: 2048)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-convert files that already have a corresponding .npz",
    )
    args = parser.parse_args()

    convert_all(args.input, args.output, args.num_points, args.overwrite)
