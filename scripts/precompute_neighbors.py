"""Precompute semantic and perceptual nearest-neighbor lists for hard distractors.

For each image, find the most SigLIP-similar (semantic) and most DINOv2-CLS-similar
(perceptual) other images. Used to build hard distractor lineups for the recognition
probe and for agent reward computation.

Output: features/neighbors.npz with arrays indexed by a fixed image_id ordering.
"""
import h5py
import numpy as np
from tqdm import tqdm

from src.paths import FEATURES_DIR

SIGLIP_FILE = "siglip_val2017.h5"
DINOV2_FILE = "dinov2_val2017.h5"
TOP_K = 50  # number of neighbors to store per image per modality


def load_global_embeddings():
    """Load SigLIP global + DINOv2 CLS for all images. Returns (image_ids, siglip, dinov2)."""
    with h5py.File(FEATURES_DIR / SIGLIP_FILE, "r") as f:
        image_ids = sorted(int(k) for k in f.keys())
        siglip = np.stack([f[str(i)][:] for i in image_ids]).astype(np.float32)  # (N, 768)

    with h5py.File(FEATURES_DIR / DINOV2_FILE, "r") as f:
        dinov2 = np.stack([f[str(i)]["cls"][:] for i in image_ids]).astype(np.float32)  # (N, 768)

    return np.array(image_ids), siglip, dinov2


def cosine_neighbors(embeddings, top_k):
    """For each row, return indices of the top_k most cosine-similar OTHER rows."""
    # Normalize so dot product = cosine similarity
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sim = normed @ normed.T                      # (N, N) cosine similarity matrix
    np.fill_diagonal(sim, -np.inf)               # exclude self
    # argsort descending, take top_k
    neighbors = np.argsort(-sim, axis=1)[:, :top_k]  # (N, top_k) — indices into image_ids
    return neighbors


def main():
    print("Loading global embeddings...")
    image_ids, siglip, dinov2 = load_global_embeddings()
    print(f"Loaded {len(image_ids)} images")

    print("Computing semantic (SigLIP) neighbors...")
    semantic_nn = cosine_neighbors(siglip, TOP_K)        # (N, TOP_K) indices

    print("Computing perceptual (DINOv2) neighbors...")
    perceptual_nn = cosine_neighbors(dinov2, TOP_K)      # (N, TOP_K) indices

    out_path = FEATURES_DIR / "neighbors.npz"
    np.savez(
        out_path,
        image_ids=image_ids,            # (N,) — the canonical ordering
        semantic_nn=semantic_nn,        # (N, TOP_K) — indices into image_ids
        perceptual_nn=perceptual_nn,    # (N, TOP_K) — indices into image_ids
    )
    print(f"\nSaved {out_path}")

    # Sanity check
    data = np.load(out_path)
    print(f"image_ids: {data['image_ids'].shape}")
    print(f"semantic_nn: {data['semantic_nn'].shape}")
    print(f"perceptual_nn: {data['perceptual_nn'].shape}")

    # Show neighbors for the first image, translated to image_ids
    ids = data["image_ids"]
    q = ids[0]
    sem = ids[data["semantic_nn"][0][:5]]
    perc = ids[data["perceptual_nn"][0][:5]]
    print(f"\nImage {q}:")
    print(f"  top-5 semantic neighbors:   {sem.tolist()}")
    print(f"  top-5 perceptual neighbors: {perc.tolist()}")


if __name__ == "__main__":
    main()