"""Train/val/test splits over COCO val2017 image_ids.

Splits are computed deterministically from a fixed seed and saved to disk so
they remain stable across sessions. Probes, selectors, and baselines should
all read splits from here so nothing leaks across the boundary.
"""
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from src.paths import COCO_ANNOTATIONS, REPO_ROOT

SPLITS_DIR = REPO_ROOT / "data" / "splits"
SEED = 231
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10  # test gets the remaining 0.10


def make_splits():
    """Generate splits from COCO val2017 image_ids and save to data/splits/."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(COCO_ANNOTATIONS / "instances_val2017.json"))
    image_ids = sorted(coco.getImgIds())  # sort for deterministic ordering

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(image_ids).tolist()

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out_path = SPLITS_DIR / f"{name}_ids.json"
        with open(out_path, "w") as f:
            json.dump(ids, f)
        print(f"{name}: {len(ids)} image_ids -> {out_path}")


def load_split(name: str) -> list[int]:
    """Load a split's image_ids. name in {'train', 'val', 'test'}."""
    path = SPLITS_DIR / f"{name}_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found. Run make_splits() first.")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    make_splits()