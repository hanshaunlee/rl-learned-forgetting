from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
COCO_DIR = DATA_DIR / "coco"
COCO_VAL_IMAGES = COCO_DIR / "val2017"
COCO_ANNOTATIONS = COCO_DIR / "annotations"
FEATURES_DIR = REPO_ROOT / "features"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
