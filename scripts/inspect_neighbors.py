import numpy as np
from PIL import Image
from src.paths import FEATURES_DIR, COCO_VAL_IMAGES

data = np.load(FEATURES_DIR / "neighbors.npz")
ids = data["image_ids"]

query_idx = 0  # first image
q_id = ids[query_idx]
sem_ids = ids[data["semantic_nn"][query_idx][:5]]
perc_ids = ids[data["perceptual_nn"][query_idx][:5]]

def show(image_id, label):
    path = COCO_VAL_IMAGES / f"{image_id:012d}.jpg"
    print(f"{label}: {path}")
    Image.open(path).show()

show(q_id, "QUERY")
for i, sid in enumerate(sem_ids):
    show(sid, f"semantic neighbor {i+1}")
for i, pid in enumerate(perc_ids):
    show(pid, f"perceptual neighbor {i+1}")