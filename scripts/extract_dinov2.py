# Extract DINOv2 features for COCO val2017

import argparse
import h5py
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from src.paths import COCO_ANNOTATIONS, COCO_VAL_IMAGES, FEATURES_DIR

MAX_OBJECTS = 12
PATCH_GRID = 16  # 224 / 14
MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda"

def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model

@torch.no_grad()
def extract_features(image_id, coco, processor, model):
    img_info = coco.loadImgs(image_id)[0]
    image = Image.open(COCO_VAL_IMAGES / img_info["file_name"]).convert("RGB")
    H, W = img_info["height"], img_info["width"]

    # Forward pass
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]                  # (257, 768)
    cls = hidden[0].cpu().numpy().astype(np.float16)       # (768,) global gist
    patches = hidden[1:].cpu().numpy().astype(np.float16)  # (256, 768) spatial

    # Object slots: mean-pool patches inside each segmentation mask
    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    anns = [a for a in anns if not a.get("iscrowd", 0)]
    anns = sorted(anns, key=lambda a: a["area"], reverse=True)[:MAX_OBJECTS]

    obj_slots, obj_categories = [], []
    for ann in anns:
        mask = coco.annToMask(ann)                                          # (H, W) uint8
        mask_pil = Image.fromarray(mask * 255)
        mask_small = np.array(mask_pil.resize((PATCH_GRID, PATCH_GRID),
                                              Image.NEAREST)) > 0           # (16, 16) bool
        mask_flat = mask_small.flatten()                                     # (256,) bool

        if mask_flat.sum() == 0:
            # Object too small to land on any patch — fall back to bbox center
            x, y, w, h = ann["bbox"]
            cx = max(0, min(PATCH_GRID - 1, int((x + w/2) / W * PATCH_GRID)))
            cy = max(0, min(PATCH_GRID - 1, int((y + h/2) / H * PATCH_GRID)))
            mask_flat[cy * PATCH_GRID + cx] = True

        slot = patches[mask_flat].mean(axis=0).astype(np.float16)            # (768,)
        obj_slots.append(slot)
        obj_categories.append(ann["category_id"])

    obj_slots = np.stack(obj_slots) if obj_slots else np.zeros((0, 768), dtype=np.float16)
    obj_categories = np.array(obj_categories, dtype=np.int32)
    return cls, obj_slots, obj_categories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="dinov2_val2017.h5")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / args.output

    coco = COCO(str(COCO_ANNOTATIONS / "instances_val2017.json"))
    img_ids = coco.getImgIds()
    if args.limit is not None:
        img_ids = img_ids[:args.limit]

    print(f"Extracting features for {len(img_ids)} images -> {out_path}")
    processor, model = load_model()

    with h5py.File(out_path, "w") as f:
        for image_id in tqdm(img_ids):
            cls, obj_slots, obj_categories = extract_features(image_id, coco, processor, model)
            g = f.create_group(str(image_id))
            g.create_dataset("cls", data=cls)
            g.create_dataset("obj_slots", data=obj_slots)
            g.create_dataset("obj_categories", data=obj_categories)

    print(f"\nSaved {out_path}, {out_path.stat().st_size / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()