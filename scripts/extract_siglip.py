# EXTRACT SIGLIP global image embedding for COCO val2017
# semantic explanation, just one per image bc dino covers items

import argparse
import h5py
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipVisionModel

from src.paths import COCO_ANNOTATIONS, COCO_VAL_IMAGES, FEATURES_DIR

MODEL_NAME = "google/siglip-base-patch16-224"
DEVICE = "cuda"


def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipVisionModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def extract_features(image_id, coco, processor, model):
    img_info = coco.loadImgs(image_id)[0]
    image = Image.open(COCO_VAL_IMAGES / img_info["file_name"]).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    # pooler_output = global image embedding
    global_embed = outputs.pooler_output[0].cpu().numpy().astype(np.float16)  # (hidden_dim,)
    return global_embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="siglip_val2017.h5")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / args.output

    coco = COCO(str(COCO_ANNOTATIONS / "instances_val2017.json"))
    img_ids = coco.getImgIds()
    if args.limit is not None:
        img_ids = img_ids[:args.limit]

    print(f"Extracting SigLIP features for {len(img_ids)} images -> {out_path}")
    processor, model = load_model()

    with h5py.File(out_path, "w") as f:
        for image_id in tqdm(img_ids):
            embed = extract_features(image_id, coco, processor, model)
            f.create_dataset(str(image_id), data=embed)

    print(f"\nSaved {out_path}, {out_path.stat().st_size / 1024**2:.1f} MB")

    # Sanity check
    with h5py.File(out_path, "r") as f:
        keys = list(f.keys())
        print(f"Total image_ids: {len(keys)}")
        sample = f[keys[0]]
        print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")


if __name__ == "__main__":
    main()