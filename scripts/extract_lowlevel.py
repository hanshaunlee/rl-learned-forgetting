# EXTRACT LOW LEVEL FEATURES FOR COCO VAL2017
# 
# HSV color histogram
# Gabor filter responses (4 orientations x 3 frequencies = 12 features)
import argparse
import h5py
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import gabor
from tqdm import tqdm

from src.paths import COCO_ANNOTATIONS, COCO_VAL_IMAGES, FEATURES_DIR

TARGET_SIZE = 128  # working resolution for color and texture
HSV_BINS = 32
GABOR_ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 orientations
GABOR_FREQUENCIES = [0.1, 0.25, 0.4]  # 3 spatial frequencies


# HSV histogram, 32 bins per channel, normalized
# Returns (96,) float32
def hsv_histogram(rgb):
    hsv = rgb2hsv(rgb)
    hist = []
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=HSV_BINS, range=(0, 1))
        h = h / (h.sum() + 1e-9)  # norm to sum to 1 per channel
        hist.append(h)

    return np.concatenate(hist).astype(np.float32)  # (96,)

# Gabor filter responses: mean magnitude across 4 orientations x 3 frequencies
# Basically gets edges/texture at diff scales and orientations
def gabor_stats(rgb):
    gray = rgb2gray(rgb)
    stats = []

    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            real, imag = gabor(gray, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)
            stats.append(magnitude.mean())

    return np.array(stats, dtype=np.float32)  # (12,)


def extract_features(image_id, coco):
    img_info = coco.loadImgs(image_id)[0]
    image = Image.open(COCO_VAL_IMAGES / img_info["file_name"]).convert("RGB")
    image = image.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    rgb = np.array(image) / 255.0  # (128, 128, 3) float64
    color = hsv_histogram(rgb)  # (96,)
    texture = gabor_stats(rgb)  # (12,)
    return np.concatenate([color, texture]).astype(np.float16)  # (108,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="lowlevel_val2017.h5")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / args.output

    coco = COCO(str(COCO_ANNOTATIONS / "instances_val2017.json"))
    img_ids = coco.getImgIds()
    if args.limit is not None:
        img_ids = img_ids[:args.limit]

    print(f"Extracting low-level features for {len(img_ids)} images -> {out_path}")

    with h5py.File(out_path, "w") as f:
        for image_id in tqdm(img_ids):
            features = extract_features(image_id, coco)
            f.create_dataset(str(image_id), data=features)

    print(f"\nSaved {out_path}, {out_path.stat().st_size / 1024**2:.1f} MB")

    with h5py.File(out_path, "r") as f:
        keys = list(f.keys())
        print(f"Total image_ids: {len(keys)}")
        sample = f[keys[0]]
        print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")


if __name__ == "__main__":
    main()