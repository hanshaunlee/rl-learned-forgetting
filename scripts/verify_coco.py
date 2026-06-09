from pycocotools.coco import COCO
from PIL import Image
from src.paths import COCO_ANNOTATIONS, COCO_VAL_IMAGES

ann_path = COCO_ANNOTATIONS / "instances_val2017.json"
coco = COCO(str(ann_path))
img_ids = coco.getImgIds()
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
print(f"Images: {len(img_ids)}")
print(f"Categories: {len(cat_ids)}")
print(f"Sample category names: {[c['name'] for c in cats[:10]]}")

sample_id = img_ids[0]
img_info = coco.loadImgs(sample_id)[0]
ann_ids = coco.getAnnIds(imgIds=sample_id)
anns = coco.loadAnns(ann_ids)
print(f"Image {sample_id}: {img_info['file_name']}, {img_info['width']}x{img_info['height']}")
print(f"  Annotations: {len(anns)}")
if anns:
    cat_name = coco.loadCats(anns[0]['category_id'])[0]['name']
    has_mask = 'segmentation' in anns[0]
    print(f"  First object: category={cat_name}, has_mask={has_mask}")

img_path = COCO_VAL_IMAGES / img_info['file_name']
im = Image.open(img_path)
print(f"  Loaded image: {im.size}, {im.mode}")
