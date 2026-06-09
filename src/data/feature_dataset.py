"""Unified candidate-menu loader for the three feature families.

For any image_id, returns a dict containing per-family feature arrays plus
metadata. Downstream models (selector, probes) handle padding and projection.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.paths import FEATURES_DIR
from src.data.splits import load_split

DINOV2_FILE = "dinov2_val2017.h5"
SIGLIP_FILE = "siglip_val2017.h5"
LOWLEVEL_FILE = "lowlevel_val2017.h5"

# Fixed candidate menu layout
MAX_SLOTS = 12  # max DINOv2 object slots
MENU_SIZE = 1 + 1 + MAX_SLOTS + 1  # siglip + dinov2_cls + slots + lowlevel = 15

# Family IDs
FAM_SIGLIP = 0
FAM_DINOV2_CLS = 1
FAM_OBJECT_SLOT = 2
FAM_LOWLEVEL = 3

# Raw feature dims per family
DIM_SIGLIP = 768
DIM_DINOV2 = 768
DIM_LOWLEVEL = 108

def collate_menus(batch: list[dict]) -> dict:
    """Collate variable-size menus into fixed-size padded tensors.

    Returns a dict of batched tensors:
        image_id:  (B,) long
        siglip:    (B, 768)
        dinov2_cls:(B, 768)
        slots:     (B, MAX_SLOTS, 768)   -- zero-padded
        lowlevel:  (B, 108)
        slot_valid:(B, MAX_SLOTS) bool   -- which object slots are real
        slot_cats: (B, MAX_SLOTS) long   -- COCO category IDs, -1 for padding
    """
    B = len(batch)
    image_ids = torch.tensor([m["image_id"] for m in batch], dtype=torch.long)
    siglip = torch.stack([m["siglip_global"] for m in batch])  # (B, 768)
    dinov2_cls = torch.stack([m["dinov2_cls"] for m in batch])  # (B, 768)
    lowlevel = torch.stack([m["lowlevel"] for m in batch])  # (B, 108)

    slots = torch.zeros(B, MAX_SLOTS, DIM_DINOV2)
    slot_valid = torch.zeros(B, MAX_SLOTS, dtype=torch.bool)
    slot_cats = torch.full((B, MAX_SLOTS), -1, dtype=torch.long)

    for i, m in enumerate(batch):
        n = min(m["dinov2_slots"].shape[0], MAX_SLOTS)
        if n > 0:
            slots[i, :n] = m["dinov2_slots"][:n]
            slot_valid[i, :n] = True
            slot_cats[i, :n] = m["obj_categories"][:n]

    return {
        "image_id": image_ids,
        "siglip": siglip,
        "dinov2_cls": dinov2_cls,
        "slots": slots,
        "lowlevel": lowlevel,
        "slot_valid": slot_valid,
        "slot_cats": slot_cats,
    }


class FeatureDataset(Dataset):
    """Dataset returning the per-image candidate menu across three feature families.

    Each item is a dict:
        image_id:      int
        siglip_global: tensor (768,) float32
        dinov2_cls:    tensor (768,) float32
        dinov2_slots:  tensor (N_obj, 768) float32   -- N_obj varies, 0 to 12
        obj_categories: tensor (N_obj,) long          -- COCO category IDs
        lowlevel:      tensor (108,) float32
    """

    def __init__(self, split: str = "train"):
        self.split = split
        self.image_ids = load_split(split)

        # Don't open h5 files here — open lazily per-worker in __getitem__.
        self._files = None

    def _open_files(self):
        """Lazily open h5 file handles. Safe across DataLoader workers."""
        if self._files is None:
            self._files = {
                "dinov2":   h5py.File(FEATURES_DIR / DINOV2_FILE, "r"),
                "siglip":   h5py.File(FEATURES_DIR / SIGLIP_FILE, "r"),
                "lowlevel": h5py.File(FEATURES_DIR / LOWLEVEL_FILE, "r"),
            }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
            return self.get_menu(self.image_ids[idx])

    def get_menu(self, image_id: int) -> dict:
        """Load the per-family candidate menu for one image."""
        self._open_files()
        key = str(image_id)

        # DINOv2: cls + per-object slots + category IDs
        dinov2_group = self._files["dinov2"][key]
        dinov2_cls = torch.from_numpy(dinov2_group["cls"][:]).float()           # (768,)
        dinov2_slots = torch.from_numpy(dinov2_group["obj_slots"][:]).float()   # (N_obj, 768)
        obj_categories = torch.from_numpy(dinov2_group["obj_categories"][:]).long()  # (N_obj,)

        # SigLIP: one global semantic embedding
        siglip_global = torch.from_numpy(self._files["siglip"][key][:]).float()  # (768,)

        # Low-level: HSV + Gabor summary
        lowlevel = torch.from_numpy(self._files["lowlevel"][key][:]).float()    # (108,)

        return {
            "image_id": image_id,
            "siglip_global": siglip_global,
            "dinov2_cls": dinov2_cls,
            "dinov2_slots": dinov2_slots,
            "obj_categories": obj_categories,
            "lowlevel": lowlevel,
        }

if __name__ == "__main__":
    from pycocotools.coco import COCO
    from src.paths import COCO_ANNOTATIONS

    ds = FeatureDataset(split="train")
    print(f"Train dataset: {len(ds)} items")

    # Sanity check on one menu
    menu = ds[0]
    print(f"\nMenu for image_id {menu['image_id']}:")
    print(f"  siglip_global:  {tuple(menu['siglip_global'].shape)}  {menu['siglip_global'].dtype}")
    print(f"  dinov2_cls:     {tuple(menu['dinov2_cls'].shape)}  {menu['dinov2_cls'].dtype}")
    print(f"  dinov2_slots:   {tuple(menu['dinov2_slots'].shape)}  {menu['dinov2_slots'].dtype}")
    print(f"  obj_categories: {tuple(menu['obj_categories'].shape)}  values: {menu['obj_categories'].tolist()}")
    print(f"  lowlevel:       {tuple(menu['lowlevel'].shape)}  {menu['lowlevel'].dtype}")

    # Sanity check: translate category IDs to names so we can eyeball it
    coco = COCO(str(COCO_ANNOTATIONS / "instances_val2017.json"))
    cat_ids = menu['obj_categories'].tolist()
    if cat_ids:
        names = [coco.loadCats(c)[0]['name'] for c in cat_ids]
        print(f"  Object names: {names}")

    # Loader test
    from torch.utils.data import DataLoader

    # loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
    # batch = next(iter(loader))
    # print(f"\nBatched menu (batch_size=1):")
    # print(f"  siglip_global: {tuple(batch['siglip_global'].shape)}")
    # print(f"  dinov2_slots:  {tuple(batch['dinov2_slots'].shape)}")
    # print("Loader works.")

    ds = FeatureDataset(split="train")
    loader = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False,
                        collate_fn=collate_menus)
    batch = next(iter(loader))
    print("Batched menu shapes (batch_size=8):")
    for k, v in batch.items():
        print(f"  {k:12s}: {tuple(v.shape)}  {v.dtype}")
    print(f"\nValid slots per image: {batch['slot_valid'].sum(dim=1).tolist()}")