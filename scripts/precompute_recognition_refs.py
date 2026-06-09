"""Precompute full-menu reference embeddings for every image using the frozen
recognition probe. These are the fixed gallery the agent's selections get matched
against at reward time.

Output: features/recognition_refs.npz with image_ids and their 32-d embeddings.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.paths import CHECKPOINTS_DIR, FEATURES_DIR
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.probes.recognition_probe import RecognitionProbe

DEVICE = "cuda"


def main():
    # Load frozen recognition probe
    probe = RecognitionProbe().to(DEVICE)
    probe.load_state_dict(torch.load(CHECKPOINTS_DIR / "recognition_probe.pt"))
    probe.eval()

    # We want references for ALL images (distractors can come from any split).
    # Embed train, val, and test full menus.
    all_ids, all_embeds = [], []
    for split in ["train", "val", "test"]:
        ds = FeatureDataset(split=split)
        loader = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=collate_menus)
        with torch.no_grad():
            for batch in loader:
                ids = batch["image_id"].tolist()
                b = {k: v.to(DEVICE) for k, v in batch.items()}
                # Full menu reference: sample_k=False uses all valid candidates
                emb = probe.encode(b, sample_k=False).cpu().numpy()  # (B, 32)
                all_ids.extend(ids)
                all_embeds.append(emb)

    image_ids = np.array(all_ids)
    embeds = np.concatenate(all_embeds, axis=0).astype(np.float32)   # (N, 32)

    out_path = FEATURES_DIR / "recognition_refs.npz"
    np.savez(out_path, image_ids=image_ids, embeds=embeds)
    print(f"Saved {out_path}")
    print(f"  image_ids: {image_ids.shape}")
    print(f"  embeds:    {embeds.shape}")

    # Sanity check: embeddings are L2-normalized (probe normalizes output)
    norms = np.linalg.norm(embeds[:5], axis=1)
    print(f"  Sample L2 norms (should be ~1.0): {norms.tolist()}")


if __name__ == "__main__":
    main()