"""Evaluate trained continuous baselines on val and append to the results table.

For each (method, K) bundle: build the method's own recognition gallery, then walk
val with a reward function that scores z directly (same attr + recog combination as
the discrete RewardFunction, no selection mask). Each method is graded against its
own gallery; the task is space-invariant so the comparison stays fair.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.paths import CHECKPOINTS_DIR, FEATURES_DIR, REPO_ROOT
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.probes.question_bank import (
    QuestionBank, _build_category_index, question_to_template_idx,
)
from src.eval.continuous import (
    make_pool,
    build_compressor,
    ContinuousAttributeProbe, ContinuousRecognitionProbe,
)

DEVICE = "cuda"
RESULTS_DIR = REPO_ROOT / "results"


@torch.no_grad()
def build_continuous_gallery(compressor, recog_probe, ref_ids,
                             batch_size: int = 128) -> torch.Tensor:
    """(N_refs, RECOG_EMBED_DIM) — embed every reference image with this method."""
    fd = FeatureDataset(split="val")     # only used for get_menu; we iterate ref_ids directly
    compressor.eval(); recog_probe.eval()
    embeds = []
    for start in range(0, len(ref_ids), batch_size):
        chunk = [int(i) for i in ref_ids[start:start + batch_size]]
        menus = [fd.get_menu(i) for i in chunk]
        batch = collate_menus(menus)
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        z = compressor.encode(make_pool(b))
        embeds.append(recog_probe(z).cpu())
    return torch.cat(embeds, dim=0)


class ContinuousRewardFunction:
    """RewardFunction for continuous-z baselines: same scoring, probes read z directly
    (no selection_mask). Same question bank, hard neighbours, and lineup_size."""
    def __init__(self, compressor, attr_probe, recog_probe,
                 alpha: float = 0.5, beta: float = 0.5,
                 n_questions: int = 4, lineup_size: int = 50,
                 p_caption_relevant: float = 0.7,
                 distractor_mode: str = "semantic", seed: int = 231):
        self.compressor = compressor
        self.attr_probe = attr_probe
        self.recog_probe = recog_probe
        self.alpha = alpha
        self.beta = beta
        self.n_questions = n_questions
        self.lineup_size = lineup_size
        self.p_caption_relevant = p_caption_relevant
        self.distractor_mode = distractor_mode
        self.rng = np.random.default_rng(seed)

        self.qb = QuestionBank()
        self.cat_id_to_idx = _build_category_index(self.qb.coco)

        # same ref ordering as the discrete gallery, so distractor lookups line up
        refs = np.load(FEATURES_DIR / "recognition_refs.npz")
        self.ref_ids = refs["image_ids"]
        self.ref_id_to_idx = {int(iid): i for i, iid in enumerate(self.ref_ids)}

        print(f"  Building continuous recognition gallery ({len(self.ref_ids)} refs)...")
        self.ref_embeds = build_continuous_gallery(
            compressor, recog_probe, self.ref_ids).to(DEVICE)

        nbrs = np.load(FEATURES_DIR / "neighbors.npz")
        self.nbr_ids = nbrs["image_ids"]
        self.nbr_id_to_idx = {int(iid): i for i, iid in enumerate(self.nbr_ids)}
        self.semantic_nn = nbrs["semantic_nn"]
        self.perceptual_nn = nbrs["perceptual_nn"]

    @torch.no_grad()
    def _attribute_reward(self, batch, image_ids):
        B = len(image_ids)
        z = self.compressor.encode(make_pool(batch))
        correct_sum = torch.zeros(B, device=DEVICE)
        for _ in range(self.n_questions):
            template_idx, answers = [], []
            for img_id in image_ids:
                q = self.qb.sample_question(int(img_id),
                                            p_caption_relevant=self.p_caption_relevant,
                                            rng=self.rng)
                template_idx.append(question_to_template_idx(q, self.cat_id_to_idx))
                answers.append(q["answer"])
            q_idx = torch.tensor(template_idx, dtype=torch.long, device=DEVICE)
            ans = torch.tensor(answers, dtype=torch.float, device=DEVICE)
            logits = self.attr_probe(z, q_idx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_sum += (preds == ans).float()
        return correct_sum / self.n_questions

    @torch.no_grad()
    def _recognition_reward(self, batch, image_ids):
        z = self.compressor.encode(make_pool(batch))
        sel_embed = self.recog_probe(z)              # (B, 32)
        nn_table = self.semantic_nn if self.distractor_mode == "semantic" else self.perceptual_nn

        rewards = torch.zeros(len(image_ids), device=DEVICE)
        for b, img_id in enumerate(image_ids):
            img_id = int(img_id)
            true_ref_idx = self.ref_id_to_idx[img_id]

            nbr_pos = self.nbr_id_to_idx[img_id]
            neighbor_global_idxs = nn_table[nbr_pos]
            neighbor_ids = [int(self.nbr_ids[n]) for n in neighbor_global_idxs]
            distractor_ref_idxs = [self.ref_id_to_idx[nid] for nid in neighbor_ids
                                   if nid in self.ref_id_to_idx][:self.lineup_size - 1]

            lineup_idxs = [true_ref_idx] + distractor_ref_idxs
            perm = self.rng.permutation(len(lineup_idxs)).tolist()
            true_pos = perm.index(0)
            lineup_idxs = [lineup_idxs[p] for p in perm]
            lineup = self.ref_embeds[lineup_idxs]    # (L, 32)

            sims = lineup @ sel_embed[b]
            pred = sims.argmax().item()
            rewards[b] = 1.0 if pred == true_pos else 0.0

        return rewards

    @torch.no_grad()
    def compute_reward(self, batch, image_ids):
        attr = self._attribute_reward(batch, image_ids)
        recog = self._recognition_reward(batch, image_ids)
        return self.alpha * attr + self.beta * recog, attr, recog


def load_bundle(method: str, k: int):
    """Reconstruct (compressor, attr_probe, recog_probe) from a saved bundle."""
    ckpt = CHECKPOINTS_DIR / f"continuous_{method}_k{k}.pt"
    state = torch.load(ckpt, map_location=DEVICE)
    compressor = build_compressor(method, k).to(DEVICE)
    compressor.load_state_dict(state["compressor"])
    attr_probe = ContinuousAttributeProbe(z_dim=k).to(DEVICE)
    attr_probe.load_state_dict(state["attr_probe"])
    recog_probe = ContinuousRecognitionProbe(z_dim=k).to(DEVICE)
    recog_probe.load_state_dict(state["recog_probe"])
    compressor.eval(); attr_probe.eval(); recog_probe.eval()
    return compressor, attr_probe, recog_probe


@torch.no_grad()
def evaluate_method(method: str, k: int, val_loader, args) -> dict:
    print(f"\n[{method}] loading bundle ...")
    compressor, attr_probe, recog_probe = load_bundle(method, k)
    rf = ContinuousRewardFunction(compressor, attr_probe, recog_probe,
                                  distractor_mode=args.distractor_mode,
                                  seed=args.seed)
    tot_r, tot_a, tot_g, n = 0.0, 0.0, 0.0, 0
    for batch in val_loader:
        image_ids = batch["image_id"].tolist()
        b = {k_: v.to(DEVICE) for k_, v in batch.items()}
        reward, attr, recog = rf.compute_reward(b, image_ids)
        tot_r += reward.sum().item()
        tot_a += attr.sum().item()
        tot_g += recog.sum().item()
        n += len(image_ids)
    return {"reward": tot_r / n, "attr": tot_a / n, "recog": tot_g / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=231)
    ap.add_argument("--distractor-mode", default="semantic",
                    choices=["semantic", "perceptual"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--methods", nargs="+", default=["pca", "ae", "vib"])
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    val_ds = FeatureDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_menus)

    rows = {}
    print(f"Evaluating continuous baselines on val at K={args.k} "
          f"({args.distractor_mode} distractors)")
    for method in args.methods:
        ckpt = CHECKPOINTS_DIR / f"continuous_{method}_k{args.k}.pt"
        if not ckpt.exists():
            print(f"  (skipping {method}: no checkpoint at {ckpt})")
            continue
        out = evaluate_method(method, args.k, val_loader, args)
        rows[method] = out
        print(f"  {method:5s}: reward={out['reward']:.4f}  "
              f"attr={out['attr']:.4f}  recog={out['recog']:.4f}  "
              f"(n={out['n']})")

    out_path = RESULTS_DIR / f"continuous_k{args.k}_{args.distractor_mode}.json"
    payload = {
        "k": args.k,
        "distractor_mode": args.distractor_mode,
        "seed": args.seed,
        "split": "val",
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
