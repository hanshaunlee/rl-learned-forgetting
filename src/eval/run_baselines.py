"""Evaluate the discrete baselines + the trained GRPO selector on val at a fixed K.

    python -m src.eval.run_baselines --k 2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.paths import CHECKPOINTS_DIR, REPO_ROOT
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.agent.selector import Selector, sample_selection
from src.agent.reward import RewardFunction
from src.eval.baselines import (
    BaselineSelector,
    RandomSelector,
    GlobalsOnlySelector,
    SaliencySelector,
)

DEVICE = "cuda"
RESULTS_DIR = REPO_ROOT / "results"


@torch.no_grad()
def evaluate_baseline(selector: BaselineSelector, k: int,
                      reward_fn: RewardFunction, loader: DataLoader,
                      reward_seed: int) -> dict:
    # Reset the RNG before every method, otherwise the shared reward RNG drifts
    # and each method sees different questions / lineup shuffles.
    reward_fn.rng = np.random.default_rng(reward_seed)
    tot_r, tot_a, tot_g, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        image_ids = batch["image_id"].tolist()
        b = {k_: v.to(DEVICE) for k_, v in batch.items()}
        mask, _ = selector.select(b, k=k)
        reward, attr, recog = reward_fn.compute_reward(b, mask, image_ids)
        tot_r += reward.sum().item()
        tot_a += attr.sum().item()
        tot_g += recog.sum().item()
        n += len(image_ids)
    return {"reward": tot_r / n, "attr": tot_a / n, "recog": tot_g / n, "n": n}


@torch.no_grad()
def evaluate_grpo(selector: Selector, k: int,
                  reward_fn: RewardFunction, loader: DataLoader,
                  reward_seed: int) -> dict:
    """Greedy top-K decode from the trained selector."""
    reward_fn.rng = np.random.default_rng(reward_seed)
    selector.eval()
    tot_r, tot_a, tot_g, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        image_ids = batch["image_id"].tolist()
        b = {k_: v.to(DEVICE) for k_, v in batch.items()}
        scores, valid = selector(b)
        mask, _ = sample_selection(scores, valid, k=k, greedy=True)
        reward, attr, recog = reward_fn.compute_reward(b, mask, image_ids)
        tot_r += reward.sum().item()
        tot_a += attr.sum().item()
        tot_g += recog.sum().item()
        n += len(image_ids)
    return {"reward": tot_r / n, "attr": tot_a / n, "recog": tot_g / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2,
                    help="Memory budget (default 2 — matches trained GRPO checkpoint).")
    ap.add_argument("--seed", type=int, default=231)
    ap.add_argument("--distractor-mode", default="semantic",
                    choices=["semantic", "perceptual"])
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reward_fn = RewardFunction(distractor_mode=args.distractor_mode, seed=args.seed)

    val_ds = FeatureDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_menus)

    baselines: list[BaselineSelector] = [
        RandomSelector(seed=args.seed),
        GlobalsOnlySelector(),
        SaliencySelector(),
    ]

    rows = {}
    print(f"\nEvaluating baselines on val at K={args.k} "
          f"({args.distractor_mode} distractors)\n")
    for sel in baselines:
        out = evaluate_baseline(sel, k=args.k, reward_fn=reward_fn,
                                loader=val_loader, reward_seed=args.seed)
        rows[sel.name] = out
        print(f"  {sel.name:13s}: reward={out['reward']:.4f}  "
              f"attr={out['attr']:.4f}  recog={out['recog']:.4f}  "
              f"(n={out['n']})")

    # Trained GRPO checkpoint, if present
    ckpt = CHECKPOINTS_DIR / f"selector_k{args.k}.pt"
    if ckpt.exists():
        sel_grpo = Selector().to(DEVICE)
        sel_grpo.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        out = evaluate_grpo(sel_grpo, k=args.k, reward_fn=reward_fn,
                            loader=val_loader, reward_seed=args.seed)
        rows["grpo"] = out
        print(f"  {'grpo':13s}: reward={out['reward']:.4f}  "
              f"attr={out['attr']:.4f}  recog={out['recog']:.4f}  "
              f"(n={out['n']})")
    else:
        print(f"\n  (no GRPO checkpoint at {ckpt} — skipped)")

    out_path = RESULTS_DIR / f"baselines_k{args.k}_{args.distractor_mode}.json"
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
