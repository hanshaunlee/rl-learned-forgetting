"""Frozen-probe reward for a (menu, selection-mask) pair: alpha*attribute + beta*recognition."""
import numpy as np
import torch

from src.paths import CHECKPOINTS_DIR, FEATURES_DIR
from src.data.feature_dataset import MENU_SIZE
from src.probes.attribute_probe import AttributeProbe
from src.probes.recognition_probe import RecognitionProbe
from src.probes.question_bank import (
    QuestionBank, _build_category_index, question_to_template_idx,
)

DEVICE = "cuda"


class RewardFunction:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 n_questions: int = 4, lineup_size: int = 50,
                 p_caption_relevant: float = 0.7,
                 distractor_mode: str = "semantic", seed: int = 231):
        self.alpha = alpha
        self.beta = beta
        self.n_questions = n_questions
        self.lineup_size = lineup_size
        self.p_caption_relevant = p_caption_relevant
        self.distractor_mode = distractor_mode    # "semantic" or "perceptual"
        self.rng = np.random.default_rng(seed)

        self.attr_probe = AttributeProbe().to(DEVICE)
        self.attr_probe.load_state_dict(torch.load(CHECKPOINTS_DIR / "attribute_probe.pt"))
        self.attr_probe.eval()

        self.recog_probe = RecognitionProbe().to(DEVICE)
        self.recog_probe.load_state_dict(torch.load(CHECKPOINTS_DIR / "recognition_probe.pt"))
        self.recog_probe.eval()

        self.qb = QuestionBank()
        self.cat_id_to_idx = _build_category_index(self.qb.coco)

        # Full-menu reference embeddings for the recognition lineup
        refs = np.load(FEATURES_DIR / "recognition_refs.npz")
        self.ref_ids = refs["image_ids"]
        self.ref_embeds = torch.from_numpy(refs["embeds"]).float().to(DEVICE)
        self.ref_id_to_idx = {int(iid): i for i, iid in enumerate(self.ref_ids)}

        nbrs = np.load(FEATURES_DIR / "neighbors.npz")
        self.nbr_ids = nbrs["image_ids"]
        self.nbr_id_to_idx = {int(iid): i for i, iid in enumerate(self.nbr_ids)}
        self.semantic_nn = nbrs["semantic_nn"]
        self.perceptual_nn = nbrs["perceptual_nn"]

    @torch.no_grad()
    def _attribute_reward(self, batch, selection_mask, image_ids):
        # mean correctness over n_questions caption-weighted questions
        B = len(image_ids)
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

            logits = self.attr_probe(batch, q_idx, selection_mask=selection_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_sum += (preds == ans).float()

        return correct_sum / self.n_questions   # (B,) in [0,1]

    @torch.no_grad()
    def _recognition_reward(self, batch, selection_mask, image_ids):
        # top-1 retrieval of the selected-subset embedding against a hard lineup
        sel_embed = self.recog_probe.encode(batch, selection_mask=selection_mask)
        nn_table = self.semantic_nn if self.distractor_mode == "semantic" else self.perceptual_nn

        rewards = torch.zeros(len(image_ids), device=DEVICE)
        for b, img_id in enumerate(image_ids):
            img_id = int(img_id)
            true_ref_idx = self.ref_id_to_idx[img_id]

            nbr_pos = self.nbr_id_to_idx[img_id]
            neighbor_ids = [int(self.nbr_ids[n]) for n in nn_table[nbr_pos]]
            distractor_ref_idxs = [self.ref_id_to_idx[nid] for nid in neighbor_ids
                                   if nid in self.ref_id_to_idx][:self.lineup_size - 1]

            lineup_idxs = [true_ref_idx] + distractor_ref_idxs
            # Shuffle so the true image isn't always at index 0; argmax's tie-breaking
            # on a uniform sims vector would otherwise hand a free win to degenerate
            # selections (e.g. empty -> zero embedding -> all-zero sims -> argmax 0).
            perm = self.rng.permutation(len(lineup_idxs)).tolist()
            true_pos = perm.index(0)
            lineup_idxs = [lineup_idxs[p] for p in perm]
            lineup = self.ref_embeds[lineup_idxs]

            sims = lineup @ sel_embed[b]
            pred = sims.argmax().item()
            rewards[b] = 1.0 if pred == true_pos else 0.0

        return rewards

    @torch.no_grad()
    def compute_reward(self, batch, selection_mask, image_ids):
        attr = self._attribute_reward(batch, selection_mask, image_ids)
        recog = self._recognition_reward(batch, selection_mask, image_ids)
        return self.alpha * attr + self.beta * recog, attr, recog
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.feature_dataset import FeatureDataset, collate_menus, MAX_SLOTS

    K = 4
    rf = RewardFunction(distractor_mode="semantic")

    ds = FeatureDataset(split="val")
    loader = DataLoader(ds, batch_size=64, num_workers=0, collate_fn=collate_menus)
    batch = next(iter(loader))
    image_ids = batch["image_id"].tolist()
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    B = len(image_ids)

    def make_mask(strategy):
        mask = torch.zeros(B, MENU_SIZE, dtype=torch.bool, device=DEVICE)
        base_valid = torch.ones(B, MENU_SIZE, dtype=torch.bool, device=DEVICE)
        base_valid[:, 2:2 + MAX_SLOTS] = batch["slot_valid"]
        for i in range(B):
            valid_idx = base_valid[i].nonzero(as_tuple=True)[0].tolist()
            if strategy == "full":
                chosen = valid_idx                       # upper bound: keep everything
            elif strategy == "random_k":
                chosen = list(rf.rng.choice(valid_idx, size=min(K, len(valid_idx)), replace=False))
            elif strategy == "globals_only":
                chosen = [c for c in valid_idx if c in (0, 1)][:K]
            elif strategy == "empty":
                chosen = []
            mask[i, chosen] = True
        return mask

    print(f"Reward discrimination test (B={B}, K={K}, semantic distractors):\n")
    for strategy in ["full", "random_k", "globals_only", "empty"]:
        mask = make_mask(strategy)
        reward, attr, recog = rf.compute_reward(batch, mask, image_ids)
        print(f"  {strategy:14s}: reward={reward.mean():.3f}  "
              f"attr={attr.mean():.3f}  recog={recog.mean():.3f}")