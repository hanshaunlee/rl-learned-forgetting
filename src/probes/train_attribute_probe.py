"""Pretrain the attribute QA probe on full menus (with slot dropout), then freeze + save.

The probe learns to answer yes/no questions from a candidate menu. Caption-weighted
question sampling biases training toward human-described content.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.paths import CHECKPOINTS_DIR
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.probes.question_bank import (
    QuestionBank, _build_category_index, question_to_template_idx,
)
from src.probes.attribute_probe import AttributeProbe

DEVICE = "cuda"
BATCH_SIZE = 64
N_EPOCHS = 8
LR = 3e-4
P_CAPTION_RELEVANT = 0.7
SEED = 231


def sample_questions_for_batch(image_ids, qb, cat_id_to_idx, p_caption, rng):
    """For each image in the batch, sample one question. Returns (template_idx, answer) tensors."""
    template_idx, answers = [], []
    for img_id in image_ids.tolist():
        q = qb.sample_question(img_id, p_caption_relevant=p_caption, rng=rng)
        template_idx.append(question_to_template_idx(q, cat_id_to_idx))
        answers.append(q["answer"])
    return (torch.tensor(template_idx, dtype=torch.long),
            torch.tensor(answers, dtype=torch.float))


@torch.no_grad()
def evaluate(probe, loader, qb, cat_id_to_idx, rng):
    """Validation accuracy, broken down by answer class to expose class-imbalance gaming."""
    probe.eval()
    correct, total = 0, 0
    yes_correct, yes_total = 0, 0
    no_correct, no_total = 0, 0
    for batch in loader:
        q_idx, answers = sample_questions_for_batch(batch["image_id"], qb, cat_id_to_idx,
                                                    P_CAPTION_RELEVANT, rng)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        q_idx, answers = q_idx.to(DEVICE), answers.to(DEVICE)
        logits = probe(batch, q_idx, sample_k=False)
        preds = (torch.sigmoid(logits) > 0.5).float()

        correct += (preds == answers).sum().item()
        total += answers.numel()

        yes_mask = answers == 1
        no_mask = answers == 0
        yes_correct += (preds[yes_mask] == answers[yes_mask]).sum().item()
        yes_total += yes_mask.sum().item()
        no_correct += (preds[no_mask] == answers[no_mask]).sum().item()
        no_total += no_mask.sum().item()

    acc = correct / total
    yes_acc = yes_correct / max(yes_total, 1)
    no_acc = no_correct / max(no_total, 1)
    balanced = (yes_acc + no_acc) / 2
    return acc, yes_acc, no_acc, balanced, yes_total, no_total


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    qb = QuestionBank()
    cat_id_to_idx = _build_category_index(qb.coco)

    train_ds = FeatureDataset(split="train")
    val_ds = FeatureDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_menus)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_menus)

    probe = AttributeProbe().to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Training attribute probe: {len(train_ds)} train, {len(val_ds)} val")
    best_val = 0.0
    for epoch in range(N_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
            q_idx, answers = sample_questions_for_batch(batch["image_id"], qb, cat_id_to_idx,
                                                        P_CAPTION_RELEVANT, rng)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            q_idx, answers = q_idx.to(DEVICE), answers.to(DEVICE)

            logits = probe(batch, q_idx, sample_k=True)
            loss = loss_fn(logits, answers)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        acc, yes_acc, no_acc, balanced, n_yes, n_no = evaluate(probe, val_loader, qb, cat_id_to_idx, rng)
        print(f"Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.4f}  "
              f"acc={acc:.4f}  yes_acc={yes_acc:.4f}  no_acc={no_acc:.4f}  "
              f"balanced={balanced:.4f}  (yes={n_yes}, no={n_no})")

        if balanced > best_val:   # select on balanced accuracy, not raw
            best_val = balanced
            torch.save(probe.state_dict(), CHECKPOINTS_DIR / "attribute_probe.pt")

    print(f"\nBest val accuracy: {best_val:.4f}")
    print(f"Saved to {CHECKPOINTS_DIR / 'attribute_probe.pt'}")


if __name__ == "__main__":
    main()