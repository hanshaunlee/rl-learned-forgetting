"""GRPO over the selection policy. Sample G selections per image, score each with
the frozen probes, normalize the rewards within the group, and step on the log-probs."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.paths import CHECKPOINTS_DIR
from src.data.feature_dataset import FeatureDataset, collate_menus
from src.agent.selector import Selector, sample_selection
from src.agent.reward import RewardFunction

DEVICE = "cuda"
K = 2
G = 8                    # group size
BATCH_SIZE = 32
N_EPOCHS = 10
LR = 3e-4
ENTROPY_COEF = 0.01
EPS = 1e-6
SEED = 231
RANDOM_BASELINE = 0.855  # measured, random K=2


def replicate_batch(batch, g):
    out = {}
    for key, val in batch.items():
        out[key] = val.repeat_interleave(g, dim=0) if torch.is_tensor(val) else val
    return out


def first_step_entropy(scores, valid):
    probs = F.softmax(scores, dim=1)
    logp = torch.log(probs + 1e-12)
    ent = -(probs * logp).masked_fill(~valid, 0.0).sum(dim=1)
    return ent.mean()


@torch.no_grad()
def evaluate_greedy(selector, reward_fn, val_loader):
    selector.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        image_ids = batch["image_id"].tolist()
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        scores, valid = selector(b)
        mask, _ = sample_selection(scores, valid, k=K, greedy=True)
        reward, _, _ = reward_fn.compute_reward(b, mask, image_ids)
        total += reward.sum().item()
        n += len(image_ids)
    selector.train()
    return total / n


def main():
    torch.manual_seed(SEED)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = FeatureDataset(split="train")
    val_ds = FeatureDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_menus)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=0, collate_fn=collate_menus)

    selector = Selector().to(DEVICE)
    opt = torch.optim.Adam(selector.parameters(), lr=LR)
    reward_fn = RewardFunction(distractor_mode="semantic")

    print(f"GRPO training: K={K}, G={G}, random baseline={RANDOM_BASELINE:.3f}")
    best_val = 0.0

    for epoch in range(N_EPOCHS):
        selector.train()
        ep_reward, ep_ent, n_steps = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
            B = batch["image_id"].shape[0]
            image_ids = batch["image_id"].tolist()
            b = {k: v.to(DEVICE) for k, v in batch.items()}

            bg = replicate_batch(b, G)
            bg_image_ids = [iid for iid in image_ids for _ in range(G)]  # order has to match repeat_interleave

            scores, valid = selector(bg)
            mask, log_prob = sample_selection(scores, valid, k=K, greedy=False)
            reward, _, _ = reward_fn.compute_reward(bg, mask, bg_image_ids)

            R = reward.view(B, G)
            adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + EPS)).view(B * G).detach()
            ent = first_step_entropy(scores, valid)
            loss = -(adv * log_prob).mean() - ENTROPY_COEF * ent

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_reward += reward.mean().item()
            ep_ent += ent.item()
            n_steps += 1

        val_reward = evaluate_greedy(selector, reward_fn, val_loader)
        print(f"Epoch {epoch+1}: train_reward={ep_reward/n_steps:.4f}  "
              f"entropy={ep_ent/n_steps:.3f}  greedy_val_reward={val_reward:.4f}  "
              f"(random={RANDOM_BASELINE:.3f})")

        if val_reward > best_val:
            best_val = val_reward
            torch.save(selector.state_dict(), CHECKPOINTS_DIR / f"selector_k{K}.pt")

    print(f"\nBest greedy val reward: {best_val:.4f}  (random baseline {RANDOM_BASELINE:.3f})")
    print(f"Saved to {CHECKPOINTS_DIR / f'selector_k{K}.pt'}")


if __name__ == "__main__":
    main()
