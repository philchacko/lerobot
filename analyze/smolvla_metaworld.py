"""
SmolVLA zero-shot eval on MetaWorld.

Run from repo root (WSL2, conda env lerobot):
    python analyze/smolvla_metaworld.py

What this does:
- Loads smolvla_base from HuggingFace Hub
- Runs it zero-shot against assembly-v3 (pick nut → peg)
- Prints per-step reward and success flag
- Expects near-zero success — this establishes the zero-shot baseline

Inputs SmolVLA expects (built manually here, no pipeline):
  observation.images.<key>    float32 tensor (1, C, H, W) in [0, 1]
  observation.state           float32 tensor (1, state_dim)
  observation.language.tokens int64  tensor (1, max_length)
  observation.language.attention_mask bool tensor (1, max_length)
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.envs.metaworld import MetaworldEnv, TASK_DESCRIPTIONS
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

TASK = "assembly-v3"
N_EPISODES = 3
MAX_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(obs, lang_tokens, lang_mask, image_key, device):
    """Convert a raw MetaWorld obs dict into the batch dict select_action expects."""
    # pixels: (H, W, C) uint8 → float32 (C, H, W) [0, 1] → add batch dim
    img = torch.from_numpy(obs["pixels"]).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)          # (1, C, H, W)

    # agent_pos: (4,) float64 → float32, add batch dim
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)  # (1, 4)

    return {
        image_key: img,
        "observation.state": state,
        "observation.language.tokens": lang_tokens.to(device),
        "observation.language.attention_mask": lang_mask.to(device),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Task: {TASK}")
    print(f"Task description: {TASK_DESCRIPTIONS[TASK]!r}\n")

    # Load policy
    print("Loading smolvla_base ...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval()
    policy = policy.to(DEVICE)

    # Inspect what image key this checkpoint expects
    image_keys = list(policy.config.image_features.keys())
    print(f"Policy image_features keys: {image_keys}")
    if len(image_keys) != 1:
        print(
            "WARNING: expected exactly 1 image key, got "
            f"{len(image_keys)}. Using the first one."
        )
    image_key = image_keys[0]
    print(f"Using image key: {image_key!r}\n")

    # Tokenize the task description once (reused across all steps)
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    task_text = TASK_DESCRIPTIONS[TASK] + "\n"      # NewLineTaskProcessorStep adds \n
    tokenized = tokenizer(
        [task_text],
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        padding="longest",
        padding_side="right",
        return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"]                         # (1, seq_len)
    lang_mask = tokenized["attention_mask"].bool()               # (1, seq_len)
    print(f"Language token shape: {lang_tokens.shape}\n")

    # Eval loop
    env = MetaworldEnv(task=TASK, obs_type="pixels_agent_pos")

    episode_successes = []
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        policy.reset()

        ep_success = False
        total_reward = 0.0

        for step in range(MAX_STEPS):
            batch = make_batch(obs, lang_tokens, lang_mask, image_key, DEVICE)

            with torch.inference_mode():
                action = policy.select_action(batch)            # (1, action_dim)

            # Policy is SO-100 trained (6D), MetaWorld needs 4D — take the first 4 dims.
            action_np = action.squeeze(0).cpu().numpy().astype(np.float64)[:4]
            obs, reward, terminated, truncated, info = env.step(action_np)

            total_reward += reward
            if info.get("is_success"):
                ep_success = True

            print(
                f"ep={ep} step={step:3d}  reward={reward:.3f}  "
                f"cumulative={total_reward:.3f}  success={info.get('is_success', False)}"
            )

            if terminated or truncated:
                break

        episode_successes.append(ep_success)
        print(f"\n--- Episode {ep} done | success={ep_success} ---\n")

    env.close()

    success_rate = sum(episode_successes) / len(episode_successes)
    print(f"Zero-shot success rate: {success_rate:.0%} ({sum(episode_successes)}/{len(episode_successes)})")


if __name__ == "__main__":
    main()
