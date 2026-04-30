"""
SmolVLA zero-shot eval on MetaWorld.

Run from repo root (WSL2, conda env lerobot):
    python analyze/smolvla_metaworld.py

Videos are saved to outputs/eval/smolvla_metaworld/ and can be opened directly
on Windows (WSL2 writes to the Windows filesystem).

Inputs SmolVLA expects (built manually here, no pipeline):
  observation.images.<key>    float32 tensor (1, C, H, W) in [0, 1]
  observation.state           float32 tensor (1, state_dim)
  observation.language.tokens int64  tensor (1, max_length)
  observation.language.attention_mask bool tensor (1, max_length)
"""

import sys
sys.path.insert(0, "src")

from pathlib import Path

import imageio
import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.envs.metaworld import MetaworldEnv, TASK_DESCRIPTIONS
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

TASK = "assembly-v3"
N_EPISODES = 3
MAX_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_DIR = Path("outputs/eval/smolvla_metaworld")


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
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Task: {TASK}")
    print(f"Task description: {TASK_DESCRIPTIONS[TASK]!r}")
    print(f"Videos will be saved to: {VIDEO_DIR.resolve()}\n")

    # Load policy
    print("Loading smolvla_base ...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval()
    policy = policy.to(DEVICE)

    # Inspect what image key this checkpoint expects
    image_keys = list(policy.config.image_features.keys())
    print(f"Policy image_features keys: {image_keys}")
    if len(image_keys) != 1:
        print(f"WARNING: expected 1 image key, got {len(image_keys)}. Using the first one.")
    image_key = image_keys[0]
    print(f"Using image key: {image_key!r}\n")

    # Tokenize the task description once — reused every step
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    task_text = TASK_DESCRIPTIONS[TASK] + "\n"      # NewLineTaskProcessorStep appends \n
    tokenized = tokenizer(
        [task_text],
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        padding="longest",
        padding_side="right",
        return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"]            # (1, seq_len)
    lang_mask = tokenized["attention_mask"].bool()  # (1, seq_len)
    print(f"Language token shape: {lang_tokens.shape}\n")

    env = MetaworldEnv(task=TASK, obs_type="pixels_agent_pos")

    episode_successes = []
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        policy.reset()

        ep_success = False
        total_reward = 0.0
        frames = [obs["pixels"]]  # capture the initial frame

        for step in range(MAX_STEPS):
            batch = make_batch(obs, lang_tokens, lang_mask, image_key, DEVICE)

            with torch.inference_mode():
                action = policy.select_action(batch)        # (1, action_dim)

            # Policy outputs 6D (SO-100 arm); MetaWorld expects 4D — truncate.
            action_np = action.squeeze(0).cpu().numpy().astype(np.float64)[:4]
            obs, reward, terminated, truncated, info = env.step(action_np)

            frames.append(obs["pixels"])
            total_reward += reward
            if info.get("is_success"):
                ep_success = True

            print(
                f"ep={ep} step={step:3d}  reward={reward:.3f}  "
                f"cumulative={total_reward:.3f}  success={info.get('is_success', False)}"
            )

            if terminated or truncated:
                break

        # Save episode video
        video_path = VIDEO_DIR / f"ep{ep:02d}_{'success' if ep_success else 'fail'}.mp4"
        imageio.mimsave(str(video_path), frames, fps=20)
        print(f"\nSaved video → {video_path}")

        episode_successes.append(ep_success)
        print(f"--- Episode {ep} done | success={ep_success} ---\n")

    env.close()

    success_rate = sum(episode_successes) / len(episode_successes)
    print(f"Zero-shot success rate: {success_rate:.0%} ({sum(episode_successes)}/{len(episode_successes)})")
    print(f"\nAll videos in: {VIDEO_DIR.resolve()}")


if __name__ == "__main__":
    main()
