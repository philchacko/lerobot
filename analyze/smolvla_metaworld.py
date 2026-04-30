"""
SmolVLA MetaWorld eval — uses the lerobot/smolvla_metaworld checkpoint,
which was trained on MetaWorld MT50 (50 tasks, 205k frames).

Run from repo root (WSL2, conda env lerobot):
    python analyze/smolvla_metaworld.py

Videos are saved to outputs/eval/smolvla_metaworld/ and can be opened directly
on Windows (WSL2 writes to the Windows filesystem).

Inputs SmolVLA expects (built manually here, no pipeline):
  observation.images.<key>         float32 (1, C, H, W) in [0, 1]
  observation.state                float32 (1, state_dim), normalized
  observation.language.tokens      int64   (1, max_length)
  observation.language.attention_mask  bool (1, max_length)

State normalization: the checkpoint ships with a policy_preprocessor that
contains MEAN_STD stats. We load them here so state is on the right scale.
Without normalization the arm would receive raw meter-scale EEF coordinates
while it was trained on z-scored inputs.
"""

import sys
sys.path.insert(0, "src")

import json
from pathlib import Path

import imageio
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from lerobot.envs.metaworld import MetaworldEnv, TASK_DESCRIPTIONS
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_ID = "lerobot/smolvla_metaworld"
TASK = "assembly-v3"
N_EPISODES = 3
MAX_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_DIR = Path("outputs/eval/smolvla_metaworld")


def load_state_norm_stats(model_dir: Path):
    """
    Pull state mean/std out of the policy_preprocessor.json that ships with the
    checkpoint. Returns (mean, std) as float32 tensors, or (None, None) if the
    preprocessor doesn't contain STATE normalizer stats.
    """
    preprocessor_path = model_dir / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        print("  policy_preprocessor.json not found — skipping state normalization")
        return None, None

    with open(preprocessor_path) as f:
        pre = json.load(f)

    # Walk the steps list looking for a NormalizerProcessorStep
    for step in pre.get("steps", []):
        if step.get("type") != "normalizer_processor":
            continue
        stats = step.get("stats", {})
        obs_state = stats.get("observation.state")
        if obs_state is None:
            continue
        mean = torch.tensor(obs_state["mean"], dtype=torch.float32)
        std  = torch.tensor(obs_state["std"],  dtype=torch.float32)
        print(f"  Loaded state norm stats: mean={mean.tolist()}, std={std.tolist()}")
        return mean, std

    print("  No STATE norm stats found in preprocessor — skipping state normalization")
    return None, None


def make_batch(obs, lang_tokens, lang_mask, image_key, state_mean, state_std, device):
    """Convert a raw MetaWorld obs dict into the batch dict select_action expects."""
    # pixels: (H, W, C) uint8 → float32 (C, H, W) in [0, 1] → add batch dim
    img = torch.from_numpy(obs["pixels"]).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)          # (1, C, H, W)

    # agent_pos: (4,) float64 → float32, normalize if stats available
    state = torch.from_numpy(obs["agent_pos"]).float()           # (4,)
    if state_mean is not None and state_std is not None:
        mean = state_mean.to(state.device)
        std  = state_std.to(state.device)
        # Clamp std to avoid div-by-zero on constant dims
        state = (state - mean[:len(state)]) / std[:len(state)].clamp(min=1e-6)
    state = state.unsqueeze(0).to(device)                        # (1, 4)

    return {
        image_key: img,
        "observation.state": state,
        "observation.language.tokens": lang_tokens.to(device),
        "observation.language.attention_mask": lang_mask.to(device),
    }


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_ID}")
    print(f"Task:   {TASK}  →  {TASK_DESCRIPTIONS[TASK]!r}")
    print(f"Videos: {VIDEO_DIR.resolve()}\n")

    # Download full snapshot so we can read the preprocessor JSON for norm stats
    print(f"Downloading {MODEL_ID} ...")
    model_dir = Path(snapshot_download(MODEL_ID))
    print(f"Checkpoint dir: {model_dir}\n")

    print("Loading state normalization stats ...")
    state_mean, state_std = load_state_norm_stats(model_dir)

    print("\nLoading policy ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()
    policy = policy.to(DEVICE)

    image_keys = list(policy.config.image_features.keys())
    image_key  = image_keys[0]
    print(f"Image key in use: {image_key!r}  (of {len(image_keys)} expected cameras)\n")

    # Tokenize task description once — reused every step
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    task_text  = TASK_DESCRIPTIONS[TASK] + "\n"
    tokenized  = tokenizer(
        [task_text],
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        padding="longest",
        padding_side="right",
        return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"]
    lang_mask   = tokenized["attention_mask"].bool()
    print(f"Language tokens: {lang_tokens.shape}\n")

    env = MetaworldEnv(task=TASK, obs_type="pixels_agent_pos")

    episode_successes = []
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        policy.reset()

        ep_success  = False
        total_reward = 0.0
        frames = [obs["pixels"]]

        for step in range(MAX_STEPS):
            batch = make_batch(
                obs, lang_tokens, lang_mask,
                image_key, state_mean, state_std, DEVICE,
            )

            with torch.inference_mode():
                action = policy.select_action(batch)    # (1, 4) — native 4D output

            action_np = action.squeeze(0).cpu().numpy().astype(np.float64)
            obs, reward, terminated, truncated, info = env.step(action_np)

            frames.append(obs["pixels"])
            total_reward += reward
            if info.get("is_success"):
                ep_success = True

            print(
                f"ep={ep} step={step:3d}  reward={reward:.3f}  "
                f"total={total_reward:.3f}  success={info.get('is_success', False)}"
            )

            if terminated or truncated:
                break

        label = "success" if ep_success else "fail"
        video_path = VIDEO_DIR / f"ep{ep:02d}_{label}.mp4"
        imageio.mimsave(str(video_path), frames, fps=20)
        print(f"\nSaved → {video_path}")

        episode_successes.append(ep_success)
        print(f"--- Episode {ep} | success={ep_success} ---\n")

    env.close()

    n_success = sum(episode_successes)
    print(f"Success rate: {n_success}/{N_EPISODES}  ({n_success/N_EPISODES:.0%})")
    print(f"Videos in:    {VIDEO_DIR.resolve()}")


if __name__ == "__main__":
    main()
