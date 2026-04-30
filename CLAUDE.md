# LeRobot — Claude Code Context

## Environment

Two machines are in use. This repo lives at `C:\Users\philc\code\lerobot` on Windows,
mounted inside WSL2 at `/mnt/c/Users/philc/code/lerobot`.

**Windows WSL2 (primary for training/eval)**
- conda env `lerobot`, Python 3.12
- LeRobot 0.5.2, installed from source
- Extras: `aloha`, `pusht`, `hilserl`, `metaworld`, `smolvla`
- CUDA: RTX 3080 Laptop, confirmed working
- HF_TOKEN in `~/.bashrc`
- Known harmless warnings: evdev build failure (gamepad HIL-SERL only), cmake 4.3.2 vs pinned <4.2.0

**Mac M2 (secondary, visualization/inspection only)**
- conda env `lerobot`, Python 3.12
- Extras: `aloha`, `pusht`, `metaworld`, `smolvla`
- MPS backend — not suitable for serious training

## Key Paths

```
src/lerobot/                        # library source
  envs/metaworld.py                 # MetaworldEnv wrapper
  policies/smolvla/modeling_smolvla.py
  policies/act/modeling_act.py
  processor/migrate_policy_normalization.py

outputs/
  migrated/act_aloha_sim_transfer_cube_human/   # migrated ACT checkpoint (0.5.x format)
  eval/2026-04-29/14-36-30_aloha_act/           # eval run: 4/5 success on AlohaTransferCube-v0

analyze/                            # scratch scripts (not committed)
```

## Normalization Migration (0.5.x)

Hub checkpoints pre-dating 0.5.x store norm stats inside the model weights.
Run `src/lerobot/processor/migrate_policy_normalization.py` to produce a migrated
checkpoint with separate `policy_preprocessor.json` / `policy_postprocessor.json`.
Migrated ACT checkpoint already lives at `outputs/migrated/act_aloha_sim_transfer_cube_human`.

## Common Commands

```bash
# Eval migrated ACT on AlohaTransferCube-v0
lerobot-eval \
  --policy.path=outputs/migrated/act_aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=5

# Training template (SO-101, once hardware arrives)
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=YOUR_HF_USERNAME/YOUR_DATASET \
  --output_dir=outputs/train/so101_act \
  --policy.device=cuda \
  --steps=50000
```

## MetaWorld Notes

- Use `MetaworldEnv` directly from `src/lerobot/envs/metaworld.py` — NOT `gym.make`
- Obs: `pixels (480×480×3)` + `agent_pos (4,)`, action space 4D
- Task descriptions in `TASK_DESCRIPTIONS` dict — these feed SmolVLA as language input

## Hardware (SO-101, arriving soon)

USB forwarding from Windows → WSL2:
```powershell
# PowerShell (admin)
usbipd list
usbipd attach --wsl --busid <BUSID>
```
Then in WSL2: `ls /dev/ttyACM*` or `/dev/ttyUSB*`.
Docs: https://huggingface.co/docs/lerobot/so101
