# LeRobot Learning Plan

## Status Legend
- [x] Done
- [~] In progress
- [ ] Not started

---

## Milestone 1 — Simulation Baseline (DONE)

**Goal:** Understand the full sim loop end-to-end before touching hardware.

- [x] Load `lerobot/aloha_sim_transfer_cube_human` dataset
- [x] Inspect 14D action/state space (6 joints + gripper × 2 arms)
- [x] Visualize episodes in rerun.io
- [x] Read dataset features: single cam `observation.images.top`, 480×640, 50fps, AV1
- [x] Migrate pretrained ACT checkpoint to 0.5.x norm format
- [x] Run eval: `act_aloha_sim_transfer_cube_human` → **4/5 success** on `AlohaTransferCube-v0`
- [x] Read reward function source (`gym_aloha/env.py`, `tasks/sim.py`) — success = `reward == 4`

**Output:** `outputs/migrated/act_aloha_sim_transfer_cube_human/`, eval videos in `outputs/eval/`

---

## Milestone 2 — SmolVLA Zero-Shot on MetaWorld (NEXT)

**Goal:** Run a language-conditioned VLA policy against a new sim env with no fine-tuning.

### Steps

- [ ] Write `analyze/smolvla_metaworld.py`
  - Load `lerobot/smolvla_base` from Hub
  - Instantiate `MetaworldEnv(task='assembly-v3', obs_type='pixels_agent_pos')`
  - Figure out `policy.select_action()` input format (check `modeling_smolvla.py`)
  - Inject task description: `TASK_DESCRIPTIONS['assembly-v3']` → `"Pick up a nut and place it onto a peg"`
  - Run 200-step rollout, print per-step reward + `info['is_success']`
- [ ] Check if SmolVLA needs a `task` key in obs dict or a separate `language_instruction` arg
- [ ] Run zero-shot eval (expect near-zero success — this establishes the baseline)
- [ ] Try 2-3 other MetaWorld tasks to compare zero-shot transfer

### Key file to read first
`src/lerobot/policies/smolvla/modeling_smolvla.py` — specifically `select_action()` signature.

**Expected outcome:** Understand what SmolVLA needs as input, document the interface.

---

## Milestone 3 — Hardware Setup (SO-101 Arriving Soon)

**Goal:** Get the physical arm talking to WSL2 and collect first teleoperated demos.

### Steps

- [ ] USB forwarding: `usbipd attach --wsl --busid <BUSID>` (PowerShell admin)
- [ ] Verify in WSL2: `ls /dev/ttyACM*`
- [ ] Run LeRobot servo calibration script
- [ ] Run teleoperation: collect 50+ demos of a simple pick-and-place task
- [ ] Push dataset to HuggingFace Hub under your username

### Reference
https://huggingface.co/docs/lerobot/so101

---

## Milestone 4 — Train ACT on Real Robot Data

**Goal:** Train a task-specific policy and close the sim→real loop.

### Steps

- [ ] Choose a repeatable, low-clutter task (e.g., pick cube and drop in bin)
- [ ] Collect 50–100 demos via teleoperation
- [ ] Train ACT (simpler architecture, faster feedback loop than SmolVLA):
  ```bash
  lerobot-train \
    --policy.type=act \
    --dataset.repo_id=YOUR_HF_USERNAME/YOUR_DATASET \
    --output_dir=outputs/train/so101_act \
    --policy.device=cuda \
    --steps=50000
  ```
- [ ] Eval on hardware, measure success rate over 10 episodes
- [ ] Iterate: more demos, longer training, or try different task

---

## Milestone 5 — SmolVLA Fine-Tuning (Stretch)

**Goal:** Fine-tune the language-conditioned VLA on your real robot data.

- [ ] Confirm SmolVLA fine-tuning is supported in 0.5.x (check `lerobot-train --policy.type=smolvla`)
- [ ] Run fine-tuning on SO-101 demos
- [ ] Compare: zero-shot SmolVLA vs fine-tuned SmolVLA vs ACT

---

## Concepts Reference

| Concept | Key Point |
|---|---|
| Gymnasium interface | `reset()` / `step(action)` / `action_space` — identical sim and real |
| ACT | VAE encoder → latent z, chunk size 100, ResNet18 vision, 14D actions |
| SmolVLA | 450M, language-conditioned, needs task description string |
| Norm migration | 0.5.x moved norm stats out of model weights into JSON files |
| Reward (Aloha) | `reward == 4` = success; staged rewards for grasp sub-stages |
| VLA vs ACT | ACT: task-specific IL, no language. SmolVLA: generalist, language-conditioned |
| Training hw | RTX 3080 fine for ACT + SmolVLA fine-tuning; data center for VLA pretraining |
