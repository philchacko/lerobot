"""
Interactive MetaWorld rollout viewer.

Pick a task from the menu (or pass it as a CLI arg), run one episode with
lerobot/smolvla_metaworld, save an MP4, and open it in Windows.

Usage (WSL2, conda env lerobot, repo root):
    python analyze/watch_rollout.py               # interactive menu
    python analyze/watch_rollout.py reach-v3      # run directly

The episode video opens automatically via Windows Explorer / default video player.
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, "src")

from lerobot.envs.metaworld import DIFFICULTY_TO_TASKS, TASK_DESCRIPTIONS

POLICY = "lerobot/smolvla_metaworld"
RENAME_MAP = '{"observation.image": "observation.images.camera1"}'
DEVICE = "cuda"

# Ordered difficulty groups for display
DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard"]


def print_menu() -> str:
    """Print numbered task menu grouped by difficulty; return chosen task name."""
    numbered: list[tuple[int, str]] = []
    idx = 1

    for level in DIFFICULTY_ORDER:
        tasks = DIFFICULTY_TO_TASKS.get(level, [])
        if not tasks:
            continue
        print(f"\n  [{level.upper()}]")
        for task in tasks:
            desc = TASK_DESCRIPTIONS.get(task, "")
            print(f"  {idx:>3}.  {task:<40}  {desc}")
            numbered.append((idx, task))
            idx += 1

    print()
    while True:
        raw = input("Enter task number or name (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            sys.exit(0)

        # Try numeric selection
        if raw.isdigit():
            n = int(raw)
            matches = [t for i, t in numbered if i == n]
            if matches:
                return matches[0]
            print(f"  No task #{n}. Try again.")
            continue

        # Try direct name match
        if raw in TASK_DESCRIPTIONS:
            return raw

        # Fuzzy: substring match
        candidates = [t for t in TASK_DESCRIPTIONS if raw.lower() in t.lower()]
        if len(candidates) == 1:
            print(f"  → matched: {candidates[0]}")
            return candidates[0]
        if len(candidates) > 1:
            print(f"  Ambiguous. Matches: {candidates}")
            continue

        print(f"  Unknown task {raw!r}. Try again.")


def run_episode(task: str) -> Path | None:
    """Run one episode via lerobot-eval, return path to the saved video."""
    desc = TASK_DESCRIPTIONS.get(task, task)
    print(f"\nTask: {task}")
    print(f"Desc: {desc}")
    print("Running rollout (1 episode)...\n")

    result = subprocess.run(
        [
            "lerobot-eval",
            f"--policy.path={POLICY}",
            "--env.type=metaworld",
            f"--env.task={task}",
            "--eval.n_episodes=1",
            "--eval.batch_size=1",
            f"--rename_map={RENAME_MAP}",
            f"--policy.device={DEVICE}",
        ],
        capture_output=False,  # stream output live so user sees progress
    )

    if result.returncode != 0:
        print("\nERROR: lerobot-eval exited with non-zero status.")
        return None

    # lerobot-eval saves videos under outputs/eval/<date>/<time>_metaworld_smolvla/
    # Find the most recently created episode video
    video_dirs = sorted(
        Path("outputs/eval").glob("**/videos/**/eval_episode_0.mp4"),
        key=lambda p: p.stat().st_mtime,
    )
    if not video_dirs:
        print("No video found under outputs/eval/")
        return None

    return video_dirs[-1]


def open_video(video_path: Path) -> None:
    """Open a video on Windows from WSL2 using explorer.exe."""
    # Convert WSL path to Windows path
    result = subprocess.run(
        ["wslpath", "-w", str(video_path.resolve())],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Video saved at: {video_path.resolve()}")
        print("(Could not auto-open — copy the path above into Windows Explorer)")
        return

    win_path = result.stdout.strip()
    print(f"\nOpening: {win_path}")
    subprocess.Popen(["explorer.exe", win_path])


def main() -> None:
    if len(sys.argv) > 1:
        task = sys.argv[1]
        if task not in TASK_DESCRIPTIONS:
            print(f"Unknown task {task!r}. Available tasks:")
            for t in sorted(TASK_DESCRIPTIONS):
                print(f"  {t}")
            sys.exit(1)
    else:
        task = print_menu()

    video = run_episode(task)
    if video:
        print(f"\nVideo: {video.resolve()}")
        open_video(video)


if __name__ == "__main__":
    main()
