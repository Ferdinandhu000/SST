"""
pick_best_checkpoints.py  ―  SSTNet
=====================================
Usage
-----
# Interactive (no args): will prompt for directory names
    python pick_best_checkpoints.py

# Pass directories on the command line:
    python pick_best_checkpoints.py .checkpoints_gaosi_trans_1 .checkpoints_TC_afno

Logic
-----
For each checkpoint directory, sort all *.pt files by the trailing epoch
number in their name (e.g. "flronetfno36.pt" → epoch 36).
Then copy the 10th-from-last file (index -10 in the sorted list) to:
    checkpoints_best/<checkpoint_dir_name>/<filename>

If fewer than 10 checkpoints exist the earliest available file is copied
and a warning is printed.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
BEST_DIR = ROOT_DIR / "checkpoints_best"


def _epoch_from_name(filename: str) -> int:
    """Extract the trailing integer (epoch) from a checkpoint filename."""
    m = re.search(r'(\d+)\.pt$', filename)
    return int(m.group(1)) if m else -1


def pick_best(checkpoint_dirs: list[str]) -> None:
    for dir_name in checkpoint_dirs:
        dir_name = dir_name.strip().rstrip("/\\")
        ckpt_dir = ROOT_DIR / dir_name
        if not ckpt_dir.is_dir():
            print(f"[SKIP]  '{dir_name}' not found at {ckpt_dir}")
            continue

        pts = sorted(
            [p for p in ckpt_dir.glob("*.pt")],
            key=lambda p: _epoch_from_name(p.name),
        )
        if not pts:
            print(f"[SKIP]  '{dir_name}' contains no .pt files")
            continue

        if len(pts) < 10:
            print(f"[WARN]  '{dir_name}' has only {len(pts)} checkpoint(s); "
                  f"picking the earliest one instead of 10th-from-last.")
            chosen = pts[0]
        else:
            chosen = pts[-10]

        dest_dir = BEST_DIR / dir_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / chosen.name
        shutil.copy2(chosen, dest)

        last_epoch = _epoch_from_name(pts[-1].name)
        chosen_epoch = _epoch_from_name(chosen.name)
        print(
            f"[OK]    '{dir_name}' | last epoch={last_epoch}, "
            f"10th-from-last epoch={chosen_epoch} → {dest.relative_to(ROOT_DIR)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the 10th-from-last checkpoint from each specified directory."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        metavar="CHECKPOINT_DIR",
        help="One or more checkpoint directory names (relative to project root). "
             "Leave empty to be prompted interactively.",
    )
    args = parser.parse_args()

    if args.dirs:
        checkpoint_dirs = args.dirs
    else:
        print("Enter checkpoint directory names (one per line).")
        print("Leave a blank line and press Enter when done.\n")
        checkpoint_dirs = []
        while True:
            line = input("  Directory: ").strip()
            if not line:
                break
            checkpoint_dirs.append(line)

    if not checkpoint_dirs:
        print("No directories specified. Exiting.")
        sys.exit(0)

    print(f"\nOutput root: {BEST_DIR}\n")
    pick_best(checkpoint_dirs)
    print("\nDone.")


if __name__ == "__main__":
    main()
