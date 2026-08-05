"""Local compatibility entry point for the WC NNLCI workflow.

The original was a Colab export with Google Drive mounts and notebook shell
syntax. The maintained train/predict scripts live beside this launcher.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROBLEM_DIR = HERE.parent
TRAIN_SCRIPT = HERE / "training" / "train.py"
PREDICT_SCRIPT = HERE / "prediction" / "predict.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local WC NNLCI workflow")
    parser.add_argument("mode", choices=("train", "predict"))
    parser.add_argument("--epochs", type=int, help="Override the training default")
    parser.add_argument("--model", help="Model filename inside models/")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        env["NNLCI_EPOCHS"] = str(args.epochs)
    if args.model:
        env["NNLCI_MODEL_FILENAME"] = args.model

    script = TRAIN_SCRIPT if args.mode == "train" else PREDICT_SCRIPT
    subprocess.run([sys.executable, str(script)], cwd=PROBLEM_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
