# ============================================================
# run_pipeline.py
# ============================================================

from preprocessing import main as preprocess_main
from train import main as train_main


if __name__ == "__main__":
    preprocess_main()
    train_main()
