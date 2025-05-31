# File: ECS189G/script/stage_5_script/run_GCN.py

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "stage_5_code")

# Add to system path
sys.path.insert(0, CODE_DIR)

from Method_GCN import run_GCN_train_eval

if __name__ == '__main__':
    os.makedirs("../../results/stage_5_results", exist_ok=True)

    datasets = ["cora", "citeseer", "pubmed"]
    for dataset in datasets:
        print(f"\nRunning GCN on dataset: {dataset}")
        run_GCN_train_eval(dataset)
