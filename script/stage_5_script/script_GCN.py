
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "stage_5_code")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "stage_5_result")

sys.path.insert(0, CODE_DIR)

from Method_GCN import run_GCN_train_eval

if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)

    datasets = ["cora", "citeseer", "pubmed"]
    for dataset in datasets:
        print(f"\nRunning GCN on dataset: {dataset}")
        run_GCN_train_eval(dataset)

    # try dropout values 0.5, 0.7, 0.9 for each dataset
    # for dp in [0.5, 0.7, 0.9]:
    #   print(f"\n--- Dropout {dp} ---")
    run_GCN_train_eval('cora', dropout=0.7)
    run_GCN_train_eval('citeseer', dropout=0.7)
    run_GCN_train_eval('pubmed', dropout=0.7)
