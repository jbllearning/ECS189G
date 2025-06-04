import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import matplotlib.pyplot as plt

# Ensure correct import paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, CURRENT_DIR)

from Dataset_Loader_Node_Classification import Dataset_Loader
from Result_Saver import Result_Saver


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = self.gc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) # adding dropout
        x = torch.mm(adj, x)
        x = self.gc2(x)
        return x

def compute_metrics(true, pred, average='macro'):
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred, average=average, zero_division=0)
    rec = recall_score(true, pred, average=average, zero_division=0)
    f1 = f1_score(true, pred, average=average, zero_division=0)
    return acc, prec, rec, f1

def train(model, features, adj, labels, idx_train, idx_test, epochs=200, lr=0.01, weight_decay=5e-4, eval_interval=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = [] #stores evaluation metrics every eval_interval epochs
    loss_history = []
    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_fn(output[idx_train], labels[idx_train])
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()

        # evaluate every eval_interval number of epochs
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(features, adj)
                pred = logits.argmax(dim=1).cpu().numpy()
                true = labels.cpu().numpy()
                acc, prec, rec, f1 = compute_metrics(true[idx_test], pred[idx_test])
                history.append({
                    "epoch": epoch + 1,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                })
                print(f"Epoch {epoch + 1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = model.state_dict()

    # restore best state for best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, model, loss_history


def evaluate(model, features, adj, labels, idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output.argmax(dim=1)
        correct = pred[idx] == labels[idx]
        acc = correct.sum().item() / len(idx)
    return acc


def run_GCN_train_eval(dataset, dropout=0.5, epochs=200, eval_interval=10, lr=0.01, weight_decay=5e-4):
    # Settings to log for your report
    settings = {
        "model": "2-layer GCN",
        "layer_dimensions": "input -> hidden -> output = {} -> {} -> {}".format("depends on data", 16, "num_classes"),
        "dropout": dropout,
        "activation": "ReLU",
        "optimizer": "Adam",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "initialization": "PyTorch default (Kaiming for Linear layers)",
        "dataset": dataset
    }
    print("=== Model & Experiment Settings ===")
    for k, v in settings.items():
        print(f"{k}: {v}")
    print("===================================")

    # Set up dataset path using absolute path
    data_path = os.path.join(PROJECT_ROOT, "data", "stage_5_data", dataset)

    # Load the dataset using the Dataset_Loader class
    loader = Dataset_Loader()
    loader.dataset_name = dataset
    loader.dataset_source_folder_path = data_path
    loaded_data = loader.load()

    features = loaded_data['graph']['X']
    adj = loaded_data['graph']['utility']['A']
    labels = loaded_data['graph']['y']
    idx_train = loaded_data['train_test_val']['idx_train']
    idx_val = loaded_data['train_test_val']['idx_val']
    idx_test = loaded_data['train_test_val']['idx_test']

    input_dim = features.shape[1]
    hidden_dim = 16
    output_dim = int(labels.max().item()) + 1

    # Print actual input/output dims for reporting
    print(f"Dataset: {dataset}")
    print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")

    model = GCN(input_dim, hidden_dim, output_dim, dropout=dropout)
    history, model, loss_history= train(
        model, features, adj, labels, idx_train, idx_test,
        epochs=epochs, lr=lr, weight_decay=weight_decay, eval_interval=eval_interval
    )

    # Final evaluation on test set with best model
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        pred = logits.argmax(dim=1).cpu().numpy()
        true = labels.cpu().numpy()
        acc, prec, rec, f1 = compute_metrics(true[idx_test], pred[idx_test])
        print("\n=== Final Test Metrics ===")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print("=========================")

    # Save result
    result_path = os.path.join(PROJECT_ROOT, "result", "stage_5_result", f"{dataset}_result.txt")
    with open(result_path, "w") as f:
        f.write("=== Final Test Metrics ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")

        f.write("=== Model Settings ===\n")
        settings['input_dim'] = input_dim
        settings['output_dim'] = output_dim
        for k, v in settings.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== Evaluation History ===\n")
        for h in history:
            f.write(
                f"Epoch {h['epoch']}: Acc={h['accuracy']:.4f}, Prec={h['precision']:.4f}, Rec={h['recall']:.4f}, F1={h['f1']:.4f}\n")

    print(f"result saved to {result_path}")

    # convergence plot
    plot_path = os.path.join(PROJECT_ROOT, "result", "stage_5_result", f"{dataset}_training_plot.png")
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset} training convergence plot")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
