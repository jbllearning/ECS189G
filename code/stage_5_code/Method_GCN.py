# File: ECS189G/code/stage_5_code/Method_GCN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Ensure correct import paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, CURRENT_DIR)

from Dataset_Loader_Node_Classification import Dataset_Loader
from Result_Saver import Result_Saver


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = torch.mm(adj, x)  # Graph convolution
        x = self.gc1(x)
        x = F.relu(x)
        x = torch.mm(adj, x)
        x = self.gc2(x)
        return x


def train(model, features, adj, labels, idx_train, idx_test, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_fn(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output.argmax(dim=1)
        correct = pred[idx_test] == labels[idx_test]
        acc = correct.sum().item() / len(idx_test)
    return acc, pred


def evaluate(model, features, adj, labels, idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output.argmax(dim=1)
        correct = pred[idx] == labels[idx]
        acc = correct.sum().item() / len(idx)
    return acc


def run_GCN_train_eval(dataset):
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

    model = GCN(input_dim, hidden_dim, output_dim)
    acc, pred = train(model, features, adj, labels, idx_train, idx_test)

    print(f"Accuracy on {dataset} test set: {acc:.4f}")

    # Save results using Result_Saver class
    saver = Result_Saver()
    saver.result_destination_folder_path = os.path.join(PROJECT_ROOT, "results", "stage_5_results") + "/"
    saver.result_destination_file_name = f"{dataset}_results"
    saver.fold_count = 0
    saver.data = {
        'pred': pred[idx_test].cpu().numpy().tolist(),
        'labels': labels[idx_test].cpu().numpy().tolist()
    }
    saver.save()
