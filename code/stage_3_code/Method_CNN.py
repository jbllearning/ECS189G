'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import time


class Method_CNN(nn.Module):
    def __init__(self, mName, mDescription, dataset_name):
        super(Method_CNN, self).__init__()
        self.method_name = mName
        self.method_description = mDescription
        self.dataset_name = dataset_name

        # Configure based on dataset
        if dataset_name == 'ORL':
            self.input_channels = 1  # Using only R channel
            self.input_size = (112, 92)
            self.num_classes = 40
            self.fc_input_size = 128 * 5 * 4  # Calculated based on conv layers
        elif dataset_name == 'MNIST':
            self.input_channels = 1
            self.input_size = (28, 28)
            self.num_classes = 10
            self.fc_input_size = 128 * 3 * 3
        elif dataset_name == 'CIFAR10':
            self.input_channels = 3
            self.input_size = (32, 32)
            self.num_classes = 10
            self.fc_input_size = 128 * 3 * 3

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

        # Training parameters
        self.max_epoch = 100
        self.learning_rate = 0.001
        self.batch_size = 64 if dataset_name != 'ORL' else 16  # Smaller batch for ORL

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def train_model(self, train_loader, test_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        train_loss_history = []
        val_loss_history = []
        accuracy_history = []

        best_accuracy = 0
        best_model = None

        for epoch in range(self.max_epoch):
            self.train()
            running_loss = 0.0
            epoch_start = time.time()

            # Training loop
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            # Validation loop
            val_loss, metrics = self.evaluate(test_loader)
            train_loss = running_loss / len(train_loader.dataset)

            # Store history
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            accuracy_history.append(metrics['accuracy'])

            # Save best model
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = self.state_dict()

            # Print statistics
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch + 1}/{self.max_epoch} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {metrics["accuracy"]:.4f} | Precision: {metrics["precision"]:.4f}')
            print(f'Recall: {metrics["recall"]:.4f} | F1: {metrics["f1"]:.4f}')
            print('-' * 50)

            # Early stopping if accuracy reaches 95%
            if metrics['accuracy'] >= 0.95:
                print(f"Early stopping at {metrics['accuracy'] * 100:.2f}% accuracy")
                break

        # Load best model
        self.load_state_dict(best_model)

        # Plot training history
        self.plot_history(train_loss_history, val_loss_history, accuracy_history)

        return best_accuracy

    def evaluate(self, data_loader):
        self.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                outputs = self(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        loss = total_loss / len(data_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return loss, metrics

    def plot_history(self, train_loss, val_loss, accuracy):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(accuracy, label='Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_training_history.png')
        plt.show()