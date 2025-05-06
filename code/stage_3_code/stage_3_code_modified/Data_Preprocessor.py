from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, data, dataset_name):
        self.data = data
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['label']

        # Convert to tensor and preprocess
        if self.dataset_name == 'ORL':
            # Use only R channel and normalize
            image = image[:, :, 0]  # Take R channel
            image = image / 255.0
            image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dim
        elif self.dataset_name == 'MNIST':
            image = image / 255.0
            image = torch.FloatTensor(image).unsqueeze(0)
        elif self.dataset_name == 'CIFAR10':
            image = image / 255.0
            image = torch.FloatTensor(image).permute(2, 0, 1)  # HWC to CHW

        label = torch.LongTensor([label])[0]  # Convert to tensor

        return image, label

def prepare_data(data, dataset_name, batch_size):
    dataset = ImageDataset(data, dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader