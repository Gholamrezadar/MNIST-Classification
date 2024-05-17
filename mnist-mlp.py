# importing libraries
import time

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme('notebook')

import torch
import torchvision
import torch.nn as nn

# Hyperparams
BATCH_SIZE = 64

# Transforms
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Loading the dataset
dataset = torchvision.datasets.MNIST(root="data", train=True, download=False, transform=train_transform)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset Loaded: {len(dataset)} images")
print(f"Train Dataloader: {len(train_dataloader)}")
print(f"Validation Dataloader: {len(val_dataloader)}")


