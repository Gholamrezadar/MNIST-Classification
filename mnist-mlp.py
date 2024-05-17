# importing libraries
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme('notebook')

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Hyperparams
BATCH_SIZE = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Transforms
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Loading the dataset
dataset = torchvision.datasets.MNIST(root="data", train=True, download=False, transform=train_transform)
classes = dataset.classes

# select the first 1000 images
# dataset = torch.utils.data.Subset(dataset, range(len(dataset)//6))

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset Loaded: {len(dataset)} images")
print(f"Train Dataloader: {len(train_dataloader)} batches({BATCH_SIZE})")
print(f"Validation Dataloader: {len(val_dataloader)} batches({BATCH_SIZE})")

# Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512) # 784x128
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10) # 128x10

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def train(model, dataloader, optimizer:torch.optim.Adam, epoch):
    model.train()
    loss_history = []
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.reshape(-1, X.shape[2]*X.shape[3])
        output = model(X)
        loss = F.cross_entropy(output, y)
        loss_history.append(loss.detach().cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx%200 == 0:
            print(f"batch_idx: {batch_idx:<4} | train loss: {loss:.4f}")
        
    return loss_history
        

def eval(model, dataloader):
    model.eval()
    loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1, X.shape[2]*X.shape[3])
            output = model(X)
            loss += F.cross_entropy(output, y)
            
            #TODO: calculate accuracy
            acc += (output.argmax(dim=1) == y).float().mean()
    
    acc /= len(dataloader)
    loss /= len(dataloader)
    print(f"validation loss: {loss:.4f} | accuracy: {acc:.4f}")
    return loss, acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

EPOCHS = 10

model = MLP()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Model Parameters: {count_parameters(model):,}")

print("Starting Training...")
train_loss_history_global = []
valid_loss_history_global = []
valid_acc_history_global = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    train_loss_history = train(model, train_dataloader, optimizer, epoch)
    train_loss_history_global.extend(train_loss_history)
    valid_loss, valid_acc = eval(model, val_dataloader)
    valid_loss_history_global.append(valid_loss)
    valid_acc_history_global.append(valid_acc)
    print()

plt.plot(train_loss_history_global, label="train_loss")
plt.plot([(i+1)*len(train_dataloader) for i in range(EPOCHS)], valid_loss_history_global, label="valid_loss", marker='o', color='r', linewidth=3)
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.title("Training Loss")
plt.ylim(0, 0.5)
# plt.show()
plt.savefig("mlp_loss.png")


plt.plot([(i+1)*len(train_dataloader) for i in range(EPOCHS)], valid_acc_history_global, label="valid_acc", marker='o', color='g', linewidth=3)
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
plt.title("Validation Accuracy")
plt.show()
plt.savefig("mlp_accuracy.png")

# Save the model
torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved!")

# Load the model
# model = MLP()
# model.load_state_dict(torch.load("mlp_model.pth"))

# confusion matrix
from sklearn.metrics import confusion_matrix
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for batch_idx, (X, y) in enumerate(val_dataloader):
        X, y = X.to(device), y.to(device)
        X = X.reshape(-1, X.shape[2]*X.shape[3])
        output = model(X)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(output.argmax(dim=1).cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
# plt.show()
plt.savefig("mlp_confusion_matrix.png")