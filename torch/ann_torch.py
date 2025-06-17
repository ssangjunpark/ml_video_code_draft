import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

import torchvision
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

train_data = MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="data", train=False, download=True, transform=ToTensor())

random_idx = random.randint(0, len(train_data.data)-1)
plt.imshow(train_data.data[random_idx])
plt.title(train_data.targets[random_idx].item())
plt.show()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, len(train_data.classes)),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=128,
    shuffle=True
)

train_losses = []
test_losses= []

epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss_accum = 0.0
    for X_train, y_train in train_loader:
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_loss_accum += loss.item()
    
    train_losses.append(train_loss_accum / len(X_train))

    model.eval()
    test_loss_accum = 0.0
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
            
        test_loss_accum += loss.item()
    test_losses.append(test_loss_accum / len(X_test))

    print(f"Epoch: {epoch}, Train Loss: {train_losses[epoch]}, Test Loss: {test_losses[epoch]}")

plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.show()

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        predicts = outputs.argmax(dim=1)
        correct += (predicts == y_test).sum().item()

    print(f"Test Data Accuracy: {correct / len(test_loader.dataset)}")

with torch.no_grad():
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        predicts = outputs.argmax(dim=1)
        print(predicts)
        print(y_test)
        
        for idx in range(len(predicts)):
            if predicts[idx] != y_test[idx]:
                plt.imshow(X_test[idx][0])
                plt.title(f"Predicted: {predicts[idx]}  Actual: {y_test[idx]}")
                plt.show()

        break

torch.save(model.state_dict(), 'mymodel.pt')