"""
In this script, we will be performng binary classificaiton using logisitc regresion for simplified Iris Dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris_dataset = load_iris()
X = iris_dataset.data
y = (iris_dataset.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = nn.Sequential(
    nn.Linear(X_train.data.shape[1], 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

epochs = 500
losses_train = []
losses_test = []

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    losses_train.append(loss.item())
    losses_test.append(loss_test.item())

    print(f"Epoch: {epoch}/{epochs}, Train Loss: {loss.item()}, Test Loss: {loss_test.item()}")

plt.plot(losses_train, label='train loss')
plt.plot(losses_test, label='test loss')
plt.legend()
plt.show()

with torch.no_grad():
    prediction_train = model(X_train)
    train_accuracy = np.mean(np.round(prediction_train.numpy()) == y_train.numpy())

    prediction_test = model(X_test)
    test_accuracy = np.mean(np.round(prediction_test.numpy()) == y_test.numpy())

print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

torch.save(model.state_dict(), 'mymodel.pt')

model2 = nn.Sequential(
    nn.Linear(X_train.data.shape[1], 1),
    nn.Sigmoid()
)

model2.load_state_dict(torch.load('mymodel.pt'))

with torch.no_grad():
    prediction_train2 = model2(X_train)
    train_accuracy2 = np.mean(np.round(prediction_train2.numpy()) == y_train.numpy())

    prediction_test2 = model2(X_test)
    test_accuracy2 = np.mean(np.round(prediction_test2.numpy()) == y_test.numpy())

print(f"Train Accuracy: {train_accuracy2:.4f}, Test Accuracy: {test_accuracy2:.4f}")

print(model.state_dict())