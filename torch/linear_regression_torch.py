"""
In this script, we will be synthetically creating a linearly correlated dataset and fitting a linear model using pytorch
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

N = 100
X = np.sort(np.random.uniform(0, 1, N))
Y = np.sort(np.random.normal(0, 1, N))

plt.scatter(X, Y)
plt.title("Dataset")
plt.show()

linear_model = torch.nn.Linear(1, 1)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.05)

# note that PyTorch uses float 32 and numpy uses float 64.
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

epochs = 200
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    X_out = linear_model(X)
    loss = criterion(X_out, Y)

    # I always found it odd to to .item(), but this is because we want to extract python float32
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item()}")

plt.plot(losses)
plt.title("losses")
plt.show()

prediction = linear_model(X).detach().numpy()

plt.scatter(X, Y, label='original')
plt.plot(X, prediction, label='prlinear_modelediction')
plt.legend()
plt.show()

with torch.no_grad():
  out = linear_model(X).numpy()

plt.scatter(X, Y, label='original 1')
plt.plot(X, out, label='prediction 1')
plt.legend()
plt.show()

w = linear_model.weight.data.numpy()
b = linear_model.bias.data.numpy()
print(w, b)