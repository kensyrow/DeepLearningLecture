

import torch
import numpy as np

# Dataset
x_train = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
y_train = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)

# Hyper-parameters
learning_rate = 0.01
num_epochs = 2000

# Linear regression model
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Load the model
model = Model()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):

    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    # Forward
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Test the model
hour = torch.Tensor([[4.0]])
output = model(hour)
print("Predicted point (after training): {}".format(output.data[0][0]))
