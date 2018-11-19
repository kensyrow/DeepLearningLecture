import torch
import torch.nn as nn
import numpy as np

# Dataset
x_train = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
y_train = np.array([[0.], [0.], [1.], [1.]], dtype=np.float32)

# Hyper-parameters
learning_rate = 0.01
num_epochs = 2000


# Logistic regression model
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        y_pred = self.sigmoid(out)
        return y_pred


# our model
model = Model()

# Loss and optimizer
criterion = torch.nn.BCELoss()
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
hour_var = torch.Tensor([[1.0]])
output = model(hour_var)
print("Predicted result (after training): {}".format(output.data[0][0] > 0.5))

hour_var = torch.Tensor([[7.0]])
output = model(hour_var)
print("Predicted result (after training): {}".format(output.data[0][0] > 0.5))
