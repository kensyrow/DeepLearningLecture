

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Data split (Train: 0.9, Test: 0.1)
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
num_of_trains = int(len(xy) * 0.9)

xy_train = xy[:num_of_trains]
xy_test = xy[num_of_trains:]

print("len(xy_train):", len(xy_train))
print("len(xy_test):", len(xy_test))


# Hyper-parameters
learning_rate = 0.01
num_epochs = 200
batch_size = 32

# Train dataset
class DiabetesTrainDataset(Dataset):
    def __init__(self):
        self.len = xy_train.shape[0]
        self.x_data = torch.from_numpy(xy_train[:, 0:-1])
        self.y_data = torch.from_numpy(xy_train[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# Test dataset
class DiabetesTestDataset(Dataset):
    def __init__(self):
        self.len = xy_test.shape[0]
        self.x_data = torch.from_numpy(xy_test[:, 0:-1])
        self.y_data = torch.from_numpy(xy_test[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = DiabetesTrainDataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = DiabetesTrainDataset()
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# Logistic regression model
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.l1(x))
        out2 = self.relu(self.l2(out1))
        y_pred = torch.sigmoid(self.l3(out2))
        return y_pred

class DeepModel(torch.nn.Module):

    def __init__(self):
        super(DeepModel, self).__init__()
        self.l1 = torch.nn.Linear(8, 16)
        self.l2 = torch.nn.Linear(16, 32)
        self.l3 = torch.nn.Linear(32, 64)
        self.l4 = torch.nn.Linear(64, 32)
        self.l5 = torch.nn.Linear(32, 16)
        self.l6 = torch.nn.Linear(16, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.l1(x))
        out2 = self.relu(self.l2(out1))
        out3 = self.relu(self.l3(out2))
        out4 = self.relu(self.l4(out3))
        out5 = self.relu(self.l5(out4))
        y_pred = torch.sigmoid(self.l6(out5))
        return y_pred

# Load the model
model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(num_epochs):
    for index, (inputs, labels) in enumerate(train_loader):

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Test the model
model.eval()

total = 0
correct = 0

for index, (inputs, labels) in enumerate(test_loader):
    # Forward
    outputs = model(inputs)

    # Count correct examples
    total += labels.size(0)
    correct += ((outputs.data > 0.5).type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()

print('Accuracy: {} %'.format(100 * correct / total))
