import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)        #input size 8, output size 6. weight is 8x6 matrix
        self.linear2 = torch.nn.Linear(6, 4)        #input size 6, output size 4. weight is 6x4 matrix
        self.linear3 = torch.nn.Linear(4, 1)        #input size 4, output size 1. weight is 4x1 matrix
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)    # size_average: 是否计算损失的均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)   # lr: 梯度

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. perpare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)      # 自动调用forward
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()   #梯度归零
        loss.backward()
        # 4. Updata
        optimizer.step()
