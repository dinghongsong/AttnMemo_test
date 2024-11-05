from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SingleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24129, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv2d(1,1,3)

    def forward_once(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

class LinearNet(nn.Module):
    def __init__(self, shots) -> None:
        super().__init__()
        
        if shots == 0:
            self.fc1 = nn.Linear(196608, 128) # 0 shot
        if shots == 5:
            self.fc1 = nn.Linear(589824, 128) # 5 shots

        # self.fc1 = nn.Linear(49152, 128) # 256
        # self.fc1 = nn.Linear(98304, 128) # 512
        # self.fc1 = nn.Linear(393216, 128) # 512


        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(128)
    
    def forward_once(self, x):
        x = torch.unsqueeze(x,1)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,3)
        self.conv2 = nn.Conv2d(1,1,3)
        self.fc1 = nn.Linear(23684, 128)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

    def forward_once(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
