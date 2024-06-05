import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
