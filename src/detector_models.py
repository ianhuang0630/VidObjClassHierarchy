"""
Implementations of novelty classification
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaDetector(nn.Module):
    def __init__(self, in_dim = 4, num_classes = 3):
        super(VanillaDetector, self).__init__()

        self.fc1 = nn.Linear(in_dim, 4)
        self.fc2 = nn.Linear(4, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        A = self.relu(self.fc1(x))
        return self.fc2(A)

class DeeperDetector(nn.Module):
    def __init__(self, in_dim=4, num_classes = 3):
        super(DeeperDetector, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 4)
        self.fc2 = nn.Linear(4, 6)
        self.fc3 = nn.Linear(6, 8)
        self.fc4 = nn.Linear(8,4)
        self.fc5 = nn.Linear(4, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        A = self.relu(self.fc1(x))
        A = self.relu(self.fc2(A))
        A = self.relu(self.fc3(A))
        A = self.relu(self.fc4(A))
        return self.fc5(A)

# GMM approach?

if __name__ == '__main__':

    pass