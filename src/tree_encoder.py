"""
Implementation of the encoder for the the hierarchy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class TreeEncoder(nn.Module):
    def __init__(self):
        super(TreeEncoder, self).__init__()

    def foward(self, x):
        pass

# from https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, input_shape, embedding_dim=40):
        super(C3D, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        # calculating the shape of the output layers
        self.flat_shape = self.get_flat_fts(input_shape, self.convs)

        self.fc6 = nn.Linear(self.flat_shape, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, embedding_dim)

    def get_flat_fts(self, in_shape, fts):
        f = fts(Variable(torch.ones(1,*in_shape)))
        return int(np.prod(f.size()[1:]))

    def convs(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        return h

    def forward(self, x):

        h = self.convs(x)

        h = h.view(-1, self.flat_shape)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        embedding = self.fc8(h)
        return embedding
        # logits = self.fc8(h)
        # probs = self.softmax(logits)
        # return probs
