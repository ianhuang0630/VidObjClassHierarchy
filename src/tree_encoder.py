"""
Implementation of the encoder for the the hierarchy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class TreeLevelPredictorWholeFrame(nn.Module):
    def __init__(self, input_shape1, input_shape2, embedding_dim=40, tree_level_option_nums = [20,20,20]):
        super(TreeLevelPredictorWholeFrame, self).__init__()

        # softmax, dropout, relu
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # conv1, conv2
        self.conv1 = nn.Conv2d(input_shape1[0], 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3,3))
        self.flat_shape = self.get_flat_fts(input_shape1, self.convs)

        self.conv1_fullframe = nn.Conv2d(input_shape2[0], 32, kernel_size=(3,3))
        self.conv2_fullframe = nn.Conv2d(32, 16, kernel_size=(3,3))
        self.flat_shape_fullframe = self.get_flat_fts(input_shape2, self.convs_fullframe)

        self.fc1 = nn.Linear(self.flat_shape + self.flat_shape_fullframe, embedding_dim)

        # assuming that the classification is done in parallel
        # self.fc_tl1 = nn.Linear(self.flat_shape, tree_level_option_nums[0])
        # self.fc_tl2 = nn.Linear(self.flat_shape, tree_level_option_nums[1])

        # assuming that the classification is done sequentially
        # fc_tl1 is
        self.fc_tl1 = nn.Linear(embedding_dim, tree_level_option_nums[0])
        self.fc_tl2 = nn.Linear(embedding_dim + tree_level_option_nums[0],
                                tree_level_option_nums[1])

        self.fc_tl3 = nn.Linear(embedding_dim + tree_level_option_nums[0] + tree_level_option_nums[1],
                                tree_level_option_nums[2])

    def convs_fullframe(self, x):
        h = self.relu(self.conv1_fullframe(x))
        h = self.relu(self.conv2_fullframe(h))
        return h

    def convs(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return h 

    def fcs(self, x):
        embedding = self.fc1(x)

        tree_level_prediction1 = self.fc_tl1(embedding)
        tree_level_prediction1 = self.relu(tree_level_prediction1)
        tree_level_prediction1 = self.softmax(tree_level_prediction1)

        # concatenating tree_level_predictions1
        pred1_embedding_concat = torch.cat((tree_level_prediction1, embedding), 1)
        tree_level_prediction2 = self.fc_tl2(pred1_embedding_concat)
        tree_level_prediction2 = self.relu(tree_level_prediction2)
        tree_level_prediction2 = self.softmax(tree_level_prediction2)

        pred2_embedding_concat = torch.cat((tree_level_prediction1, tree_level_prediction2, embedding), 1)
        tree_level_prediction3 = self.fc_tl3(pred2_embedding_concat)
        tree_level_prediction3 = self.relu(tree_level_prediction3)
        tree_level_prediction3 = self.softmax(tree_level_prediction3)

        return {'embedding': embedding,
                'tree_level_pred1': tree_level_prediction1,
                'tree_level_pred2': tree_level_prediction2,
                'tree_level_pred3': tree_level_prediction3}

    def get_flat_fts(self, in_shape, fts):
        f = fts(Variable(torch.ones(1,*in_shape)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x, fullframes):
        h_fullframes = self.convs_fullframe(fullframes)
        h = self.convs(x)
        h = h.view(-1, self.flat_shape)
        h_fullframes = h.view(-1, self.flat_shape_fullframe)

        # concatenating h_fullframes
        h_cat = torch.cat((h, h_fullframes), 1); 

        results = self.fcs(h_cat)
        return results



class TreeLevelPredictor(nn.Module):
    def __init__(self, input_shape, embedding_dim=40, tree_level_option_nums = [20,20,20]):
        # input_shape = (batch_size, 512, 7, 7)
        super(TreeLevelPredictor, self).__init__()

        # softmax, dropout, relu
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # conv1, conv2
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3,3))

        self.flat_shape = self.get_flat_fts(input_shape, self.convs)
        self.fc1 = nn.Linear(self.flat_shape, embedding_dim)

        # assuming that the classification is done in parallel
        # self.fc_tl1 = nn.Linear(self.flat_shape, tree_level_option_nums[0])
        # self.fc_tl2 = nn.Linear(self.flat_shape, tree_level_option_nums[1])

        # assuming that the classification is done sequentially
        # fc_tl1 is
        self.fc_tl1 = nn.Linear(embedding_dim, tree_level_option_nums[0])
        self.fc_tl2 = nn.Linear(embedding_dim + tree_level_option_nums[0],
                                tree_level_option_nums[1])

        self.fc_tl3 = nn.Linear(embedding_dim + tree_level_option_nums[0] + tree_level_option_nums[1],
                                tree_level_option_nums[2])


    def convs(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return h 

    def fcs(self, x):
        embedding = self.fc1(x)

        tree_level_prediction1 = self.fc_tl1(embedding)
        tree_level_prediction1 = self.relu(tree_level_prediction1)
        tree_level_prediction1 = self.softmax(tree_level_prediction1)

        # concatenating tree_level_predictions1
        pred1_embedding_concat = torch.cat((tree_level_prediction1, embedding), 1)
        tree_level_prediction2 = self.fc_tl2(pred1_embedding_concat)
        tree_level_prediction2 = self.relu(tree_level_prediction2)
        tree_level_prediction2 = self.softmax(tree_level_prediction2)

        pred2_embedding_concat = torch.cat((tree_level_prediction1, tree_level_prediction2, embedding), 1)
        tree_level_prediction3 = self.fc_tl3(pred2_embedding_concat)
        tree_level_prediction3 = self.relu(tree_level_prediction3)
        tree_level_prediction3 = self.softmax(tree_level_prediction3)

        return {'embedding': embedding,
                'tree_level_pred1': tree_level_prediction1,
                'tree_level_pred2': tree_level_prediction2,
                'tree_level_pred3': tree_level_prediction3}

    def get_flat_fts(self, in_shape, fts):
        f = fts(Variable(torch.ones(1,*in_shape)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        h = self.convs(x)
        h = h.view(-1, self.flat_shape)
        results = self.fcs(h)
        return results


class C3D_simplified(nn.Module):
    def __init__(self, input_shape, embedding_dim=40):
        super(C3D_simplified, self).__init__()
        self.dropout=nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=(3, 1, 1), padding=(1, 1, 1), stride=(2, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2,2,2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

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


        self.conv1 = nn.Conv3d(input_shape[0], 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
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

if __name__=='__main__':
    foo = torch.rand((2, 512, 7, 7))
    import ipdb; ipdb.set_trace()
    TLP = TreeLevelPredictor((512, 7, 7))
    print(TLP(foo))

