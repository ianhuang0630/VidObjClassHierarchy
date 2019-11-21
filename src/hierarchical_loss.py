from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pickle

class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, gt):
        return self.loss(logits[0], gt[:,0].type(torch.long)) + self.loss(logits[1], gt[:, 1].type(torch.long))

class HierarchicalLoss(nn.Module):
    def __init__(self, device = 'cuda:0', class_freq=None):
        super(HierarchicalLoss, self).__init__()
        self.device = device
        self.weights = np.array([0.5, 0.25, 0.25])
        self.weights_unknown = np.array([0.5, 0.5, 0]) # last layer tree prediction doesn't matter
        # self.weights_unknown = np.array([0.75, 0.25, 0])
        # self.weights_unknown = np.array([1.0, 0.0, 0.0])
        self.weights = torch.from_numpy(self.weights).to(self.device)
        self.weights_unknown = torch.from_numpy(self.weights_unknown).to(self.device)

        # for weighted hierarchical loss, we weight the loss at predictions inversely proportional to the
        # frequency at which they occur in the ground truth labelings.
        class_weights = None
        if class_freq is not None:
            with open (class_freq, 'rb') as f:
                freq = pickle.load(f)
            total_size = sum(freq.values())
            # naive way of calculating weights: 
            prob = {element: freq[element]/total_size for element in freq}
            class_weights = {element: 1/prob[element] for element in prob}
           
        self.class_weights = class_weights

    def get_class_weights(self, classes):
        weights = []
        for class_ in classes:
            if tuple(class_) in self.class_weights:
                weights.append(self.class_weights[tuple(class_)])
            else:
                weights.append(1)

        return torch.Tensor(weights).to(self.device)

    def forward(self, probabilities, gt):

        # probabilities is a list of batchsize x max_leaves, batchsize x max_leaves, batchsize x max_leaves
        # gt: batchsize x 3
        batchsize = gt.shape[0]
        gt = gt.type(torch.LongTensor)

        weights_per_batch = []
        for is_unknown in gt[:,-1] == -1: # looking for unknown classes
            weights_per_batch.append(is_unknown * self.weights_unknown + (1- is_unknown) * self.weights)
        weights_per_batch = torch.stack(weights_per_batch).type(torch.float32)

        indices_tensor = torch.arange(batchsize)
        indices_tensor = indices_tensor.type(torch.LongTensor)
        indices_tensor = indices_tensor.to(self.device)

        selections = torch.stack(
            [level[indices_tensor, gt[:, idx] if len(gt[:, idx]) >1 else [gt[:, idx]] ] for idx, level in enumerate(probabilities)]).transpose(0,1)        
        wins = torch.sum(torch.mul(selections, weights_per_batch), dim=1)
        if self.class_weights is not None:
            class_weights = self.get_class_weights(gt.detach().cpu().numpy().tolist())
            # harsher punishment
            return torch.mean(- class_weights * torch.log(wins))

            # return torch.mean(- torch.log(class_weights) * torch.log(wins))

        else:
            return torch.mean(-torch.log(wins))


        # wins = torch.matmul(selections.type(torch.float64), self.weights.t())



if __name__=='__main__':

    HL = HierarchicalLoss()

    level1 = np.array([[0.2, 0.2, 0.6], 
                        [0.2, 0.3, 0.5]])
    level1 = torch.from_numpy(level1)
    level2 = np.array([[0.1, 0.3, 0.6], 
                        [0.2, 0.3, 0.5]])
    level2 = torch.from_numpy(level2)
    level3 = np.array([[0.4, 0.3, 0.3], 
                        [0.3, 0.2, 0.5]])
    level3 = torch.from_numpy(level3)

    # move to cuda
    level1 = level1.to('cuda:0')
    level2 = level2.to('cuda:0')
    level3 = level3.to('cuda:0')

    gt = np.array([[0,1,2], 
                    [0,0,1]])
    gt = torch.from_numpy(gt)
    gt = gt.to('cuda:0')

    print(HL([level1, level2, level3], gt))

