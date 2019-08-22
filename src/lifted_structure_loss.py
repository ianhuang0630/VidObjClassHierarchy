from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class HierarchicalLiftedStructureLoss(nn.Module):
    def __init__(self, m, device, penalties=[0,2,4], hard_mining='hard_positive', **kwargs):
        super(HierarchicalLiftedStructureLoss, self).__init__()
        self.penalties = penalties
        self.hard_mining = hard_mining
        self.m = m
        if self.m <=2:
            raise ValueError('Cannot have groupsize less than 2.')
        # setup the number of "negative" pairs to take out from each 
        # instance in positive pair from the different penalties

        if self.m %2 == 0:
            left_neg = int(self.m/2) - 1
            right_neg = int(self.m/2) - 1
        else:
            left_neg = int(self.m/2) - 1
            right_neg = int(self.m/2)

        assert left_neg + right_neg + 2 == self.m
        # splitting left_neg
        num_negs = len(self.penalties)-1 # [2,4]
        self.num_instances_left = [int(np.round(left_neg / num_negs)) for i in range(num_negs-1)]
        self.num_instances_left.append(left_neg - sum(self.num_instances_left))
        assert sum(self.num_instances_left) == left_neg

        self.num_instances_right = [int(np.round(right_neg / num_negs)) for i in range(num_negs-1)]
        self.num_instances_right.append(right_neg - sum(self.num_instances_right))
        assert sum(self.num_instances_right) == right_neg
        self.device = device 

        self.max = nn.ReLU()
        

    def forward(self, inputs_batch, targets_batch):
        """
        Args:
            inputs: nxm matrix, m-dimensional embedding of n entities
            targets: square nxn distance matrix.
        Returns:
            loss: hierarchical lifted structure loss
        """

        loss = 0#Variable(torch.Tensor([0]).squeeze(), requires_grad=True)

        for batch_num in range(inputs_batch.shape[0]):
            inputs = inputs_batch[batch_num]
            targets = targets_batch[batch_num]

            n = inputs.size(0)

            # calculating similarity *according to the paper*
            x_tilde = torch.norm(inputs, dim=1).unsqueeze(0).t().pow(2)
            one_vec = torch.ones(n).unsqueeze(0).t()
            if self.device == 'gpu':
                one_vec = one_vec.to('cuda:0')

            similarity = torch.matmul(inputs, inputs.t())

            dist_sqrd = torch.matmul(x_tilde, one_vec.t()) + torch.matmul(one_vec, x_tilde.t()) - 2 * similarity

            if self.hard_mining is not None:
                J_ijs = torch.Tensor([])
                if self.device=='gpu':
                    J_ijs = J_ijs.to('cuda:0')

                for i in range(n):
                    
                    pos_pair_ = torch.masked_select(dist_sqrd[i], targets[i] == 0)
                    pos_pair_ = torch.masked_select(pos_pair_, pos_pair_>0)
                    pos_pair_, pos_pair_idx_ = torch.sort(pos_pair_)

                    if len(pos_pair_) == 0: # skip if cannot find positive pair
                        continue

                    # choose random pos_pair

                    if self.hard_mining == 'random_positive':
                        pos_idx = np.random.choice(len(pos_pair_idx_))
                        pos_pair_idx = pos_pair_idx_[pos_idx]
                    elif self.hard_mining == 'hard_positive':
                        pos_idx = -1
                        pos_pair_idx = pos_pair_idx_[pos_idx]

                    pos_pair = pos_pair_[pos_idx]
        
                    logs = torch.Tensor([])
                    if self.device=='gpu':
                        logs = logs.to('cuda:0')

                    # find the positive pair for the ith example
                    for idx, p in enumerate(self.penalties[1:]):
                        # from left
                        neg_pair_left = torch.masked_select(dist_sqrd[i], targets[i] == p)

                        if len(neg_pair_left) == 0:
                            continue
                        elif len(neg_pair_left) >= self.num_instances_left[idx]:
                            neg_pairs_left = torch.sort(neg_pair_left)[0][:self.num_instances_left[idx]]
                        else: # not greater than self.num_instances_left
                            neg_pairs_left = neg_pair_left

                        neg_pair_right = torch.masked_select(dist_sqrd[pos_pair_idx], targets[pos_pair_idx] == p)
                        if len(neg_pair_right) == 0:
                            continue
                        elif len(neg_pair_right) >= self.num_instances_right[idx]:
                            neg_pairs_right = torch.sort(neg_pair_right)[0][:self.num_instances_right[idx]]
                        else: # not greater than self.num_instances_right
                            neg_pairs_right = neg_pair_right
                        


                        # calculate the loss
                        logs = torch.cat((logs,
                            torch.logsumexp(torch.abs(p-neg_pairs_left), dim=0, keepdim=True)))
                            #torch.log(torch.sum(torch.exp(torch.abs(p-neg_pairs_left))))), 0

                        logs = torch.cat((logs,
                            torch.logsumexp(torch.abs(p-neg_pairs_right), dim=0, keepdim=True)))
                            # torch.log(torch.sum(torch.exp(torch.abs(p-neg_pairs_right))))), 0

                    J_ij = torch.sum(logs, 0, keepdim=True) + pos_pair
                    J_ijs = torch.cat((J_ijs , self.max(J_ij).pow(2)), 0)
                    if torch.sum(J_ijs) == 0:
                        import ipdb; ipdb.set_trace()
                if len(J_ijs) == 0:
                    loss += torch.sum(J_ijs)
                else:
                    loss += 1/len(J_ijs) * torch.sum(J_ijs)

        else:
            pass

        if loss.grad_fn is None:
            import ipdb; ipdb.set_trace()
            
        return loss/inputs_batch.shape[0]



class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=40, beta=2, margin=0.5, hard_mining=None,  **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:
                
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
            
                if len(neg_pair) < 1 or len(pos_pair) < 1: # if no such pairs exist
                    c += 1
                    continue 
            
                pos_loss = 2.0/self.beta * torch.log(torch.sum(torch.exp(-self.beta*pos_pair)))
                neg_loss = 2.0/self.alpha * torch.log(torch.sum(torch.exp(self.alpha*neg_pair)))

            else:  
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

                pos_loss = 2.0/self.beta * torch.log(torch.sum(torch.exp(-self.beta*pos_pair)))
                neg_loss = 2.0/self.alpha * torch.log(torch.sum(torch.exp(self.alpha*neg_pair)))

            if len(neg_pair) == 0:
                c += 1
                continue

            loss.append(pos_loss + neg_loss)
        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)

    import numpy as np
    y_ = np.round(np.random.uniform(0,1, (32,32)))
    a, b = np.diag_indices(32)
    y_[a,b] = np.zeros(32)
    y_ = 2*(y_ + y_.transpose())

    assert np.all(y_ == y_.transpose())
    targets = Variable(torch.IntTensor(y_)) 


    print(HierarchicalLiftedStructureLoss(12)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')