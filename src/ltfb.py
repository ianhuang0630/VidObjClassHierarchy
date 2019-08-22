"""
Pure-pytorch implementation of Long-Term Feature Banks for Detailed Video Understanding
https://arxiv.org/pdf/1812.05038.pdf

Intended here to implement self-attention
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.normalization import LayerNorm
from torch.nn.functional import max_pool3d
from torchvision.ops import roi_align

class LongTermFeatureBank(nn.Module):
    def __init__(self, input_shape, embedding_dim, 
                    classifier_output_layers = [200, 50, 10],
                    num_stacks=2, precomputed=True):
        super(LongTermFeatureBank, self).__init__()

        self.input_shape = tuple(np.array(input_shape)[[1,0,2,3]]) # time, num_channels, height, width
        self.num_stacks = num_stacks

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.layernorm = LayerNorm(self.input_shape[:2])
        self.dropout=nn.Dropout(p=0.2)
    
        self.attention_layers = self.make_attention_block(self.num_stacks)

        self.precomputed = precomputed
        self.embedding_dim = embedding_dim

        self.classifier = nn.Sequential(* self.make_classifier(16*512*2, classifier_output_layers + [embedding_dim]))

    def make_attention_block(self, num_stacks):

        layers_per_stack = []

        for i in range(num_stacks):

            fc_a = nn.Linear(self.input_shape[1], self.input_shape[1])
            fc_b = nn.Linear(self.input_shape[1], self.input_shape[1])
            fc_c = nn.Linear(self.input_shape[1], self.input_shape[1])
            fc_1 = nn.Linear(self.input_shape[1], self.input_shape[1])

            layers_per_stack.append(fc_a)
            layers_per_stack.append(fc_b)
            layers_per_stack.append(fc_c)
            layers_per_stack.append(fc_1)

        return nn.ModuleList(layers_per_stack)

    def make_classifier(self, input_dimension, output_shapes):
        # input_dimension would be BxNx512
        layers = []
        input_shapes = [input_dimension]

        for idx, output_dim in enumerate(output_shapes):
            # create fully connected
            layers.append(nn.Linear(input_shapes[idx], output_dim))
            layers.append(nn.ReLU())
            input_shapes.append(output_dim)

        layers = layers[:-1] # cutting out the last relu
        return layers

    def attention_forward(self, x, L, layer_num):
        assert x.shape == L.shape
        # reshaping x, squashing into BN x 152
        x_original_shape = x.shape
        x = x.reshape(x_original_shape[0]*x_original_shape[1], x_original_shape[2])
        L = L.reshape(x_original_shape[0]*x_original_shape[1], x_original_shape[2])

        # assuming that x is [N x d]
        # A1 = Linear transform 1 on x
        a = self.attention_layers[layer_num*4+0](x)
        # A2 = Linear transform 2 on L
        b = self.attention_layers[layer_num*4+1](L)

        # reshaping a and b 
        a = a.reshape(x_original_shape)
        b = b.reshape(x_original_shape)


        # A4 = softmax(A1.dot(A2) * 1/sqrt(num_dimensions) )
        ab = self.softmax(torch.matmul(a,b.transpose(1,2)) * 1/np.sqrt(self.input_shape[1]))
        # A3 = Linear transform 3 on L
        c = self.attention_layers[layer_num*4+2](L)
        
        # reshaping x back
        x = x.reshape(x_original_shape)
        # reshaping c
        c = c.reshape(x_original_shape)

        # return x + Dropout(Linear(relu(layer_normalization(softmax(A4.dot(A3))))))
        e = self.relu(self.layernorm(torch.matmul(ab, a)))
        d = x + self.dropout(self.attention_layers[layer_num*4+3](e.reshape(x_original_shape[0]*x_original_shape[1], x_original_shape[2]))).reshape(x_original_shape)

        return d

    def forward(self, x):
        x = x.transpose(2,1)
        if self.precomputed:
            x = max_pool3d(x, kernel_size =(1,7,7)) # should give Tx512x1x1
            x = x.squeeze()
            x_original_shape = x.shape
        else:
            raise ValueError('Not implemented.')

        new_x = x.clone()
        for i in range(self.num_stacks):
            new_x = self.attention_forward(new_x, x, i)

        # original x and d into classifier.
        # need to concatenate and flatten first
        
        out_embedding = self.classifier(torch.cat((x,new_x), dim=1).reshape(x_original_shape[0], -1))

        return out_embedding

if __name__ == '__main__':

    B = torch.randn(512, 16, 7, 7)
    FBO = FeatureBankOperator(A.shape)

    print(FBO(A))
    