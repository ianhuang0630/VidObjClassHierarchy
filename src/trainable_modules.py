"""
Implementations of the models that predicts placement of a new object in a video
sequence
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 
import scipy as scp
import cv2
import pandas as pd

class VanillaModel(nn.Module):
    def __init__(self):
        super(VanillaModel, self).__init__()
    
    def forward(self, x):
        pass


class BaselineVanillaModel(nn.Module):
    def __init__(self, feature_input_shape, image_input_shape,
                num_fcs = 3, fc_output_shapes = [72, 36, 18],
                output_shape=16, time_conv_filter_width=2
                feature_conv_filter_width=4, time_stride=1, feature_stride=2):

        super(BaselineVanillaModel, self).__init__()
        self.feature_input_shape = feature_input_shape
        self.image_input_shape = image_input_shape
        self.num_fcs = num_fcs
        self.fc_output_shapes = fc_output_shapes

        self.output_shape = output_shape # 
            
        # fully connected layer for hand/object feature input
        fc_layers = self.make_fcs(self.num_fcs, self.feature_input_shape, 
                self.fc_output_shapes)
        
        self.feature_fc = nn.Sequential(*fc_layers)
        
        # convolutional layers for hand/object feature input

        # convolution layers for image input

        # concatenation of tensors and flattening by mean

        # final head, composed of multiple fully connected layers.
        # loss will be multi-level loss, so loss of different dimensions are
        # weighted differently. Predict each of these sections separately

    def make_fcs(self, num_layers, input_shape, output_shapes):
        layers = []
        input_shapes = [input_shape] 
        assert len(output_shapes) == num_layers

        for i in range(num_layers):
            layers.append(nn.Linear(input_shapes[-1], output_shapes[i])) # linear unit
            layers.append(nn.ReLu()) # nonlinearity
            input_shapes = output_shapes[i]
        
        return layers

    def get_flat_fts(self, in_shape, fts):
        f = fts(Variable(torch.ones(1,*in_shape)))
        return int(np.prod(f.size()[1:]))
        
    def forward(self, x):
        # [feature] -> linear transform -> conv(t stride 1) -> [D1xT representaiton]
        # [image] -> frame by frame conv 2d -> flatten accross time -> conv (t stride 1) -> [D2xT] representation 
        # stack [D1xT] and [D2xT] concatenation --> conv2d -> flatten -> fully connected
        feature_embedding = self.feature_conv_layer(x['precomputed_features'])
        image_embedding = self.image_conv_layer(x['image']) 
        
        # stacking feature_embedding and image_embedding

        pass

class ThreeDConv(nn.Module):
    def __init__(self):
        super(ThreeDConv, self).__init__()
    
    def forward(self, x):
        pass

class TreeEncoder(nn.Module):
    def __init__(self):
        super(TreeEncoder, self).__init__()
    
    def foward(self, x): 
        pass

if __name__=='__main__':
    BVM = BaselineVanillaModel(108, (23, 23, 108), )
