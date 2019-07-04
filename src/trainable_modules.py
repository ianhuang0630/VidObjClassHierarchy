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

from input_layers import InputLayer 

class VanillaModel(nn.Module):
    def __init__(self):
        super(VanillaModel, self).__init__()
    
    def forward(self, x):
        pass


class BaselineVanillaModel(nn.Module):
    def __init__(self, feature_input_shape, image_input_shape,
                num_fcs = 3, fc_output_shapes = [72, 36, 18],
                output_shape=16, time_conv_filter_width=2,
                conv_kernel_time_size = 3, conv_kernel_feat_size=3,
                feature_conv_filter_width=4, time_stride=1, feature_stride=2):

        super(BaselineVanillaModel, self).__init__()
        self.feature_input_shape = feature_input_shape
        self.image_input_shape = image_input_shape
        # for fully connected layers
        self.num_fcs = num_fcs
        self.fc_output_shapes = fc_output_shapes
        # for convolutions on matrix
        self.feature_stride = feature_stride
        self.time_stride = time_stride
        self.conv_kernel_time_size = conv_kernel_time_size
        self.conv_kernel_feat_size = conv_kernel_feat_size
       
        assert self.conv_kernel_time_size % 2 == 1, 'conv_kernel_time_size needs to be odd'
        assert self.conv_kernel_feat_size % 2 == 1, 'conv_kernel_feat_size needs to be odd'
        self.time_padding = int((self.conv_kernel_time_size-1)/2)
        self.feat_padding = int((self.conv_kernel_feat_size-1)/2)
        self.output_shape = output_shape # 
        
        # fully connected layer for hand/object feature input
        fc_layers = self.make_feature_fcs(self.num_fcs, self.feature_input_shape, 
                self.fc_output_shapes)
        self.feature_fc = nn.Sequential(*fc_layers)
         
        # convolutional layers for hand/object feature input
        # input should now be NxT
        conv_layers = self.make_feature_convs()
        self.feature_conv = nn.Sequential(*conv_layers)
         
        # convolution layers for image input
        image_conv_layers = self.make_image_3dconv()
        self.image_conv = nn.Sequential(*image_conv_layers)
        # output should be NxT

        # concatenation of tensors and flattening by mean
        
        # final head, composed of multiple fully connected layers.
        # loss will be multi-level loss, so loss of different dimensions are
        # weighted differently. Predict each of these sections separately
    
    def make_feature_convs(self):
        # NOTE input would have to be (batch_size, 1, num_features, num_timesteps)
        layers = []

        layers.append(nn.Conv2d(in_channels=1, out_channels=64, 
                kernel_size=(self.conv_kernel_feat_size, self.conv_kernel_time_size), 
                stride=(self.feature_stride, self.time_stride), 
                padding=(self.feat_padding, self.time_padding)))
        layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(in_channels=64, out_channels=1,
                kernel_size=(self.conv_kernel_feat_size, self.conv_kernel_time_size), 
                stride=(self.feature_stride, self.time_stride), 
                padding=(self.feat_padding, self.time_padding)))
        layers.append(nn.ReLU()) 

        return layers

    def make_feature_fcs(self, num_layers, input_shape, output_shapes):
        layers = []
        input_shapes = [input_shape] 
        assert len(output_shapes) == num_layers

        for i in range(num_layers):
            layers.append(nn.Linear(input_shapes[-1], output_shapes[i])) # linear unit
            layers.append(nn.ReLU()) # nonlinearity
            input_shapes.append(output_shapes[i])
        
        return layers
    
    def make_image_3dconv(self):
        layers = []
        layers.append(nn.Conv3d(3, 64, kernel_size=(3,3,3), stride=(2,2,1), padding=(4,2,1)))
        layers.append(nn.ReLU())
        return layers

    def get_flat_fts(self, in_shape, fts):
        f = fts(Variable(torch.ones(1,*in_shape)))
        return int(np.prod(f.size()[1:]))
        
    def forward(self, x):
        # [feature] -> linear transform -> conv(t stride 1) -> [D1xT representaiton]
        # [image] -> frame by frame conv 2d -> flatten accross time -> conv (t stride 1) -> [D2xT] representation 
        # stack [D1xT] and [D2xT] concatenation --> conv2d -> flatten -> fully connected
        precomp_feats = x['precomputed_features']
        image = x['image'] 
        
        # stacking feature_embedding and image_embedding
        
        # extracting feature information
        feat_fc_out = self.feature_fc(precomp_feats.t()) 
        feat_conv_out = self.feature_conv(feat_fc_out.t())
         
        # extracting image information
        image_conv_out = self.image_conv(image)

        import ipdb; ipdb.set_trace() 


if __name__=='__main__':
    # loading and preparing tensor
    dataset_path = '/vision/group/EPIC-KITCHENS/'
    visual_dataset_path = os.path.join(dataset_path, 'EPIC_KITCHENS_2018.Bingbin')
    visual_images_foldername = 'object_detection_images'
    visual_images_folderpath = os.path.join(visual_dataset_path, visual_images_foldername)
    
    # TODO: select starting time
    # TODO: select_ending_time
    
    images = []
    for element in ['P04/P04_03/0000007471.jpg', 'P04/P04_03/0000007501.jpg']:
        image_path = os.path.join(os.path.join(visual_images_folderpath, 'train'), element) 
        assert os.path.exists(image_path), '{} does not exit'.format(image_path)
        images.append(image_path)
     
    IL = InputLayer()
    feats = IL.get_feature_layer(images)
    
    image_mats = np.array([cv2.imread(image_loc).tolist() for image_loc in images])
     

    x = {'precomputed_features': torch.Tensor(feats), 'image': torch.Tensor(image_mats)}
    
    import ipdb; ipdb.set_trace()

    BVM = BaselineVanillaModel(x['precomputed_features'].shape[0], x['image'].shape, )
    output = BVM(x)
    print(output)
