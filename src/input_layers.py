import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 
import scipy as scp
from scipy.misc import imread
import cv2
import pandas as pd

import sys
sys.path.insert(0, 'utilities/maskrcnn-benchmark/demo')
sys.path.insert(0, '.')
from src import object_detection_modules, processing_modules
# necessary for maskrcnn
from maskrcnn_benchmark.config import cfg


class InputLayer(object):
    def __init__(self, cache_loc='cache/', overwrite=False):
        self.cache_loc = cache_loc
        if not os.path.exists(self.cache_loc):
            os.makedirs(self.cache_loc)
        self.overwrite = overwrite
       
        # from hand features
        self.HPE = processing_modules.HandPositionEstimator(overwrite=self.overwrite)
        self.HD = processing_modules.HandDetector(overwrite=self.overwrite)
        self.HMP = processing_modules.HandMeshPredictor(overwrite=self.overwrite)
        
        # from object 
        config_file = 'utilities/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
        cfg.merge_from_file(config_file)
        #cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
        cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
        self.RP = object_detection_modules.RegionProposer(cfg)         

    def get_feature_layer(self, image_locs):
        for image_loc in image_locs:
            assert os.path.exists(image_loc), '{} does not exist.'.format(image_loc)
        # outputs singlwe vector, concatenation of all features
        
        # get hand bounding box and center point
        images = [[image_loc,imread(image_loc)] for image_loc in image_locs] 
        pose_estimate = self.HPE.process(images)
        
        binary_masks = [(element['image_name'], element['binary_mask'], 
                        element['confidence'], element['original_shape']) 
                        for element in pose_estimate]  
        hand_bounding_boxes = self.HD.process(binary_masks) 
        
        input_mesh = [(element['image_name'], element['hand']) for element in hand_bounding_boxes] 
        mesh_joints = self.HMP.process(input_mesh)
        
        # get hand bounding  
        images_cv = [[image_loc, cv2.imread(image_loc)] for image_loc in image_locs]
        obj_bounding_boxes = self.RP.process(images_cv)

        # organization of the information
        import ipdb; ipdb.set_trace()

if __name__=='__main__':
    
    IL = InputLayer() 
    IL.get_feature_layer(['viz/viz_data/tmp_dataset/P01/P01_01/0000024871.jpg'])
