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
sys.path.insert(0, './')
sys.path.insert(0, 'utilities/maskrcnn-benchmark/demo')
sys.setrecursionlimit(1000) # same as ipython
# necessary for maskrcnn
import object_detection_modules, processing_modules
from maskrcnn_benchmark.config import cfg


class InputLayer(object):
    def __init__(self, cache_loc='cache/', overwrite=False, max_num_boxes=4):
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
        
        self.max_num_boxes = max_num_boxes
    
    def filter_obj_bboxes(self, bboxes, hand_locations):
        # simply naively calculating the distance to the hand
        assert len(hand_locations) != 0, 'No hands present in this image' 
        rank_batch = []
        for bbox in bboxes:
            bbox_center = (bbox[:2] + bbox[2:])/2.0 
            # calculate the min distance to either hand
            bbox_center = (bbox[2:] + bbox[2:])
            options = []
            if 'left' in hand_locations:

                hand_center = np.array([
                    np.mean([hand_locations['left']['left_x'], hand_locations['left']['right_x']]),
                    np.mean([hand_locations['left']['top_y'], hand_locations['left']['bottom_y']])
                    ])

                sqr_distance= np.dot(hand_center - bbox_center, hand_center - bbox_center)
                options.append(sqr_distance)

            if 'right' in hand_locations:
                hand_center = np.array([
                    np.mean([hand_locations['right']['left_x'], hand_locations['right']['right_x']]),
                    np.mean([hand_locations['right']['top_y'], hand_locations['right']['bottom_y']])
                    ])

                sqr_distance= np.dot(hand_center - bbox_center, hand_center - bbox_center) 
                options.append(sqr_distance)
            
            rank_batch.append([bbox, min(options)]) 
        
        rank_batch_sorted = sorted(rank_batch, key=lambda x: x[1])
        filtered_bboxes = rank_batch_sorted[:min(self.max_num_boxes, len(rank_batch_sorted))]
        assert len(filtered_bboxes)>0
        return [ element[0] for element in filtered_bboxes ]

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
        
        results = []
        # organization of the information
        for idx, image_loc in enumerate(image_locs):
            # first, hand location 
            if 'left' in hand_bounding_boxes[idx]['hand']:
                # bottom left, top right
                left_bbox = np.array([hand_bounding_boxes[idx]['hand']['left']['left_x'],
                                    hand_bounding_boxes[idx]['hand']['left']['bottom_y'],
                                    hand_bounding_boxes[idx]['hand']['left']['right_x'],
                                    hand_bounding_boxes[idx]['hand']['left']['top_y']])
                left_pose = mesh_joints[idx]['left']['joints'].flatten()
            else:
                # 0000
                left_bbox = np.zeros(4)
                left_pose = np.zeros((21, 3)).flatten()

            if 'right' in hand_bounding_boxes[idx]['hand']:
                # bottom left, top right
                right_bbox = np.array([hand_bounding_boxes[idx]['hand']['right']['left_x'],
                                    hand_bounding_boxes[idx]['hand']['right']['bottom_y'],
                                    hand_bounding_boxes[idx]['hand']['right']['right_x'],
                                    hand_bounding_boxes[idx]['hand']['right']['top_y']])
                right_pose = mesh_joints[idx]['right']['joints'].flatten()

            else:
                # 0000   
                right_bbox = np.zeros(4) 
                right_pose = np.zeros((21, 3)).flatten() 
            
            # then object bounding box -- find the ones that are the closest
            object_bounding_boxes = np.zeros(self.max_num_boxes * 4) # intializing 
            if len(hand_bounding_boxes[idx]['hand']) != 0 \
                    and len(obj_bounding_boxes[idx]['bounding_boxes'].numpy())!=0:
                selected_obj_boxes = self.filter_obj_bboxes(obj_bounding_boxes[idx]['bounding_boxes'].numpy(), 
                                                            hand_bounding_boxes[idx]['hand'])
                concat = np.concatenate(selected_obj_boxes)
                object_bounding_boxes[:len(concat)] = concat
            
            # left box, right box, left joints, right joints, object_bounding_box
            results.append(np.concatenate([left_bbox, right_bbox, left_pose, right_pose, object_bounding_boxes]))
        
        # concatenate each of them vertically to make NxT
        results = np.vstack(results).transpose()
        return results 
                
if __name__=='__main__':
    
    dataset_path = '/vision/group/EPIC-KITCHENS/'
    visual_dataset_path = os.path.join(dataset_path, 'EPIC_KITCHENS_2018.Bingbin')
    visual_images_foldername = 'object_detection_images'
    visual_images_folderpath = os.path.join(visual_dataset_path, visual_images_foldername)
   
    images = []
    for element in os.listdir(os.path.join(visual_images_folderpath, 'train')):
        lower_path = os.path.join(os.path.join(visual_images_folderpath, 'train'), element)
        if os.path.isdir(lower_path):
            for element2 in os.listdir(lower_path):
                lowerer_path = os.path.join(lower_path, element2)
                if os.path.isdir(lowerer_path): 
                    for element3 in os.listdir(lowerer_path):
                        lowererer_path = os.path.join(lowerer_path, element3)
                        images.append(lowererer_path)
    IL = InputLayer() 
        
    counter = 0
    batch_size = 10 
    
    while counter < len(images):
        print('processing batch #{}'.format(counter+1))
        IL.get_feature_layer(images[counter:min(counter+batch_size, len(images))])
        print('done')
        counter += batch_size
         
