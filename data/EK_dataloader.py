"""
This implements the dataset object for the Pytorch dataloader for training
"""

import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
import cv2
import os

DEBUG = True

if DEBUG:
    random.seed(7)

def default_filter_function(d):
    # filters out the ones where there are too few frames present
    return (d['end_frame'] - d['start_frame'])/30 > 10
    
class EK_Dataset(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            image_data_folder,
            filter_function = default_filter_function,
            transform=None):
        
        self.image_data_folder = image_data_folder

        super(EK_Dataset, self).__init__()
        self.DF = DatasetFactory(knowns, unknowns, 
                    object_data_path, action_data_path, class_key_path)        
        
        self.dataset = self.DF.get_dataset
        assert 'unknown_frame2bbox' in self.dataset \
                and 'known_frame2bbox' in self.dataset, 'frame2bbox conversion not found'
        self.pretrain_knowns = self.dataset['known_pretrain']
        self.training_knowns = self.dataset['known']
        self.training_unknowns = self.dataset['unknown']
        self.known_f2bbox = self.dataset['known_f2bbox']
        self.unknown_f2bbox = self.dataset['unknown_f2bbox']
        # merging both of the f2bbox's
        self.f2bbox = {}
        for frame in list(set(self.known_f2bbox.keys()) + set(self.unknown_f2bbox.keys())):
            for_this_frame = []
            if frame in self.known_f2bbox:
                for_this_frame.extend(self.known_f2bbox[frame])
            if frame in self.unknown_f2bbox:
                for_this_frame.extend(self.unknown_f2bbox[frame])
            self.f2bbox[frame] = for_this_frame

        self.training_data = self.training_knowns + self.training_unknowns
        for idx, sample in enumerate(self.training_data):
            if not filter_function(sample):
                self.training_data.pop(idx)
        random.shuffle(self.training_data) 

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        sample_dict = self.training_data[idx]
        video_id = sample_dict['video_id']
        participant_id = sample_dict['participant_id']
        start_frame = sample_dict['start_frame']
        end_frame = sample_dict['end_frame']
        
        a = start_frame
        frames = []
        gt_bbox = []
        while a < end_frame:
            # loading this frame
            file_path = participant_id + '/' + video_id + '/' + ('0000000000' + str(a))[-10:]+'.jpg'
            image_path = os.path.join(self.image_data_folder, file_path)
            import ipdb; ipdb.set_trace()
            frames.append(cv2.imread(image_path))
            bboxes = self.f2bbox[a]
            valid_candidates = [bbox for bbox in bboxes if bbox[0]==sample_dict['noun_class']]
            if len(valid_candidates)==0 or valid_candidates[0] == '[]':
                gt_bbox.append(np.array([0,0,0,0]))
            else:
                # parse valid_candidates into array
                # TODO
                import ipdb; ipdb.set_trace() 
                gt_bbox.append(np.array([0,0,0,0]))
            
            a += 30
        frames = np.stack(frames, axis=3)
        
        # TODO: load bounding boxes
             
        # raw_image
        return {'frames': frames}
        


if __name__=='__main__':
    pass