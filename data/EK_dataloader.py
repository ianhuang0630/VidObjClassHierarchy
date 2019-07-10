"""
This implements the dataset object for the Pytorch dataloader for training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os


class EK_Dataset(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            
            transform=None):
        super(EK_Dataset, self).__init__()
        self.DF = DatasetFactory(knowns, unknowns, 
                    object_data_path, action_data_path, class_key_path)        
        assert type(self.DF.dataset) is dict
        # NOTE: assuming the dataset is not skewed between knowns and unknowns
        # g is pretrained on subset of the instances of knowns 
        # f is trained on other subset of knowns, and on the whole set of 
        # unknowns
        
    def __len__(self):
        pass

    def __getitem__(self, idx):
        # returns dictionary with clip and labels (class position + bounding box)

if __name__=='__main__':
    pass
