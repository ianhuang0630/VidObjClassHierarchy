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
import ast

from gt_hierarchy import *
from EK_dataset import DatasetFactory

DEBUG = True

if DEBUG:
    random.seed(7)

def resize_transform(d):
    assert 'frames' in d, 'need "frames" in d'
    new_d = d.copy()
    # TODO
    new_d['frames'] = cv2.resize()
    return new_d

def default_filter_function(d):
    # filters out the ones where there are too few frames present
    return (d['end_frame'] - d['start_frame'])/30 > 10

class EK_Dataset_pretrain(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            filter_function = default_filter_function,
            transform=None):
        # purpose of the dataset object: either for pretraining or for training/testing
        super(EK_Dataset_pretrain, self).__init__()
        self.image_data_folder = image_data_folder
        self.knowns = knowns
        self.unknowns = unknowns
        self.transform = transform
        self.class_key_df = pd.read_csv(class_key_path)
        # TODO: using the key, convert strings into unkowns
        self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
        self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

        self.DF = DatasetFactory(knowns, unknowns,
                    object_data_path, action_data_path, class_key_path)

        self.dataset = self.DF.get_dataset()
        assert 'unknown_frame2bbox' in self.dataset \
                and 'known_frame2bbox' in self.dataset, 'frame2bbox conversion not found'
        self.training_data = self.dataset['known_pretrain']
        for idx, sample in enumerate(self.training_data):
            if not filter_function(sample):
                self.training_data.pop(idx)
        self.f2bbox = self.dataset['known_frame2bbox']

    def __len__(self):
        return len(self.pretrain_knowns)

    def __getitem__(self, idx):
        sample_dict = self.training_data[idx]
        video_id = sample_dict['video_id']
        participant_id = sample_dict['participant_id']
        start_frame = sample_dict['start_frame']
        end_frame = sample_dict['end_frame']

        a = start_frame
        frames = []
        while a < end_frame:
            # loading this frame
            file_path = participant_id + '/' + video_id + '/' + ('0000000000' + str(a))[-10:]+'.jpg'
            image_path = os.path.join(self.image_data_folder, file_path)
            try:
                bboxes = self.f2bbox[participant_id+'/' + video_id+ '/' + str(a)]
            except KeyError:
                a += 30
                # print('skipping frame {} for participant {} video {}'.format(a, participant_id, video_id))
                continue # this would ignore all the cases where the bounding box doesn't exist
            image = cv2.imread(image_path)
            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class']==sample_dict['noun_class']]
            if len(valid_candidates)==0 or valid_candidates[0] == '[]':
                a+=30
                continue
            else:
                this_bbox = np.array(ast.literal_eval(valid_candidates[0]['bbox']))
                # crop gt_bbox
                y, x, yd, xd = this_bbox[0]
                image_black = np.zeros_like(image)
                image_black[y: y+yd , x:x+xd, : ] = image[y:y+yd, x:x+xd, :]
                frames.append(image_black)
            a += 30
        frames = np.stack(frames, axis=3) # T x W x H x C # TODO: reshape needed?
        # get position in the tree
        encoding = get_tree_position(self.noun_dict[sample_dict['noun_class']], self.knowns)
        if encoding is None:
            top_levels = tuple(get_tree_position(self.noun_dict[sample_dict['noun_class']], self.unknowns)[:-1])
            assert top_levels in self.unknown_lowest_level_label
            encoding = np.array(list(top_levels)+[self.unknown_lowest_level_label[top_levels][0]])

        d = {'frames': frames,
             'noun_label': self.noun_dict[sample_dict['noun_class']],
             'hierarchy_encoding': encoding}
        if self.transform is not None:
            d = self.transform(d)
        return d


class EK_Dataset(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            filter_function = default_filter_function,
            transform=None):
        # purpose of the dataset object: either for pretraining or for training/testing
        super(EK_Dataset, self).__init__()

        self.image_data_folder = image_data_folder
        self.knowns = knowns
        self.unknowns = unknowns
        self.transform = transform
        self.class_key_df = pd.read_csv(class_key_path)
        # TODO: using the key, convert strings into unkowns
        self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
        self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

        self.DF = DatasetFactory(knowns, unknowns,
                    object_data_path, action_data_path, class_key_path)

        self.dataset = self.DF.get_dataset()
        assert 'unknown_frame2bbox' in self.dataset \
                and 'known_frame2bbox' in self.dataset, 'frame2bbox conversion not found'
        self.pretrain_knowns = self.dataset['known_pretrain']
        self.training_knowns = self.dataset['known']
        self.training_unknowns = self.dataset['unknown']
        self.known_f2bbox = self.dataset['known_frame2bbox']
        self.unknown_f2bbox = self.dataset['unknown_frame2bbox']

        # merging both of the f2bbox's
        self.f2bbox = {}
        for frame in list(set(self.known_f2bbox.keys()).union(set(self.unknown_f2bbox.keys()))):
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

        # prepare tree
        self.unknown_lowest_level_label = survey_tree(self.knowns)

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
            try:
                bboxes = self.f2bbox[participant_id+'/' + video_id+ '/' + str(a)]
            except KeyError:
                a += 30
                # print('skipping frame {} for participant {} video {}'.format(a, participant_id, video_id))
                continue # this would ignore all the cases where the bounding box doesn't exist
            frames.append(cv2.imread(image_path))
            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class']==sample_dict['noun_class']]
            if len(valid_candidates)==0 or valid_candidates[0] == '[]':
                gt_bbox.append(np.array([0,0,0,0]))
            else:
                # parse valid_candidates into array
                # TODO
                gt_bbox.append(np.array(ast.literal_eval(valid_candidates[0]['bbox'])))
            a += 30
        frames = np.stack(frames, axis=3) # T x W x H x C
        gt_bbox = np.stack(frames, axis=1)  # T x 4
        # get position in the tree
        encoding = get_tree_position(self.noun_dict[sample_dict['noun_class']], self.knowns)
        if encoding is None:
            top_levels = tuple(get_tree_position(self.noun_dict[sample_dict['noun_class']], self.unknowns)[:-1])
            assert top_levels in self.unknown_lowest_level_label
            encoding = np.array(list(top_levels)+[self.unknown_lowest_level_label[top_levels][0]])

        d = {'frames': frames, 'bboxes': gt_bbox,
             'noun_label': self.noun_dict[sample_dict['noun_class']],
             'hierarchy_encoding': encoding}
        if self.transform is not None:
            d = self.transform(d)
        return d


if __name__=='__main__':

    dataset_path = '/vision/group/EPIC-KITCHENS/'
    annotations_foldername = 'annotations'
    annotations_folderpath = os.path.join(dataset_path, annotations_foldername)
    visual_dataset_path = os.path.join(dataset_path, 'EPIC_KITCHENS_2018.Bingbin')
    visual_images_foldername = 'object_detection_images'
    visual_images_folderpath = os.path.join(visual_dataset_path, visual_images_foldername)

    # training data
    training_action_labels = 'EPIC_train_action_labels.csv'
    training_object_labels = 'EPIC_train_object_labels.csv'
    train_action_csvpath = os.path.join(annotations_folderpath, training_action_labels)
    train_object_csvpath = os.path.join(annotations_folderpath, training_object_labels)

    class_key = 'EPIC_noun_classes.csv'
    class_key_csvpath = os.path.join(annotations_folderpath, class_key)

    assert os.path.exists(train_action_csvpath), "{} does not exist".format(train_action_csvpath)
    assert os.path.exists(train_object_csvpath), "{} does not exist".format(train_object_csvpath)
    assert os.path.exists(visual_images_folderpath), "{} does not exist".format(visual_images_folderpath)
    assert os.path.exists(class_key_csvpath), "{} does not exist".format(class_key_csvpath)

    image_data_folder = os.path.join(visual_images_folderpath, 'train')
    knowns = ['pan', 'onion']
    unknowns = ['plate', 'meat']
    DF = EK_Dataset(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder)
    # print(DF[2])

    DF_pretrain = EK_Dataset_pretrain(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder)
    print(DF_pretrain[30])
