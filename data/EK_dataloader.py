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
import pickle
from torch.utils import data

try:
    from gt_hierarchy import *
    from EK_dataset import DatasetFactory
except: 
    from data.gt_hierarchy import *
    from data.EK_dataset import DatasetFactory

DEBUG = True

if DEBUG:
    random.seed(7)
    np.random.seed(7)


def default_filter_function(d):
    # filters out the ones where there are too few frames present
    return (d['end_frame'] - d['start_frame'])/30 > 10

def create_config_file(threshold, processed_frame_number, cache_dir='dataloader_cache/blackout_crop'):
    # TODO import things: threshold, 
    config = {'threshold': threshold, 'processed_frame_number': processed_frame_number}
    with open (os.path.join(cache_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

def blackout_crop_wrapper(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                            cache_dir='dataloader_cache/blackout_crop'):
    
    if os.path.exists(os.path.join(cache_dir, 'config.json')):
        with open (os.path.join(cache_dir, 'config.json'), 'r') as f:
            prev_config = json.load(f)
        overwrite = not (prev_config == {'threshold': threshold, 'processed_frame_number': processed_frame_number})
    else:
        overwrite = True

    return blackout_crop(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=threshold, 
                    cache_dir=cache_dir, overwrite=overwrite)

def blackout_crop(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                cache_dir='dataloader_cache/backout_crop/', overwrite=False, scale = 0.5):

    video_id = sample_dict['video_id']
    participant_id = sample_dict['participant_id']
    start_frame = sample_dict['start_frame']
    end_frame = sample_dict['end_frame']

    # TODO: checking cache and config files
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    cache_filename = 'v#{}p#{}s#{}e#{}.npy'.format(video_id, participant_id, start_frame, end_frame)

    if os.path.exists(os.path.join(cache_dir, cache_filename)) and not overwrite:
        frames = np.load(os.path.join(cache_dir, cache_filename))

    else: 

        a = start_frame
        frames = []
        if ((end_frame-start_frame)/30)/processed_frame_number > threshold:
            skip_interval = np.floor(((end_frame-start_frame)/30)/processed_frame_number)
            skip_interval = int(skip_interval)
        else:
            skip_interval = 1

        while a < end_frame:
            # loading this frame
            file_path = participant_id + '/' + video_id + '/' + ('0000000000' + str(a))[-10:]+'.jpg'
            image_path = os.path.join(image_data_folder, file_path)
            try:
                bboxes = f2bbox[participant_id+'/' + video_id+ '/' + str(a)]
            except KeyError:
                a += 30 * skip_interval
                print('skipping frame {} for participant {} video {}'.format(a, participant_id, video_id))
                continue # this would ignorein all the cases where the bounding box doesn't exist
            image = cv2.imread(image_path)
            # resizing the image
            image = cv2.resize(image, tuple([int(dim*scale) for dim in image.shape][:2][::-1]))

            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class']==sample_dict['noun_class']]
            if len(valid_candidates)==0 or valid_candidates[0] == '[]':
                a+=30 * skip_interval
                continue
            else:
                this_bbox = np.array(ast.literal_eval(valid_candidates[0]['bbox']))
                # crop gt_bbox
                if len(this_bbox) == 0:
                    a += 30 * skip_interval
                    continue
                y, x, yd, xd = this_bbox[0]
                y, x, yd, xd = int(y*scale), int(x*scale), int(yd*scale), int(xd*scale)
                
                image_black = np.zeros_like(image)
                image_black[y: y+yd , x:x+xd, : ] = image[y:y+yd, x:x+xd, :]
                frames.append(image_black)
            a += 30 * skip_interval
        frames = np.stack(frames, axis=3) # T x W x H x C # TODO: reshape needed?

        # TODO: save cache
        np.save(os.path.join(cache_dir, cache_filename), frames)
        create_config_file(threshold, processed_frame_number, cache_dir=cache_dir)

    return frames


class EK_Dataset_pretrain_pairwise(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            num_samples=10000,
            filter_function = default_filter_function,
            processed_frame_number = 20, 
            individual_transform=None,
            pairwise_transform=None):
        # purpose of the dataset object: either for pretraining or for training/testing
        super(EK_Dataset_pretrain_pairwise, self).__init__()
        self.image_data_folder = image_data_folder
        self.knowns = knowns
        self.unknowns = unknowns
        self.individual_transform = individual_transform
        self.pairwise_transform = pairwise_transform
        self.num_samples = num_samples
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
        buffer_ = []
        for idx, sample in enumerate(self.training_data):
            if filter_function(sample):
                buffer_.append(sample)
        self.training_data = buffer_
        del buffer_
        self.f2bbox = self.dataset['known_frame2bbox']
        
        # used to handle cases when the clip is especially long
        self.processed_frame_number = processed_frame_number

        # naive sampling of pairwise
        self.rand_selection_indices = []
        for i in range(num_samples):
            selection_indices = list(np.random.choice(len(self.training_data), 2))
            self.rand_selection_indices.append(selection_indices)
    
    def __len__(self):
        return len(self.rand_selection_indices)

    def __getitem__(self, idx):
        sample_a = self.training_data[self.rand_selection_indices[idx][0]]
        sample_b = self.training_data[self.rand_selection_indices[idx][1]]

        indiv_output_d = []
        for sample_dict in [sample_a, sample_b]:
            video_id = sample_dict['video_id']
            participant_id = sample_dict['participant_id']
            start_frame = sample_dict['start_frame']
            end_frame = sample_dict['end_frame']

            frames = blackout_crop_wrapper(sample_dict, self.processed_frame_number, self.f2bbox, self.image_data_folder, threshold=2, 
                                cache_dir='dataloader_cache/blackout_crop')

            # get position in the tree
            encoding = get_tree_position(self.noun_dict[sample_dict['noun_class']], self.knowns)
            if encoding is None:
                top_levels = tuple(get_tree_position(self.noun_dict[sample_dict['noun_class']], self.unknowns)[:-1])
                assert top_levels in self.unknown_lowest_level_label
                encoding = np.array(list(top_levels)+[self.unknown_lowest_level_label[top_levels][0]])

            d = {'frames': frames,
                 'noun_label': self.noun_dict[sample_dict['noun_class']],
                 'hierarchy_encoding': encoding}
            if self.individual_transform is not None:
                d = self.individual_transform(d)
            indiv_output_d.append(d)
        # get distance bretween the two noun classes
        pairwise_tree_dist = get_tree_distance(indiv_output_d[0]['noun_label'],
                                            indiv_output_d[1]['noun_label'])

        output = {'frames_a': indiv_output_d[0]['frames'],
                    'frames_b':indiv_output_d[1]['frames'],
                    'noun_label_a': indiv_output_d[0]['noun_label'],
                    'noun_label_b': indiv_output_d[1]['noun_label'],
                    'dist': pairwise_tree_dist}

        if self.pairwise_transform is not None:
            output = self.pairwise_transform(output)
            
        return output

class EK_Dataset_pretrain(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            processed_frame_number = 20,
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
        buffer_ = []
        for idx, sample in enumerate(self.training_data):
            if filter_function(sample):
                buffer_.append(sample)
        self.training_data = buffer_
        del buffer_
        self.f2bbox = self.dataset['known_frame2bbox']

        # used to handle cases when the clip is espeically long
        self.processed_frame_number = processed_frame_number

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        sample_dict = self.training_data[idx]
        video_id = sample_dict['video_id']
        participant_id = sample_dict['participant_id']
        start_frame = sample_dict['start_frame']
        end_frame = sample_dict['end_frame']
        
        frames = blackout_crop_wrapper(sample_dict, self.processed_frame_number, self.f2bbox, self.image_data_folder, threshold=2, 
                                cache_dir='dataloader_cache/blackout_crop')

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
        buffer_ = []
        for idx, sample in enumerate(self.training_data):
            if filter_function(sample):
                buffer_.append(sample)
        self.training_data = buffer_
        del buffer_
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
                # gt_bbox.append(np.array([0,0,0,0]))
                a+= 30
                continue # ignoring all cases where bounding boxes are empty
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
    
    # DF = EK_Dataset(knowns, unknowns,
    #         train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder)
    # print(DF[2])
    
    with open('current_split.pkl','rb') as f:
        split = pickle.load(f)
    knowns = split['training_known']
    unknowns = split['training_unknown']
    
    DF_pretrain = EK_Dataset_pretrain_pairwise(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder)
    import ipdb; ipdb.set_trace()
    print(DF_pretrain[14])
    # for i in range(4):
    #     import ipdb; ipdb.set_trace()
    #     print(DF_pretrain[112+i])
    
    
