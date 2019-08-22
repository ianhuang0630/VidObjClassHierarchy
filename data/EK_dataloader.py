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
    from sampler import Selector
except: 
    from data.gt_hierarchy import *
    from data.EK_dataset import DatasetFactory
    from data.sampler import Selector

DEBUG = False

if DEBUG:
    random.seed(7)
    np.random.seed(7)


def default_filter_function(d):
    # filters out the ones where there are too few frames present
    return (d['end_frame'] - d['start_frame'])/30 > 10

def create_config_file(threshold, processed_frame_number, scaling=0.5, cache_dir='dataloader_cache/blackout_crop'):
    # TODO import things: threshold, 
    config = {'threshold': threshold, 'processed_frame_number': processed_frame_number, 
            'scaling': scaling}
    with open (os.path.join(cache_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

def crop_wrapper(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, scaling=0.5,
                            cache_dir='dataloader_cache/blackout_crop', mode='blackout'):
    
    if os.path.exists(os.path.join(cache_dir, 'config.json')):
        with open (os.path.join(cache_dir, 'config.json'), 'r') as f:
            prev_config = json.load(f)
        overwrite = not (prev_config == {'threshold': threshold, 'processed_frame_number': processed_frame_number,
                                        'scaling': scaling})
    else:
        overwrite = True
    if mode == 'blackout':
        return blackout_crop(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=threshold, 
                    cache_dir=cache_dir, overwrite=overwrite, scale=scaling)
    else:
        return rescaling_crop(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=threshold, 
                    cache_dir=cache_dir, overwrite=overwrite, scale=scaling)

def rescaling_crop(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2,
            cache_dir='dataloader_cache/rescaling_crop', overwrite=False, scale=0.5):
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

                rescaled_crop = cv2.resize(image[y:y+yd, x:x+xd, :], 
                                    tuple([int(dim*scale) for dim in image.shape[:2][::-1]]))
                frames.append(rescaled_crop)

            a += 30 * skip_interval
        frames = np.stack(frames, axis=3) # T x W x H x C # TODO: reshape needed?

        # TODO: save cache
        np.save(os.path.join(cache_dir, cache_filename), frames)
        create_config_file(threshold, processed_frame_number, cache_dir=cache_dir)
    return frames
    


def blackout_crop(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                cache_dir='dataloader_cache/blackout_crop/', overwrite=False, scale=0.5):

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

class EK_Dataset_pretrain_batchwise(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            model_saveloc,
            batch_size = 8,
            training_num_samples=10000,
            validation_num_samples=200,
            filter_function = default_filter_function,
            processed_frame_number = 20, 
            individual_transform=None,
            batchwise_transform=None,
            mode='resnet', output_cache_folder='dataloader_cache/', 
            snip_threshold=32,
            crop_type='rescale',
            sampling_mode='equality',
            selector_train_ratio=0.75):
        super(EK_Dataset_pretrain_batchwise, self).__init__()
        self.image_data_folder = image_data_folder 
        self.knowns = knowns
        self.unknowns = unknowns
        self.individual_transform = individual_transform
        self.batchwise_transform = batchwise_transform
        self.train_num_samples = training_num_samples
        self.val_num_samples = validation_num_samples

        self.class_key_df = pd.read_csv(class_key_path)

        self.mode = mode
        assert self.mode == 'resnet' or self.mode == 'noresnet'
        self.output_cache_folder = output_cache_folder
        if not os.path.exists(self.output_cache_folder):
            os.makedirs(self.output_cache_folder, exist_ok = True)
        self.snip_threshold = snip_threshold
        self.crop_type = crop_type
        assert crop_type == 'blackout' or crop_type == 'rescale', 'crop_type must either be blackout or rescale'
        # TODO: using the key, convert strings into unkowns
        self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
        self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

        self.output_cache_fullpath = os.path.join(os.path.join(self.output_cache_folder, self.mode + '_out'), self.crop_type)


        self.DF = DatasetFactory(knowns, unknowns,
                    object_data_path, action_data_path, class_key_path)
        self.dataset = self.DF.get_dataset()

        assert 'unknown_frame2bbox' in self.dataset \
                and 'known_frame2bbox' in self.dataset, 'frame2bbox conversion not found'
        self.training_data = self.dataset['known_pretrain']
        self.unknown_lowest_level_label = survey_tree(self.knowns)

        self.snip_clips()

        buffer_ = []
        for idx, sample in enumerate(self.training_data):
            if filter_function(sample):
                buffer_.append(sample)
        self.training_data = buffer_
        del buffer_

        self.f2bbox = self.dataset['known_frame2bbox']
        self.processed_frame_number = processed_frame_number

        # calculating tree distance matrix
        self.train_gt_dists = np.zeros((len(self.training_data), len(self.training_data)))
        for idx1, nc1 in enumerate([self.noun_dict[element['noun_class']] for element in self.training_data]):
            for idx2, nc2 in enumerate([self.noun_dict[element['noun_class']] for element in self.training_data]):
                self.train_gt_dists [idx1, idx2] = get_tree_distance(nc1, nc2)

        # intializing the first batch
        self.batch_size = batch_size
        self.num_batches = 5000
        selector = Selector(self.training_data, option='fullyconnected', train_ratio=selector_train_ratio)
        self.rand_selection_indices = selector.get_minibatch_indices('train', self.batch_size, self.num_batches)
        self.val_indices = selector.get_minibatch_indices('val', self.batch_size, 40)
    
    @staticmethod
    def process(data, output_cache_fullpath, crop_type, processed_frame_number,f2bbox, 
                image_data_folder, noun_dict, knowns, unknowns, unknown_lowest_level_label, 
                individual_transform, batchwise_transform, overwrite=False):
        indiv_output_d = []
        for sample_dict in data:
            video_id = sample_dict['video_id']
            participant_id = sample_dict['participant_id']
            start_frame = sample_dict['start_frame']
            end_frame = sample_dict['end_frame']

            # TODO: make cache name, search for cache file, if found load.
            cache_filename = 'v#{}p#{}s#{}e#{}.pkl'.format(video_id, participant_id, start_frame, end_frame)
            if os.path.exists(os.path.join(output_cache_fullpath, cache_filename)) and not overwrite:
                # loding d
                with open(os.path.join(output_cache_fullpath, cache_filename), 'rb') as f:
                    d = pickle.load(f)
                indiv_output_d.append(d)

            else:

                if crop_type == 'blackout':
                    frames = crop_wrapper(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                                    scaling = 0.5, cache_dir='dataloader_cache/blackout_crop', mode=crop_type)
                else:

                    frames = crop_wrapper(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                                    scaling = 0.5, cache_dir='dataloader_cache/rescale_crop', mode=crop_type)

                # get position in the tree
                encoding = get_tree_position(noun_dict[sample_dict['noun_class']], knowns)
                if encoding is None:
                    top_levels = tuple(get_tree_position(noun_dict[sample_dict['noun_class']], unknowns)[:-1])
                    assert top_levels in unknown_lowest_level_label
                    encoding = np.array(list(top_levels)+[unknown_lowest_level_label[top_levels][0]])

                d = {'frames': frames,
                     'noun_label': noun_dict[sample_dict['noun_class']],
                     'hierarchy_encoding': encoding}

                if individual_transform is not None:
                    d = individual_transform(d)

                # saving into cache file
                with open(os.path.join(output_cache_fullpath, cache_filename), 'wb') as f:
                    pickle.dump(d, f)

                indiv_output_d.append(d)
                
        # get distance between all samples in this batch
        gt_tree_dist = np.zeros((len(indiv_output_d), len(indiv_output_d)))
        for idx1, nc1 in enumerate([element['noun_label'] for element in indiv_output_d]):
            for idx2, nc2 in enumerate([element['noun_label'] for element in indiv_output_d]):
                gt_tree_dist [idx1, idx2] = get_tree_distance(nc1, nc2)

        output = {'batch_frames': [clip['frames'] for clip in indiv_output_d],
                    'noun_labels': [clip['noun_label'] for clip in indiv_output_d],
                    'dist_matrix': gt_tree_dist}

        if batchwise_transform is not None:
            output = batchwise_transform(output)
        
        return output

    def get_val_dataset(self):
        val_set = []
        counter = 0

        for batch in self.val_indices:
            val_set.append(EK_Dataset_pretrain_batchwise.process(
                                                [self.training_data[element] for element in batch],
                                                self.output_cache_fullpath, self.crop_type, 
                                                self.processed_frame_number, self.f2bbox, 
                                                self.image_data_folder, self.noun_dict, self.knowns, 
                                                self.unknowns, self.unknown_lowest_level_label, 
                                                self.individual_transform, self.batchwise_transform))
            # val_set.append(sample_batch)

        return val_set

    def snip_clips(self): 
        snip_clips = []
        for clip in self.training_data:
            if int((clip['end_frame']-clip['start_frame'])/30) > self.snip_threshold:
                a = clip['start_frame']
                while a + (self.snip_threshold-1)*30 <= clip['end_frame']:
                    b = a + (self.snip_threshold-1)*30
                    # add the clip 
                    snip_clip = clip.copy()
                    snip_clip['start_frame'] = a
                    snip_clip['end_frame'] = b
                    snip_clips.append(snip_clip)
                    # update the starting point
                    a = b + 30
            else:
                snip_clips.append(clip)

        self.training_data = snip_clips
        del snip_clips

        

    def __len__(self):
        return len(self.rand_selection_indices)

    # def initialize_batch(self, batch_size, negative_levels = [2,4]):
    #     assert batch_size%2 == 0, 'expected even number for batch_size'
    #     row, col = np.where(self.train_gt_dists == 0) # same group
    #     # pick 2 that are 0 distance from eachother
    #     p1 = np.random.choice(row)
    #     p2 = np.random.choice(np.where(self.train_gt_dists[p1,:]==0)[0])

    #     # for p1 and p2, pick ones that are 2 away, 4 away and 6 away, if available.
        
    #     samples_per_level = [int(np.round(((batch_size - 2)/2)/len(negative_levels))) for i in range(len(negative_levels))]
    #     if sum(samples_per_level) != int((batch_size-2)/2):
    #         samples_per_level[-1] -= sum(samples_per_level) - int((batch_size-2)/2)
    #     assert sum(samples_per_level) == int((batch_size -2)/2)

    #     get_indices = [p1, p2]
    #     import ipdb; ipdb.set_trace()

    #     for p in [p1, p2]:
    #         for idx, dist in enumerate(negative_levels):
    #             match_indices = np.where(self.train_gt_dists[p,:] == dist)[0]
    #             if len(match_indices) < samples_per_level[idx]:
    #                 # then let choice be replace = True
    #                 neg_indices = np.random.choice(match_indices, samples_per_level[idx], replace=True)
    #             else:
    #                 neg_indices = np.random.choice(match_indices, samples_per_level[idx], replace=False)

    #             get_indices.extend(neg_indices.tolist())

    #     assert len(get_indices) == batch_size

    #     return [self.training_data[i] for i in get_indices]

    def upate_next_batch(self, distance_matrix):
        # distance_matrix is mxm, where the index along both rows *and* columns are
        # the corresponding video in self.training_data


        # find random GT dist 0 pairs, 

        # Then, for each, find closest GT dist 2 pairs, then 4

        


        pass 

    def __getitem__(self, idx):

        batch = [self.training_data[element] for element in self.rand_selection_indices[idx]]

        return EK_Dataset_pretrain_batchwise.process(batch, self.output_cache_fullpath, self.crop_type, self.processed_frame_number, self.f2bbox, 
                self.image_data_folder, self.noun_dict, self.knowns, self.unknowns, self.unknown_lowest_level_label, 
                self.individual_transform, self.batchwise_transform)

         


class EK_Dataset_pretrain_pairwise(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            model_saveloc,
            training_num_samples=10000,
            validation_num_samples=200,
            filter_function = default_filter_function,
            processed_frame_number = 20, 
            individual_transform=None,
            pairwise_transform=None,
            mode='resnet', output_cache_folder='dataloader_cache/', 
            snip_threshold=32,
            crop_type='blackout',
            sampling_mode='equality',
            selector_train_ratio=0.75):
        # purpose of the dataset object: either for pretraining or for training/testing
        super(EK_Dataset_pretrain_pairwise, self).__init__()
        self.image_data_folder = image_data_folder
        self.knowns = knowns
        self.unknowns = unknowns
        self.individual_transform = individual_transform
        self.pairwise_transform = pairwise_transform
        self.train_num_samples = training_num_samples
        self.val_num_samples = validation_num_samples

        self.class_key_df = pd.read_csv(class_key_path)

        self.mode = mode
        assert self.mode == 'resnet' or self.mode == 'noresnet'
        self.output_cache_folder = output_cache_folder
        if not os.path.exists(self.output_cache_folder):
            os.makedirs(self.output_cache_folder, exist_ok = True)
        self.snip_threshold = snip_threshold
        self.crop_type = crop_type
        assert crop_type == 'blackout' or crop_type == 'rescale', 'crop_type must either be blackout or rescale'
        # TODO: using the key, convert strings into unkowns
        self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
        self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

        self.DF = DatasetFactory(knowns, unknowns,
                    object_data_path, action_data_path, class_key_path)

        self.dataset = self.DF.get_dataset()
        assert 'unknown_frame2bbox' in self.dataset \
                and 'known_frame2bbox' in self.dataset, 'frame2bbox conversion not found'
        self.training_data = self.dataset['known_pretrain']

        self.snip_clips()

        buffer_ = []
        for idx, sample in enumerate(self.training_data):
            if filter_function(sample):
                buffer_.append(sample)
        self.training_data = buffer_
        del buffer_
        self.f2bbox = self.dataset['known_frame2bbox']

        # used to handle cases when the clip is especially long
        self.processed_frame_number = processed_frame_number

        selector = Selector(self.training_data, option='equality', train_ratio=selector_train_ratio)
        self.rand_selection_indices = selector.get_indices('train')
        self.val_indices = selector.get_indices('val')

        # self.rand_selection_indices = [np.random.choice(int(len(self.training_data)*0.8),2, replace=False).tolist() for i in range(self.train_num_samples)]
        # self.val_indices = [[element[0] + int(len(self.training_data)*0.8), element[1] + int(len(self.training_data)*0.8)] for element in \
        #             [list(np.random.choice(len(self.training_data) - int (len(self.training_data)*0.8), 2, replace=False)) for i in range(self.val_num_samples)]]
        # # naive sampling of pairwise
        # self.rand_selection_indices = []
        # for i in range(self.train_num_samples + self.val_num_samples):
        #     selection_indices = list(np.random.choice(len(self.training_data), 2))
        #     self.rand_selection_indices.append(selection_indices)
        
        # self.val_indices = self.rand_selection_indices[self.train_num_samples:]
        # self.rand_selection_indices = self.rand_selection_indices[:self.train_num_samples]
        
        self.output_cache_fullpath = os.path.join(os.path.join(self.output_cache_folder, self.mode + '_out'), self.crop_type)
        if not os.path.exists(self.output_cache_fullpath):
            os.makedirs(self.output_cache_fullpath, exist_ok = True)

        self.unknown_lowest_level_label = survey_tree(self.knowns)


        # saving processing_params.pkl
        with open(os.path.join(model_saveloc, 'processing_params.pkl'),'wb') as f:
            pickle.dump({'output_cache_fullpath': self.output_cache_fullpath, 
                        'crop_type': self.crop_type, 
                        'processed_frame_number': self.processed_frame_number,
                        'f2bbox': self.f2bbox, 
                        'image_data_folder': self.image_data_folder, 
                        'noun_dict': self.noun_dict, 
                        'knowns': self.knowns, 
                        'unknowns': self.unknowns, 
                        'unknown_lowest_level_label': self.unknown_lowest_level_label, 
                        'individual_transform': self.individual_transform, 
                        'pairwise_transform': self.pairwise_transform}, f )

    def snip_clips(self):
        snip_clips = []
        for clip in self.training_data:
            if int((clip['end_frame']-clip['start_frame'])/30) > self.snip_threshold:
                a = clip['start_frame']
                while a + (self.snip_threshold-1)*30 <= clip['end_frame']:
                    b = a + (self.snip_threshold-1)*30
                    # add the clip 
                    snip_clip = clip.copy()
                    snip_clip['start_frame'] = a
                    snip_clip['end_frame'] = b
                    snip_clips.append(snip_clip)
                    # update the starting point
                    a = b + 30
            else:
                snip_clips.append(clip)

        self.training_data = snip_clips
        del snip_clips

    def get_val_dataset(self):
        val_set = []
        counter = 0
        
        for i in range(int(len(self.val_indices)/4)):
            sample_batch = []
            for val_index in self.val_indices[i*4 : i*4 + 4]:
                sample_a = self.training_data[val_index[0]]
                sample_b = self.training_data[val_index[1]]
                sample_batch.append(EK_Dataset_pretrain_pairwise.process(sample_a, sample_b, self.output_cache_fullpath, self.crop_type, 
                                                self.processed_frame_number, self.f2bbox, 
                                                self.image_data_folder, self.noun_dict, self.knowns, 
                                                self.unknowns, self.unknown_lowest_level_label, 
                                                self.individual_transform, self.pairwise_transform))
            val_set.append(sample_batch)

        return val_set

    @staticmethod
    def process(sample_a, sample_b, output_cache_fullpath, crop_type, processed_frame_number,f2bbox, 
                image_data_folder, noun_dict, knowns, unknowns, unknown_lowest_level_label, 
                individual_transform, pairwise_transform, overwrite=False):
        indiv_output_d = []
        for sample_dict in [sample_a, sample_b]:
            video_id = sample_dict['video_id']
            participant_id = sample_dict['participant_id']
            start_frame = sample_dict['start_frame']
            end_frame = sample_dict['end_frame']

            # TODO: make cache name, search for cache file, if found load.
            cache_filename = 'v#{}p#{}s#{}e#{}.pkl'.format(video_id, participant_id, start_frame, end_frame)
            if os.path.exists(os.path.join(output_cache_fullpath, cache_filename)) and not overwrite:
                # loding d
                with open(os.path.join(output_cache_fullpath, cache_filename), 'rb') as f:
                    d = pickle.load(f)
                indiv_output_d.append(d)

            else:

                if crop_type == 'blackout':
                    frames = crop_wrapper(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                                    scaling = 0.5, cache_dir='dataloader_cache/blackout_crop', mode=crop_type)
                else:

                    frames = crop_wrapper(sample_dict, processed_frame_number, f2bbox, image_data_folder, threshold=2, 
                                    scaling = 0.5, cache_dir='dataloader_cache/rescale_crop', mode=crop_type)
                    # import ipdb; ipdb.set_trace()

                # get position in the tree
                encoding = get_tree_position(noun_dict[sample_dict['noun_class']], knowns)
                if encoding is None:
                    top_levels = tuple(get_tree_position(noun_dict[sample_dict['noun_class']], unknowns)[:-1])
                    assert top_levels in unknown_lowest_level_label
                    encoding = np.array(list(top_levels)+[unknown_lowest_level_label[top_levels][0]])

                d = {'frames': frames,
                     'noun_label': noun_dict[sample_dict['noun_class']],
                     'hierarchy_encoding': encoding}

                if individual_transform is not None:
                    # import ipdb; ipdb.set_trace()

                    d = individual_transform(d)
                    # import ipdb; ipdb.set_trace() 


                # saving into cache file
                with open(os.path.join(output_cache_fullpath, cache_filename), 'wb') as f:
                    pickle.dump(d, f)

                indiv_output_d.append(d)
                
        # get distance bretween the two noun classes
        pairwise_tree_dist = get_tree_distance(indiv_output_d[0]['noun_label'],
                                            indiv_output_d[1]['noun_label'])

        output = {'frames_a': indiv_output_d[0]['frames'],
                    'frames_b':indiv_output_d[1]['frames'],
                    'noun_label_a': indiv_output_d[0]['noun_label'],
                    'noun_label_b': indiv_output_d[1]['noun_label'],
                    'dist': pairwise_tree_dist}

        if pairwise_transform is not None:
            output = pairwise_transform(output)
        
        return output

    def __len__(self):
        return len(self.rand_selection_indices)

    def __getitem__(self, idx):        
        sample_a = self.training_data[self.rand_selection_indices[idx][0]]
        sample_b = self.training_data[self.rand_selection_indices[idx][1]]

        return EK_Dataset_pretrain_pairwise.process(sample_a, sample_b, self.output_cache_fullpath, self.crop_type, self.processed_frame_number, self.f2bbox, 
                self.image_data_folder, self.noun_dict, self.knowns, self.unknowns, self.unknown_lowest_level_label, 
                self.individual_transform, self.pairwise_transform)
        
# NOTE: DEPRECATED
class EK_Dataset_pretrain(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            processed_frame_number = 20,
            filter_function = default_filter_function,
            transform=None,
            crop_type='blackout'):
        # purpose of the dataset object: either for pretraining or for training/testing
        super(EK_Dataset_pretrain, self).__init__()
        self.image_data_folder = image_data_folder
        self.knowns = knowns
        self.unknowns = unknowns
        self.transform = transform
        self.class_key_df = pd.read_csv(class_key_path)
        self.crop_type = crop_type
        assert crop_type == 'blackout' or crop_type == 'rescale', 'crop_type must either be blackout or rescale'
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
        self.unknown_lowest_level_label = survey_tree(self.knowns)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        sample_dict = self.training_data[idx]
        video_id = sample_dict['video_id']
        participant_id = sample_dict['participant_id']
        start_frame = sample_dict['start_frame']
        end_frame = sample_dict['end_frame']
        
        if self.crop_type == 'blackout':
            frames = crop_wrapper(sample_dict, self.processed_frame_number, self.f2bbox, self.image_data_folder, threshold=2, 
                                scaling = 0.5, cache_dir='dataloader_cache/blackout_crop', mode=self.crop_type)
        else:
            frames = crop_wrapper(sample_dict, self.processed_frame_number, self.f2bbox, self.image_data_folder, threshold=2, 
                                scaling = 0.5, cache_dir='dataloader_cache/rescale_crop', mode=self.crop_type)
            
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
    split = get_known_unknown_split()
    # with open('current_split.pkl','rb') as f:
    #     split = pickle.load(f)
    knowns = split['training_known']
    unknowns = split['training_unknown']
    
    # DF_pretrain = EK_Dataset_pretrain_pairwise(knowns, unknowns,
    #         train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder)

    DF_pretrain = EK_Dataset_pretrain_batchwise(knowns, unknowns,
             train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder, 'rm_me')
    import ipdb; ipdb.set_trace()
    print(DF_pretrain[14])
    # for i in range(4):
    #     import ipdb; ipdb.set_trace()
    #     print(DF_pretrain[112+i])
    
    
