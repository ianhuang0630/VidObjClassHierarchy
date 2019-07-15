"""
This script is created for classes that loads the data in various different ways
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import json
import os
import pickle
# this class takes in knowns, unknowns, among other parameters, then searches
# database for sequences of videos in which at least one instance of the known
# and one instance of the unknown dataset is present. Other formats of this 
# dataset will be available according to other settings of parameters.

DEBUG=True
if DEBUG:
    random.seed(7)

class DatasetFactory(object):
    def __init__(self, knowns, unknowns, object_data_path, action_data_path,
            class_key_path, cache_folder='dataset_cache/', options='separate',
            known_format = 'clips', overwrite=False):
        """
        Args:
            knowns: list of known class ids
            unknowns: list of unknown class ids
            object_data_path: path to the object data csv
            action_data_path: path to the action data csv
            class_key_path: path to the csv converting between numerical labels
                to linguistic ones
            cache_folder: Where to save the results for future use
            options: can be either 'separate' or 'coexistence'. When ='separate',
                we only require that the output for unknown classes contain clips
                featuringsone of the unknown classes (with no restrictions on
                whether any known classes are present). When ='coexistence', it 
                is required that at least one of each set (known and unknown) is
                featured in a video clip.
            known_format: can either be 'clips' or 'video'. Clips are cut out to
                tightly temporally bound the presence of known classes, whereas
                when 'video' option is on, the whole video containing a known
                class is returned.
            overwrite: whether to overwrite cache
        """
        self.known_classes = knowns
        self.unknown_classes = unknowns 
        # paths to the datasets
        self.object_data_path = object_data_path
        self.action_data_path = action_data_path
        self.class_key_path = class_key_path
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder, exist_ok=True)
        self.overwrite = overwrite
        self.known_format = known_format
        # path to the cache folder
        self.cache_folder = cache_folder
        # loading all datasets using pandas
        self.object_data = pd.read_csv(self.object_data_path)
        self.action_data = pd.read_csv(self.action_data_path)
        self.class_key_df = pd.read_csv(self.class_key_path) 
        
        # TODO: using the key, convert strings into unkowns
        class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
        if all([type(element) is str for element in self.known_classes]):
            self.known_classes = [class_key_dict[element] for element in self.known_classes]
        if all([type(element) is str for element in self.unknown_classes]):
            self.unknown_classes = [class_key_dict[element] for element in self.unknown_classes]


        # listing all available files
        config_pkls = os.listdir(self.cache_folder)
        assert len(config_pkls)%2==0, 'missing files in the config folder {}'.format(self.cache_folder)
        version_number = int(len(config_pkls)/2 + 1 )

        self.cache_filename = 'data_version{}.json'.format(version_number)
        self.config_filename = 'config_version{}.pkl'.format(version_number)

        #'config_known_'+'_'.join([str(element) for element in sorted(self.unknown_classes)]) \
        # +'_unknown_'+'_'.join([str(element) for element in sorted(self.known_classes)]) +'.pkl'
        self.config = {'known_classes': self.known_classes,
                'unknown_classes': self.unknown_classes,
                'object_csv': self.object_data_path,
                'action_csv': self.action_data_path,
                'class_key_csv': self.class_key_path,
                'overwrite': self.overwrite,
                'known_format': self.known_format,
                'cache_folder': self.cache_folder}
        

        self.options = options
        # first search the cache folder
        cache_return = self.found_in_cache()
        if cache_return is not None and not self.overwrite:
            # loading cache
            print('Exact requirements found in cache, loading from cache folder...')
            self.final_dataset = self.load_cache(cache_return)
            print('Done.')
        else:
            # if not in cache folder, call self.construct_dataset()
            print('Requirements not satisfied in cache, constructing dataset now...')
            self.dataset = self.construct_dataset()
            print('Done.')
            print('Saving to file: ')
            # pretraining and training split over the knowns
            split = self.get_pretrain_training_split()
            self.final_dataset = {'known': split['train'], 
                             'unknown': self.dataset['unknown'],
                             'known_pretrain': split['pretrain']}
            
            if 'unknown_frame2bbox' in self.dataset and 'known_frame2bbox' in self.dataset:
                self.final_dataset['unknown_frame2bbox'] = self.dataset['unknown_frame2bbox']
                self.final_dataset['known_frame2bbox'] = self.dataset['known_frame2bbox']

            with open(os.path.join(self.cache_folder, self.config_filename), 'wb') as f:
                pickle.dump(self.config, f) 
            with open(os.path.join(self.cache_folder, self.cache_filename), 'w') as f:
            	json.dump(self.final_dataset, f)  
            print('Done.')
    
    def get_dataset(self):
        return self.final_dataset

    def get_pretrain_training_split(self):
        # now, we split self.dataset into two parts
        # a subset of all knowns to pretrain the tree hierarchy predictor g
        
        known_data = self.dataset['known']
        knowns = {}
        for sample in known_data:
            if sample['noun_class'] not in knowns:
                knowns[sample['noun_class']] = [sample]
            else:
                knowns[sample['noun_class']].append(sample)
        
        # below is a very stringent, and if it happens that there exists a key that
        # assert all([len(knowns[key]) >=2  for key in knowns])
         
        known_pretrain = []
        known_train = []
        # splitting 80-20 training-pretraining
        for class_ in knowns:
            num_pretrain_samples = int(np.ceil(len(knowns[class_]) * 0.2)) # round up
            pretrain_indices = np.random.choice(range(len(knowns[class_])), num_pretrain_samples,
                                    replace=False)  
            pretrain_data = [knowns[class_][i] for i in pretrain_indices]
            known_pretrain.extend(pretrain_data)

            train_indices = list(set(range(len(knowns[class_]))) - set(pretrain_indices))
            train_data = [knowns[class_][i] for i in train_indices]
            known_train.extend(train_data) 
        random.shuffle(known_pretrain)
        random.shuffle(known_train)
        return {'pretrain': known_pretrain, 'train': known_train}

    def visualize_dataset_info(self):
        pass

    def found_in_cache(self):
        files = os.listdir(self.cache_folder)
        config_files = [filename for filename in files if filename [-4:] == '.pkl']
        
        for config_file in config_files:
            with open(os.path.join(self.cache_folder, config_file), 'rb') as f:
                candidate = pickle.load(f)
            if set(candidate['known_classes']) == set(self.known_classes) \
                and set(candidate['unknown_classes']) == set(self.unknown_classes):
                
                return int(config_file[len('config_version'):-len('.pkl')])

        return None 

        # # TODO
        # if os.path.exists(os.path.join(self.cache_folder, self.config_filename)):
        #     with open(os.path.join(self.cache_folder, self.config_filename), 'rb') as f:
        #         return self.config == pickle.load(f) \
        #                 and os.path.exists(os.path.join(self.cache_folder, self.cache_filename))
        # else:
        #     return False

    def load_cache(self, version_number):
        cache_filename = 'data_version{}.json'.format(version_number)
        with open(os.path.join(self.cache_folder, cache_filename), 'r') as f:
            return json.load(f)
         
    def construct_dataset(self):
        print('Constructing datasets')
        # first get_coexistence_candidates() or get_known_unknown_candidates()
        if self.options == 'coexistence':
            video_candidates = self.get_coexistence_candidates()
            raise ValueError('Not fully implemented.')
        elif self.options == 'separate':
            video_candidates = self.get_known_unknown_candidates()
            unknown_clips, unknown_frame2bbox = self.search_clips(video_candidates, search_target = 'unknown')
            if self.known_format == 'clips':
                known_clips, known_frame2bbox = self.search_clips(video_candidates, search_target = 'known')
                print('Done.')
                return {'known': known_clips, 'unknown': unknown_clips, 
                        'known_frame2bbox': known_frame2bbox,
                        'unknown_frame2bbox': unknown_frame2bbox}
            elif self.known_format == 'videos':
                known_videos = self.organize_known(video_candidates)
                print('Done.')
                return {'known': known_videos, 'unknown': unknown_clips}
        # filter candidates
        else:
            raise ValueError('{} not recognized as an option'.format(self.options))

    def get_coexistence_candidates(self):
        # filter out data with both known and unknowns
        raise ValueError('Not fully implemented')
        video_candidates = []
        print("Scanning each subject for candidate video frames...")
        for subject in tqdm(list(set(self.object_data['participant_id']))):
            sub_df = self.object_data.loc[self.object_data['participant_id'] == subject]
            for video_id in list(set(sub_df['video_id'])):
                subsub_df = sub_df.loc[sub_df['video_id'] == video_id]
                # check within this df if it contains both known classes and
                # unknown classes. If at least one of each is present, add to list.
                bb_subsub_df = subsub_df.loc[subsub_df['bounding_boxes']!='[]']
                nc_bb_subsub_df = set(bb_subsub_df['noun_class'])
                
                if np.any([element in nc_bb_subsub_df for element in self.known_classes]) \
                    and np.any([element in nc_bb_subsub_df for element in self.unknown_classes]):
                    # saving relevant information
                    video_candidates.append((subject, video_id, subsub_df))
        print('Done.')
        return video_candidates
    
    def get_known_unknown_candidates(self):
        video_candidates = {'known': [], 'unknown':[]}
        print("Scanning each subject for candidate video frames...")
        for subject in tqdm(list(set(self.object_data['participant_id']))):
            sub_df = self.object_data.loc[self.object_data['participant_id'] == subject]
            for video_id in list(set(sub_df['video_id'])):
                subsub_df = sub_df.loc[sub_df['video_id'] == video_id]
                # check within this df if it contains both known classes and
                # unknown classes. If at least one of each is present, add to list.
                bb_subsub_df = subsub_df.loc[subsub_df['bounding_boxes']!='[]']
                nc_bb_subsub_df = set(bb_subsub_df['noun_class'])
                
                if np.any([element in nc_bb_subsub_df for element in self.known_classes]):
                    video_candidates['known'].append((subject, video_id, subsub_df)) 

                if np.any([element in nc_bb_subsub_df for element in self.unknown_classes]):
                    # saving relevant information
                    video_candidates['unknown'].append((subject, video_id, subsub_df))
        print("Done")
        return video_candidates
            
     
    def search_clips(self, video_candidates, search_target):
        # organize current video by frame number, video_candidates
        # return 'participant_id', 'video id', 'start_frame', 'end_frame' for 
        # every single class in only unknonw (if mode is known_unknown)
        dataset = []
        if search_target == 'unknown':
            set_of_interest = self.unknown_classes
        elif search_target == 'known':
            set_of_interest = self.known_classes
        else:
            raise ValueError('{} is not a valid objection for search_target.'.format(search_target))
        
        frame_to_bounding_boxes = {}
        for video in video_candidates[search_target]:
            participant_id = video[0]
            video_id = video[1]
            # import ipdb; ipdb.set_trace() 
            # helpful datastructures 
            states_dict = {element: 'off' for element in set_of_interest}
            stacks_dict = {element: [] for element in set_of_interest} 
            sorted_video = video[2].sort_values(by=['frame']) 
            
            # now proceed to find the clips.
            frames_and_classes = []
            frame_index = 0
            for index, row  in sorted_video.iterrows():
                if row['frame'] not in frame_to_bounding_boxes:
                    frame_to_bounding_boxes[str(participant_id)+'/'+str(video_id)+'/'+str(row['frame'])] = \
                            [{'noun_class':row['noun_class'], 
                                'bbox':row['bounding_boxes']}]
                else:
                    frame_to_bounding_boxes[str(participant_id)+'/'+str(video_id)+'/'+str(row['frame'])].append(
                            {'noun_class': row['noun_class'], 
                                'bbox': row['bounding_boxes']}
                            )

                if row['frame'] > frame_index:
                    # first bounding box in new frame
                    if row['bounding_boxes'] != '[]':
                        frames_and_classes.append([row['frame'], row['noun_class']])
                        frame_index = row['frame']
                elif row['frame'] == frame_index:
                    if row['bounding_boxes'] != '[]':
                        frames_and_classes[-1].append(row['noun_class'])
                else:
                    raise ValueError("current frame is {} but frame_index is {}".format(row['frame'], frame_index))
            #all_frame_to_bounding_boxes[(participant_id, video_id)] = frame_to_bounding_boxes 
            previous_frame = None
            for idx, element in enumerate(tqdm(frames_and_classes)):
                for class_ in states_dict:
                    # if new frame isn't 30 from the previous, 
                    if idx!= 0 and (element[0] - previous_frame == 30):
                        if class_ in element[1:] and states_dict[class_] == 'off':
                            stacks_dict[class_].append(element[0]) # element[0] is the frame number ..
                            states_dict[class_] = 'on'
                            juststarted = True
                        if (class_ not in element[1:] and states_dict[class_] == 'on') \
                            or (idx == len(frames_and_classes)-1 and states_dict[class_]=='on'):
                            end_frame = frames_and_classes[idx-1][0]
                            start_frame = stacks_dict[class_].pop()
                            states_dict[class_] = 'off'
                            dataset.append({'video_id': video_id,
                                            'participant_id': participant_id,
                                            'start_frame': start_frame,
                                            'end_frame': end_frame,
                                            'noun_class': class_})
                    else:
                        if states_dict[class_] == 'on':
                            end_frame = frames_and_classes[idx-1][0]
                            start_frame = stacks_dict[class_].pop()
                            states_dict[class_] = 'off'
                            dataset.append({'video_id': video_id,
                                            'participant_id': participant_id,
                                            'start_frame': start_frame,
                                            'end_frame': end_frame,
                                            'noun_class': class_})
                previous_frame = element[0]

        return dataset, frame_to_bounding_boxes 
    
    def organize_known(self, video_candidates):
        dataset = []
        for video in video_candidates['known']:
            video_id = video[0]
            participant_id = video[1]
            start_frame = min(video[2]['frame'])
            end_frame = max(video[2]['frame'])
            class_= list(set(video[2]['noun_class']).intersection(set(self.known_classes)))
            dataset.append({'video_id': video_id,
                            'participant_id':participant_id,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'noun_class': class_}) 
        return dataset 

    def window_search_coexistence(self):
        pass

if __name__=='__main__':
    dataset_path = '/vision/group/EPIC-KITCHENS/'
    annotations_foldername = 'annotations'
    annotations_folderpath = os.path.join(dataset_path, annotations_foldername)
    visual_dataset_path = os.path.join(dataset_path, 'EPIC_KITCHENS_2018')
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

    action_df = pd.read_csv(train_action_csvpath)
    object_df = pd.read_csv(train_object_csvpath)
    class_key_df = pd.read_csv(class_key_csvpath)
    class_key_dict = dict(zip(class_key_df.class_key, class_key_df.noun_id))
    knowns = ['pan', 'onion'] 
    unknowns = ['plate', 'meat']
    knowns_id = [class_key_dict[element] for element in knowns]
    unknowns_id = [class_key_dict[element] for element in unknowns]
    DF = DatasetFactory(knowns_id, unknowns_id, train_object_csvpath, train_action_csvpath, class_key_csvpath)
    
