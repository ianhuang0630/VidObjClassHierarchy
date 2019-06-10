"""
This script is created for classes that loads the data in various different ways
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# this class takes in knowns, unknowns, among other parameters, then searches
# database for sequences of videos in which at least one instance of the known
# and one instance of the unknown dataset is present. Other formats of this 
# dataset will be available according to other settings of parameters.

class DatasetFactory(object):
    def __init__(self, knowns, unknowns, object_data_path, action_data_path,
            class_key_path, cache_folder='dataset_cache/', options='separate'):
        """
        Args:
            knowns: list of known class ids
            unknowns: list of unknown class ids
        """
        self.known_classes = knowns
        self.unknown_classes = unknowns 
        # paths to the datasets
        self.object_data_path = object_data_path
        self.action_data_path = action_data_path
        self.class_key_path = class_key_path
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder, exist_ok=True)
        # path to the cache folder
        self.cache_folder = cache_folder
        # loading all datasets using pandas
        self.object_data = pd.read_csv(self.object_data_path)
        self.action_data = pd.read_csv(self.action_data_path)
        self.class_key = pd.read_csv(self.class_key_path) 
        
        self.options = options
        # first search the cache folder
        if self.found_in_cache():
            # loading cache
            print('Exact requirements found in cache, loading from cache folder...')
            self.dataset = self.load_cache()
            print('Done.')
        else:
            # if not in cache folder, call self.construct_dataset()
            print('Requirements not satisfied in cache, constructing dataset now...')
            self.dataset = self.construct_dataset()
            print('Done.')
    def found_in_cache(self):
        # TODO
        return False

    def load_cache(self):
        # TODO
        pass
    
    def construct_dataset(self):
        print('Constructing datasets')
        # first get_coexistence_candidates() or get_known_unknown_candidates()
        if self.options == 'coexistence':
            video_candidates = self.get_coexistence_candidates()
            return None
        elif self.options == 'separate':
            video_candidates = self.get_known_unknown_candidates()
            return self.search_known_unknown(video_candidates)
        # filter candidates
        else:
            raise ValueError('{} not recognized as an option'.format(self.options))
        print('Done.')
        

    def get_coexistence_candidates(self):
        # filter out data with both known and unknowns
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
    
    def search_known_unknown(self, video_candidates):
        # organize current video by frame number, video_candidates
        # return 'participant_id', 'video id', 'start_frame', 'end_frame' for 
        # every single class in only unknonw (if mode is known_unknown)
        dataset = []
        for video in video_candidates['unknown']:
            video_id = video[0]
            participant_id = video[1]
            # helpful datastructures 
            states_dict = {element: 'off' for element in self.unknown_classes}
            stacks_dict = {element: [] for element in self.unknown_classes}
            sorted_video = video[2].sort_values(by=['frame']) 
            
            # now proceed to find the clips.
            frames_and_classes = []
            frame_index = 0
            for index, row  in sorted_video.iterrows():
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

            for idx, element in enumerate(tqdm(frames_and_classes)):
                for class_ in states_dict:
                    if class_ in element[1:] and states_dict[class_] == 'off':
                        stacks_dict[class_].append(element[0])
                        states_dict[class_] = 'on'
                    if (class_ not in element[1:] and states_dict[class_] == 'on') or (idx == len(frames_and_classes)-1 and states_dict[class_]=='on'):
                        end_frame = frames_and_classes[idx-1][0]
                        start_frame = stacks_dict[class_].pop()
                        states_dict[class_] = 'off'
                        dataset.append({'video_id': video_id,
                                        'participant_id': participant_id,
                                        'start_frame': start_frame,
                                        'end_frame': end_frame,
                                        'noun_class': class_})
        return dataset 
         
    def window_search_coexistence(self):
        pass

    def get_dataset(self):
        # TODO
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
    
