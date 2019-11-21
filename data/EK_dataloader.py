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

import sys
sys.path.insert(0, './') # necessary for loading the models
sys.path.insert(0, 'src')

from torchvision import transforms
from input_layers import InputLayer
try:
    from gt_hierarchy import *
    from EK_dataset import DatasetFactory
    from sampler import Selector
    from transforms import *
except: 
    from data.gt_hierarchy import *
    from data.EK_dataset import DatasetFactory
    from data.sampler import Selector
    from data.transforms import *

from scipy import stats
import time

from tqdm import tqdm

DEBUG = False

if DEBUG:
    random.seed(7)
    np.random.seed(7)


# from maskrcnn_benchmark.structures.bounding_box import BoxList
# class COCO_EK_Dataset_detection(EK_Dataset_detection):
#     def __init__(self, 
#                 img_height=1080, img_width=1920):
#         self.img_height = img_height
#         self.img_width = img_width
        
#     def __getitem__(self, idx):
#         # get the idx_th image,

#         # get the list of bounding boxes
        
#         # get the list of knowns and unknowns
#         pass

#     def get_img_info(self, idx):
#         return {'height': self.img_height, 'width': self.img_width}
        

def default_filter_function(d):
    # filters out the ones where there are too few frames present
    return (d['end_frame'] - d['start_frame'])/30 > 10

def create_config_file(threshold, processed_frame_number, scaling=0.5, cache_dir='dataloader_cache/blackout_crop'):
    # TODO import things: threshold, 
    config = {'threshold': threshold, 'processed_frame_number': processed_frame_number, 
            'scaling': scaling}
    with open (os.path.join(cache_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

class EK_Dataset_detection(Dataset):
    def __init__(self, knowns, unknowns, object_data_path, action_data_path,
                class_key_path, image_data_folder, dataset, inputlayer,
                tree_encoder = 'models/pretraining_tree/framelevel_pred_run0/net_epoch0.pth',
                device='cuda:0', iou_threshold=0.5, manual_anchors = False, max_num_boxes=200,
                max_gt_boxes=10, purpose='train'):

        super(EK_Dataset_detection,self).__init__()
        self.device = device
        self.knowns = knowns
        self.unknowns = unknowns
        self.image_data_folder = image_data_folder
        self.iou_thresh = iou_threshold

        self.max_num_boxes = max_num_boxes
        self.max_gt_boxes = max_gt_boxes
        self.purpose = purpose

        self.class_key_df = pd.read_csv(class_key_path)
        self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
        self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

        self.admissible_classes = set(knowns + unknowns)
        self.admissible_noun_id = set([self.class_key_dict[element] for element in list(self.admissible_classes)])

        # dataset containing the images under known and unknown classes
        self.dataset = dataset
        # prune both the known and unknown datasets to static images
        unknown = self.prune_to_frames(target = 'unknown')
        known = self.prune_to_frames(target = 'known')

        self.uk_dataset = unknown + known
        random.shuffle(self.uk_dataset)

        self.manual_anchors = manual_anchors

        if not self.manual_anchors: # we won't be using inputlayer if manual_anchors is True
            self.inputlayer = inputlayer # this generates the region proposals
        else:
            self.anchors = self.generate_anchors()
            print('{} anchors generated'.format(len(self.anchors)))
        # for the tree encoder
        
        if purpose=='train' and purpose=='test': 
            self.tree_encoder_location = tree_encoder
            self.tree_encoder = torch.load(self.tree_encoder_location)
            self.tree_encoder = self.tree_encoder.to(self.device)
            self.tree_encoder.eval()
            self.tree_encoder_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225]),
                                                GetResnetFeats()
                                                ])
        elif purpose == 'resnet101':
            assert tree_encoder is None, 'Just for safety, set self.tree_encoder_location to None'
            self.resnet101 = GetResnetFeatsGeneral(version='resnet101', mode='vec', device=self.device)

            self.tree_encoder = self.tree_encoder_substitute
            self.tree_encoder_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225])])

    def tree_encoder_substitute(self, x):
        with torch.no_grad():
            res101_embeddings = self.resnet101(x)
        return {'embedding': res101_embeddings}

    def prune_to_frames(self, target='known_pretrain'):
        # check that 1: has a bounding box of the known class
        frames2bbox = self.dataset['known_frame2bbox' if target in ('known_pretrain', 'known') else 'unknown_frame2bbox']

        frames = []
        print('turning into frames for target {}'.format(target))

        for clip in tqdm(self.dataset[target]):
            vid_id = clip['video_id']
            part_id = clip['participant_id']
            start_frame = clip['start_frame']
            end_frame = clip['end_frame']
            
            a = int(start_frame)
            while a <= int(end_frame):
                # process
                key = '/'.join([part_id, vid_id, str(a)])
                gt_bboxes = []
                gt_classes = []

                if key in frames2bbox:
                    candidates = [bbox for bbox in frames2bbox[key] if bbox['noun_class'] in self.admissible_noun_id]
                    for element in candidates:
                        this_bbox = ast.literal_eval(element['bbox'])
                        if len(this_bbox) > 0:
                            gt_bboxes.append(this_bbox[0])
                            gt_classes.append(self.noun_dict[element['noun_class']])

                    # save: vid_id, part_id, a, this_bbox, class
                    frames.append({'video_id': vid_id,
                                    'participant_id': part_id,
                                    'frame_number': a,
                                    'bboxes': gt_bboxes,
                                    'noun_classes': gt_classes})
                a += 30

        # WARNING frames may contain repeats

        return frames


    def generate_anchors(self, image_width=1920, image_height=1080, num_w_strides=28, num_h_strides=16, 
            patterns=[[200,200], [160, 240], [240, 160], [600, 600], [540, 660], [660, 540]]):
        
        w_anchor_centers = np.linspace(0, image_width-1, num_w_strides+2)[1:-1]
        h_anchor_centers = np.linspace(0, image_height-1, num_h_strides+2)[1:-1]

        patterns = np.array(patterns)
        add = patterns/2
        
        all_anchors = []    
        for h in h_anchor_centers:
            for w in w_anchor_centers:

                center = np.array([w,h])
                minus = np.maximum(center-add, np.zeros_like(center-add))
                plus = np.minimum(center+add, np.array([image_width, image_height]) * np.ones_like(center+add))

                anchors = np.hstack((minus, plus)) # all of these should be centered around w,h

                all_anchors.append(anchors.astype(np.int))

        return all_anchors    




    def crop_and_transform(self, bboxes, image, sample_dict, cache_folder='cache/bounding_box_feature_map', overwrite=True):

        if not os.path.exists(os.path.join(cache_folder, self.purpose)):
            os.makedirs(os.path.join(cache_folder, self.purpose))
        video_id = sample_dict['video_id']
        participant_id = sample_dict['participant_id']
        frame_number = sample_dict['frame_number']
        filename = '{}-{}-{}.pkl'.format(participant_id, video_id, frame_number)
        filepath = os.path.join(os.path.join(cache_folder, self.purpose), filename)

        if os.path.exists(filepath):
            # print("loading cache {}".format(filepath))
            with open(filepath, 'rb') as f:
                crop_tfs = pickle.load(f)

        else:
            crop_tfs = []
            # cropping first 
            for bbox in bboxes:
                top_left = bbox[:2] # w x h
                bottom_right = bbox[2:] # w x h

                crop = image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0]), :]
                # crop_resz = cv2.resize(crop, tuple([int(dim*scale) for dim in crop.shape[:2][::-1]]))
                crop = crop[:, :, [2,1,0]]

                crop_tf = self.tree_encoder_transforms(crop)
                # now pass through transforms
                crop_tfs.append(crop_tf)

            with open(filepath, 'wb') as f:
                pickle.dump(crop_tfs, f)

        return torch.stack(crop_tfs)

    def bb_intersection_over_union(self, boxA, boxB):
        # Topleft and bottomright corners format 

        # fr
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
     
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
     
        # return the intersection over union value
        return iou

    def get_gt_label(self, noun_name):
        if noun_name in self.knowns:
            return 1
        elif noun_name in self.unknowns:
            return 2
        else:
            raise ValueError('gt_bboxes contain other classes than permissible ones. recheck it.')

    def get_labels(self, predictions, gt_bboxes, gt_noun_classes):
        # convert gt_bboxes to the corner format, but also in the format width,height instead of height_width
        gt_bboxes = [np.array(tup) for tup in gt_bboxes]
        gt_bboxes_corner = [np.hstack([element[:2], element[:2] + element[2:]]) for element in gt_bboxes]
        gt_bboxes_corner = [np.hstack([element[1], element[0], element[3], element[2]]) for element in gt_bboxes_corner]

        gt_labels = []
        ious = []
        # TODO: might be able to parallelize
        for pred in predictions: #iterating through region proposals
            this_ious = []
            this_gt_labels = []
            for idx, gt_b in enumerate(gt_bboxes_corner): # iterating through groundtruth
                # evaluate the iou
                iou = self.bb_intersection_over_union(gt_b, pred)
                this_ious.append(iou)
                if iou > self.iou_thresh:
                    this_gt_labels.append(self.get_gt_label(gt_noun_classes[idx]))

                    # # we "record" what the label is
                    # if gt_noun_classes[idx] in self.knowns:
                    #     this_gt_labels.append(1)
                    # elif gt_noun_classes[idx] in self.unknowns:
                    #     this_gt_labels.append(2)
                    # else:
                    #     ValueError('gt_bboxes contain other classes than permissible ones. recheck it.')
                else:
                    # we label this pred as the background
                    this_gt_labels.append(0)
            
            ious.append(this_ious)
            gt_labels.append(this_gt_labels)

        ious = np.array(ious)
        # [x for _, x in sorted(zip(Y,X), key=lambda pair: pair[0])]
            
        max_iou_gt = [(max(ious[i]), gt_labels[i][np.argmax(ious[i])]) for i in range(len(ious))]
        max_iou = [element[0] for element in max_iou_gt]
        max_gt = [element[1] for element in max_iou_gt]

        predictions_sorted_idx = [pred for pred, _ in sorted(zip(range(predictions.shape[0]), max_iou), key=lambda x: x[1], reverse=True)]
        max_gt = torch.Tensor(max_gt)
        return predictions_sorted_idx, max_gt

    def check_bbox_validity(self, bboxes, image_width=1920, image_height=1080):

        # rule 1) can't have 0 area, otherwise throw out
        areas = (bboxes[:,2] - bboxes[:,0])*(bboxes[:,3] - bboxes[:,1])
        keep_idx = np.where(areas > 0)
        keep_idx = keep_idx[0]
        # rule 2) if exceed the bounds of the image, shorten the box
        new_bboxes = bboxes[keep_idx, :]
        top_clip = np.minimum(new_bboxes, np.array([image_width,image_height, image_width, image_height]))
        bottom_clip = np.maximum(new_bboxes, np.array([0,0,0,0]))

        return bottom_clip

    def run_tree_encoder(self, x, sample_dict, cache_folder='cache/bounding_box_embeddings'):
        if not os.path.exists(os.path.join(cache_folder, self.purpose)):
            os.makedirs(os.path.join(cache_folder, self.purpose))
        video_id = sample_dict['video_id']
        participant_id = sample_dict['participant_id']
        frame_number = sample_dict['frame_number']
        filename = '{}-{}-{}.npy'.format(participant_id, video_id, frame_number)
        filepath = os.path.join(os.path.join(cache_folder, self.purpose), filename)

        if os.path.exists(filepath):
            # print('Loading embeddings from {}'.format(filepath))
            embeddings = np.load(filepath)
        else:
            x = x.type(torch.FloatTensor).to(self.device)
            results = self.tree_encoder(x)
            embeddings = results['embedding'].detach().cpu().numpy()
            np.save(filepath, embeddings)
        return torch.Tensor(embeddings)

   
    def restrict_to_cached(self, cache_embedding='cache/bounding_box_embeddings', 
                        cache_fmap='cache/bounding_box_feature_map'):

        cache_emb_path = os.path.join(cache_embedding, self.purpose)
        cache_fmap_path = os.path.join(cache_fmap, self.purpose)

        # check that both cache_files exist
        uk_dataset_prime = []
        for sample_dict in tqdm(self.uk_dataset):
            video_id = sample_dict['video_id']
            participant_id = sample_dict['participant_id']
            frame_number = sample_dict['frame_number']

            emb_file = '{}-{}-{}.npy'.format(participant_id, video_id, frame_number)
            fmap_file = '{}-{}-{}.pkl'.format(participant_id, video_id, frame_number)
            emb_path = os.path.join(cache_emb_path, emb_file)
            fmap_path = os.path.join(cache_fmap_path, fmap_file)

            if os.path.exists(emb_path) and os.path.exists(fmap_path):
                # then we can keep this sample
                uk_dataset_prime.append(sample_dict)

        print('Of {} frames, {} are preprocessed and cached in this dataset.'\
            .format(len(self.uk_dataset), len(uk_dataset_prime)))
        self.uk_dataset = uk_dataset_prime

    def only_unique_frames(self):
        unique_frame_samples = []
        unique_frame = set([])
        unique_frame_indices = set([])
        print('Finding unique frames...')
        for idx, sample in enumerate(tqdm(self.uk_dataset)):
            if (sample['participant_id'], sample['video_id'], sample['frame_number']) not in unique_frame:
                unique_frame.add((sample['participant_id'], sample['video_id'], sample['frame_number']))
                unique_frame_samples.append(sample)

        print('Narrowing down from {} frames to {} unique frames'.\
                format(len(self.uk_dataset), len(unique_frame)))

        self.uk_dataset = unique_frame_samples

    def preprocess_and_cache(self):
        # find the unique frames in the knowns and unknowns
        unique_frame = set([])
        unique_frame_indices = set([])
        print('Finding unique frames...')
        for idx, sample in enumerate(tqdm(self.uk_dataset)):
            if (sample['participant_id'], sample['video_id'], sample['frame_number']) not in unique_frame:
                unique_frame.add((sample['participant_id'], sample['video_id'], sample['frame_number']))
                unique_frame_indices.add(idx)
        print('Saving to cache...')
        for idx in tqdm(list(unique_frame_indices)):
            self.__getitem__(idx)
    
    def __len__(self):
        return len(self.uk_dataset)

    def __getitem__(self, idx):

        # get the image at self.data[idx]
        sample = self.uk_dataset[idx]

        video_id = sample['video_id']
        participant_id = sample['participant_id']
        frame_number = sample['frame_number']
        gt_bboxes= sample['bboxes'] # y_h, x_w, h, w
        gt_noun_classes = sample['noun_classes']
        
        # ground truth labels for ground truth bounding boxes, used during evaluation
        gt_labels = [self.get_gt_label(element) for element in gt_noun_classes ]

        file_path = participant_id + '/' + video_id + '/' + ('0000000000' + str(frame_number))[-10:]+'.jpg'
        whole_path = os.path.join(self.image_data_folder, file_path)
        # run input layer (only detections on it)
        if not self.manual_anchors:
            feats = self.inputlayer.get_feature_layer([whole_path], results_format='dictionary')
            bbox_proposals = [element['object_bounding_boxes'] for element in feats][0]
            # x1_w, y1_h, x2_w, y2_h
        else:
            bbox_proposals = np.vstack(self.anchors) 
        # first two elements corresponds to top left
        # second two elements correspodns to bottom = right
        img = cv2.imread(whole_path)
        
        # get labels first, so that we know what ot take out from the 1000 proposals
        # check validity first
        bbox_proposals = self.check_bbox_validity(bbox_proposals)

        # using IOU of every bounding box and every ground truth bounding boxes
        sorted_idx, labels = self.get_labels(bbox_proposals, gt_bboxes, gt_noun_classes)

        # now we're going to cut both embeddings and labels
        if len(sorted_idx) > self.max_num_boxes:
            clipped_sorted_idx = sorted_idx[:min(len(sorted_idx), self.max_num_boxes)]
        else:
            clipped_sorted_idx = sorted_idx + [0]*(self.max_num_boxes - len(sorted_idx))
        if len(gt_bboxes) > self.max_gt_boxes:
            idx = list(range(len(gt_bboxes))) [:self.max_gt_boxes]
        else:
            idx = list(range(len(gt_bboxes))) + [0]*(self.max_gt_boxes - len(gt_bboxes))

        # only keeping around a subset of the bboxes and proposals
        bbox_proposals = bbox_proposals[clipped_sorted_idx]
        labels = labels[clipped_sorted_idx]

        tf_img_minibatch = self.crop_and_transform(bbox_proposals, img, sample)
        # pass it through tree encoder
        # results = self.tree_encoder(tf_img_minibatch)
        embeddings = self.run_tree_encoder(tf_img_minibatch, sample)
        # incase what's loaded is bigger than max_num_boxes
        if len(embeddings)>=self.max_num_boxes:
            embeddings = embeddings[:self.max_num_boxes, ...]
        else:
            raise ValueError('Your cache files need to be regenerated for more bounding boxes')
        # return the embeddings of the tree encoder
        bbox_proposals = torch.Tensor(bbox_proposals)
        gt_bboxes = torch.Tensor(gt_bboxes)[idx]
        gt_bboxes_labels = torch.Tensor(gt_labels)[idx]

        return {'embeddings': embeddings, 'pred_bboxes': bbox_proposals,
                'gt_bboxes': gt_bboxes, 'labels': labels, 
                'gt_bboxes_label': gt_bboxes_labels}


class EK_Dataset_discovery(Dataset):
    def __init__(self, knowns, unknowns, object_data_path, action_data_path, 
                class_key_path, image_data_folder, dataset, inputlayer,
                tree_encoder = 'models/pretraining_tree/framelevel_pred_run0/net_epoch0.pth',
                video_transforms = None,
                video_feat_extract_transforms = None,
                hand_pose_transforms = None,
                hand_bbox_transforms = None,
                embedding_transforms = None,
                label_transforms = None, 
                image_bbox_scaling = 0.5,
                filter_function = default_filter_function, 
                output_cache_folder='dataloader_cache',
                feature_cache_folder='feature_cache',
                feature_cache_overwrite = False,
                device='cuda:0',
                naive_known=True,
                naive_unknown=True,
                snip_threshold = 128, is_video_baseline=False,
                use_resnet101 = False):

        super(EK_Dataset_discovery, self).__init__()
        self.device = device

        self.naive_known = naive_known
        self.naive_unknown = naive_unknown
    
        self.is_video_baseline = is_video_baseline
        # the feature cache folder and overwrite
        self.feature_cache_folder=feature_cache_folder
        if not os.path.exists(self.feature_cache_folder):
            os.makedirs(self.feature_cache_folder, exist_ok=True)

        self.feature_cache_overwrite=feature_cache_overwrite

        # sets of knowns and unknowns
        self.knowns = knowns
        self.unknowns = unknowns
        self.object_data_path = object_data_path
        self.action_data_path = action_data_path
        self.class_key_path = class_key_path
        self.image_data_folder = image_data_folder
        self.filter_function = filter_function
        self.scaling = image_bbox_scaling

        # all transforms
        self.video_transforms = video_transforms
        self.video_feat_extract_transforms = video_feat_extract_transforms
        self.hand_pose_transforms = hand_pose_transforms
        self.hand_bbox_transforms = hand_bbox_transforms
        self.embedding_transforms = embedding_transforms
        self.label_transforms = label_transforms

        # class keys and dictionaries

        if self.device != 'cpu':
            self.class_key_df = pd.read_csv(self.class_key_path)
            self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
            self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

            self.unknowns_id = [self.class_key_dict[element] for element in self.unknowns]
            self.knowns_id = [self.class_key_dict[element] for element in self.knowns]

            
        self.dataset = dataset
        # self.dataset['unknown']
        self.training_data = self.dataset['unknown']
        # now split the absurdly long clips into shorter ones
        self.snip_threshold = snip_threshold
        self.snip_clips() # changes the length of self.training_data
        self.unknown_lowest_level_label = survey_tree(self.knowns)
        
        buffer_ = []
        for idx, sample in enumerate(self.training_data):
            if filter_function(sample):
                buffer_.append(sample)
        self.training_data = buffer_
        del buffer_

        self.f2bbox = self.dataset['known_frame2bbox']

        self.inputlayer = inputlayer

        # initalizing tree encoder
        # if use_resnet101 is false
        if not use_resnet101:
            self.tree_encoder_location = tree_encoder
            self.tree_encoder = torch.load(self.tree_encoder_location)
            self.tree_encoder = self.tree_encoder.to(self.device)
            self.tree_encoder.eval()
            self.tree_encoder_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                    GetResnetFeats()
                                                    ])
        else:
            assert tree_encoder is None, 'Just for safety, set self.tree_encoder_location to None'
            self.resnet101 = GetResnetFeatsGeneral(version='resnet101', mode='vec', device=self.device)

            self.tree_encoder = self.tree_encoder_substitute
            self.tree_encoder_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225])])
    def tree_encoder_substitute(self, x):
        with torch.no_grad():
            res101_embeddings = self.resnet101(x)
        return {'embedding': res101_embeddings.squeeze(-1).squeeze(-1)}

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

    def get_frames(self, participant_id, video_id, start_frame, end_frame):
        a = start_frame
        frames = []
        while a < end_frame:
            file_path = participant_id + '/' + video_id + '/' + ('0000000000' + str(a))[-10:]+'.jpg'
            image = cv2.imread(os.path.join(self.image_data_folder, file_path))

            # reshaping the image by half
            image = cv2.resize(image, (int(image.shape[1]*self.scaling), int(image.shape[0]*self.scaling))) 
            frames.append(image)
            a += 30
        frames = np.stack(frames)
        return frames

    def get_features(self, participant_id, video_id, start_frame, end_frame):
        a = start_frame
        image_paths=[]

        while a < end_frame:
            file_path = participant_id + '/' + video_id + '/' + ('0000000000' + str(a))[-10:]+'.jpg'
            image_paths.append(os.path.join(self.image_data_folder, file_path))
            a += 30

        feats = self.inputlayer.get_feature_layer(image_paths, results_format='dictionary')

        for element in feats:
            element['left_bbox'] *= self.scaling
            element['right_bbox'] *= self.scaling
            element['object_bounding_boxes'] *= self.scaling

        return feats

    def get_unknown_bounding_box(self, participant_id, video_id, start_frame, 
        end_frame, unknown_class, mode='single'):
        
        a = start_frame
        candidates =[]

        while a < end_frame:
            try:
                bboxes = self.f2bbox[participant_id+'/'+video_id+'/'+str(a)]
            except KeyError:
                a += 30
                continue
            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class'] == unknown_class and bbox['bbox']!='[]']
            if len(valid_candidates) == 0:
                a += 30
                continue
            else:
                this_bbox = np.array(ast.literal_eval(valid_candidates[0]['bbox'])[0])
                if len(this_bbox) == 0:
                    a += 30
                    continue
                y, x, yd, xd = this_bbox
                y = int(y * self.scaling)
                x = int(x * self.scaling)
                yd = int(yd * self.scaling)
                xd = int(xd * self.scaling)

                candidates.append(np.array([y, x, yd, xd]))
                a += 30

        if mode == 'single':
            # random selection
            if len(candidates) == 0: # TODO: come up with better solution
                candidates = [np.array([0, 0, int(1080*self.scaling), int(1920*self.scaling)])]

            choice_index = np.random.choice(range(len(candidates)),  1)

            result = candidates[choice_index[0]]
            return result, choice_index
        else:
            result = np.stack(candidates)
            return result, list(range(len(candidates)))

    def crop_and_transform(self, anchor_bboxes, unknown_bbox, frames, idx):
        # choose a bounding box!
        # currently, this method is *very* vanilla
        crop_anchor_frames = []
        # calculate for all boxes
        for timestep, bboxes in enumerate(anchor_bboxes):
            bboxes_reshaped = bboxes.reshape((-1, 4))
            # eliminating extra 0's
            bboxes_reshaped = bboxes_reshaped[np.where(np.sum(bboxes_reshaped, 1))[0], :]
            for element in bboxes_reshaped:


                # MAJOR BUG FROM PREVIOUS VERSION!
                x, y, yd, xd = np.round(element)

                # top_left = element[:2] 
                # bottom_right = element[2:]
                # current_frame = frames[timestep][int(np.round(top_left[0])):int(np.round(bottom_right[0])), 
                #                                 int(np.round(top_left[1])):int(np.round(bottom_right[1])), :]
                current_frame = frames[timestep][int(y):int(y+yd), 
                                                int(x):int(x+xd), :]

                current_frame = current_frame[:,:,[2,1,0]]
                current_frame = self.tree_encoder_transforms(current_frame)
                crop_anchor_frames.append(current_frame)

        if len(unknown_bbox.shape) == 1:
            y, x, yd, xd = np.round(unknown_bbox)
            y, x, yd, xd = int(y), int(x), int(yd), int(xd)
            crop_unknown_frames = frames[idx.item()][y:y+yd, x:x+xd,:]
            crop_unknown_frames = crop_unknown_frames[:,:,[2,1,0]]
            crop_unknown_frames = self.tree_encoder_transforms(crop_unknown_frames)

        else:
            # temp = np.round(unknown_bbox*self.scaling)
            temp=np.round(unknown_bbox)
            y = [int(el) for el in temp[:,0].tolist()]
            x = [int(el) for el in temp[:,1].tolist()]
            yd = [int(el) for el in temp[:,2].tolist()]
            xd = [int(el) for el in temp[:,3].tolist()]

            crop_unknown_frames = []
            for idx, frame in enumerate(frames):
                crop = frame[y[idx]:y[idx]+yd[idx], x[idx]:x[idx]+xd[idx],:]
                crop = crop[:, :, [2,1,0]]
                crop = self.tree_encoder_transforms(crop)
                crop_unknown_frames.append(crop)

            # import pdb; pdb.set_trace()

        return crop_anchor_frames, crop_unknown_frames

    def select_known_bounding_box(self, participant_id, video_id, start_frame, 
        end_frame):
        
        a = start_frame

        candidates = []
        class_count = {element: 0 for element in self.knowns_id}

        # find out the most common class by number of frames of appearance from
        while a < end_frame:
            try:
                bboxes = self.f2bbox[participant_id+'/'+video_id+'/'+str(a)]
            except KeyError:
                # means there are no bounding boxes
                a += 30
                continue

            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class'] in self.knowns_id]

            valid_candidates_classes = tuple(set([bbox['noun_class'] for bbox in bboxes if bbox['bbox']!='[]']))
            
            for element in valid_candidates_classes:
                if element in class_count:
                    class_count[element] += 1

            a += 30
        
        class_count = [(element, class_count[element]) for element in class_count]
        class_count = sorted(class_count, key= lambda x: x[1])
        anchor_known_class = class_count[0][0]

        a = start_frame
        anchor_bboxes = [] # list of the number of timesteps
        while a < end_frame:
            try:
                bboxes = self.f2bbox[participant_id+'/'+video_id+'/'+str(a)]
            except KeyError:
                # means there are no bounding boxes
                # just add a numpy array of 0's
                anchor_bboxes.append(np.zeros((1, 4)))
                a += 30
                continue

            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class']==anchor_known_class]
            if len(valid_candidates) == 0:
                anchor_bboxes.append(np.zeros((1, 4)))
                a += 30
                continue
            else:
                this_frame_anchor_bboxes = []

                for candidate in valid_candidates:
                    if len(ast.literal_eval(candidate['bbox'])) == 0:
                        this_bbox = np.array([0, 0, int(1080*self.scaling), int(1920*self.scaling)])
                    else: 
                        this_bbox = np.array(ast.literal_eval(candidate['bbox'])[0])
                        
                    y, x, yd, xd = this_bbox
                    y = int(y * self.scaling)
                    x = int(x * self.scaling)
                    yd = int(yd * self.scaling)
                    xd = int(xd * self.scaling)

                    # MAJOR BUG FROM PREVIOUS VERSION! OTHER END IS EXPECTING TOPLEFT-BOTTOMRIGHT format.
                    this_frame_anchor_bboxes.append([y, x, yd, xd])

                anchor_bboxes.append(np.array(this_frame_anchor_bboxes))


        return anchor_bboxes, anchor_known_class



    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        """
        Returns:
            a dictionary with keys 'handpose', 'handbbox', 'frames', 'unknown', 'known'
            'hierarchy_encoding'
            
        """

        if self.device == 'cpu':
            self.class_key_df = pd.read_csv(self.class_key_path)
            self.class_key_dict = dict(zip(self.class_key_df.class_key, self.class_key_df.noun_id))
            self.noun_dict = dict(zip(self.class_key_df.noun_id, self.class_key_df.class_key))

            self.unknowns_id = [self.class_key_dict[element] for element in self.unknowns]
            self.knowns_id = [self.class_key_dict[element] for element in self.knowns]

        times = [('start_time', time.time())]

        sample = self.training_data[idx]
        video_id = sample['video_id']
        participant_id = sample['participant_id']
        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        noun_class = sample['noun_class']

        # INPUTS TO MODEL
        # get the frames, uncroppped
        # pass through video transforms
        RGB_frames = self.get_frames(participant_id, video_id, start_frame, end_frame)
        times.append(("RGB_frames_time", time.time()))

        # get the hand features, handpose and boundingbox information
        # pass through hand transforms
        cache_filename = sample['participant_id'] + '_' + sample['video_id'] \
                        + '_' + str(sample['start_frame']) + '_' + str(sample['end_frame']) + '.pkl'
        cache_path = os.path.join(self.feature_cache_folder, cache_filename)

        if os.path.exists(cache_path) and not self.feature_cache_overwrite:
            with open(cache_path, 'rb') as f:
                feats = pickle.load(f)
        else:
            feats = self.get_features(participant_id, video_id, start_frame, end_frame)
            with open(cache_path, 'wb') as f:
                pickle.dump(feats, f)

        times.append(('get_hand_features_time', time.time()))
        handpose = np.stack([np.hstack([element['left_pose'], element['right_pose']]) for element in feats])
        handbbox = np.stack([np.hstack([element['left_bbox'], element['right_bbox']]) for element in feats])
        
        if self.hand_pose_transforms is not None:
            handpose = self.hand_pose_transforms(handpose)
            handpose = torch.Tensor(handpose)
        if self.hand_bbox_transforms is not None:
            handbbox = self.hand_bbox_transforms(handbbox)
            handbbox = torch.Tensor(handbbox)
        times.append(('hand_feature_time', time.time()))

        # get ground truth unknown bounding box
        if self.naive_unknown:
             # get detected bounding boxes
            unknown_bounding_boxes, index = self.get_unknown_bounding_box(
                            participant_id, video_id, start_frame, end_frame,
                            noun_class, mode='single' if not self.is_video_baseline else 'all' )

            # get embeddings through g
            # crop the images, push each through the transforms
            times.append(('unknown_bounding_boxes_time', time.time()))
            if self.naive_known:
                bounding_boxes, anchor_known_class = \
                                self.select_known_bounding_box(participant_id,
                                    video_id, start_frame, end_frame)
                times.append(('unknown_bounding_boxes_time', time.time()))

            else:
                # MAJOR BUG FIX FROM PREVIOUS VERSION
                # bounding_boxes = [element['object_bounding_boxes'][:2]  for element in feats]
                bounding_boxes = [np.hstack([element['object_bounding_boxes'][:2], (element['object_bounding_boxes'][2:] - element['object_bounding_boxes'][:2])]) for element in feats]
            crop_anchor_frames, crop_unknown_frames = self.crop_and_transform(
                    bounding_boxes, unknown_bounding_boxes, RGB_frames, index)
            times.append(('crop_and_transform_time', time.time()))
            if type(crop_unknown_frames) is not list or (type(crop_unknown_frames) is list and len(crop_unknown_frames) == 1):
                bundle_crop_frames = torch.stack(crop_anchor_frames + [crop_unknown_frames])
            elif not self.is_video_baseline:
                bundle_crop_frames = torch.stack(crop_anchor_frames + crop_unknown_frames)
            else:
                bundle_crop_frames = torch.stack(crop_unknown_frames)

            a = 0

            embeddings = []
            while a < bundle_crop_frames.shape[0]:
                this_bundle = bundle_crop_frames[a:min(a+16, bundle_crop_frames.shape[0]), ...]
                this_bundle = this_bundle.type(torch.FloatTensor).to(self.device)

                these_results = self.tree_encoder(this_bundle)
                if not self.is_video_baseline:
                    embeddings.append(these_results['embedding'].data)
                else:
                    embeddings.append(torch.stack([torch.argmax(these_results['tree_level_pred1'][:, :3], 1),
                                        torch.argmax(these_results['tree_level_pred2'], 1),
                                        torch.argmax(these_results['tree_level_pred3'], 1)]).t())
                a += 16


            # naive protection againstscases when crop_anchor_frames is empty
            if not self.is_video_baseline and len(crop_anchor_frames) == 0:
                filler = torch.zeros_like(embeddings[-1]).to(self.device)
                embeddings.insert(0, filler)

            embeddings = torch.cat(embeddings, 0)

            if self.embedding_transforms is not None:
                embeddings = self.embedding_transforms(embeddings)
            times.append(('visual_encoder_time', time.time()))

            if not self.is_video_baseline:
                anchor_embeddings = embeddings[:-1, ...]
                unknown_embeddings = embeddings[-1]
            else:
                anchor_embeddings = torch.zeros((1,1))

                unknown_embeddings = embeddings
                # take the majority vote
                unknown_embeddings = stats.mode(unknown_embeddings.detach().cpu().numpy())[0][0] # first indexing to only get the modes,  second indexing for squeezing
                unknown_embeddings = torch.Tensor(unknown_embeddings)

        # transform RGB_frames
        if self.video_transforms is not None:
            RGB_frames = self.video_transforms(RGB_frames)
            feat_extract_input = sample.copy()
            feat_extract_input['RGB'] = RGB_frames
            RGB_frames = self.video_feat_extract_transforms(feat_extract_input)
            RGB_frames = torch.Tensor(RGB_frames)
            RGB_frames = RGB_frames.squeeze(dim=0)
            times.append(('video_transform_time', time.time()))

        # getting encojding
        hierarchy_encoding = get_tree_position(self.noun_dict[noun_class], self.knowns)
        hierarchy_encoding = torch.Tensor(hierarchy_encoding)
        if self.label_transforms is not None:
            hierarchy_encoding = self.label_transforms(hierarchy_encoding)
        times.append(('hierarchy_encoding_time', time.time()))
        if self.naive_unknown:

            results_dict = {'handpose': handpose, # [B] * T * 126
                            'handbbox': handbbox, # [B] * T * 8
                            'frames': RGB_frames, # [B] * 1024 * T'
                            'unknown': unknown_embeddings, # [B] * 10
                            'known': anchor_embeddings, # [B] * num_known * 10
                            'hierarchy_encoding': hierarchy_encoding # [B] * 3
                            }
        delta_times = []
        for i in range(1, len(times)):
            delta_times.append((times[i][0], times[i][1] - times[i-1][1]))
        # import pdb; pdb.set_trace()
        # print(delta_times)

        return results_dict


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
            if image is None:
                a += 30 * skip_interval
                print('{} not found, skipping.'.format(image_path))
                continue
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
            if image is None:
                a += 30 * skip_interval
                print('{} not found, skipping.'.format(image_path))
                continue
            # resizing the image
            image = cv2.resize(image, tuple([int(dim*scale) for dim in image.shape][:2][::-1]))

            valid_candidates = [bbox for bbox in bboxes if bbox['noun_class']==sample_dict['noun_class']]
            if len(valid_candidates)==0 or valid_candidates[0] == '[]':
                a+=30 * skip_interval
                continue
            else:
                # this only takes the first bounding box corresponding to that class!
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
        frames = np.stack(frames, axis=3) # T x W x H x C 

        np.save(os.path.join(cache_dir, cache_filename), frames)
        create_config_file(threshold, processed_frame_number, cache_dir=cache_dir)
    return frames

class EK_Dataset_pretrain_framewise_prediction(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            model_saveloc,
            validation_num_samples=200,
            filter_function = default_filter_function,
            mode='resnet', output_cache_folder='dataloader_cache/', 
            snip_threshold=32,
            crop_type='rescale', 
            prune_target='known_pretrain',
            resnet_out=True):

        super(EK_Dataset_pretrain_framewise_prediction, self).__init__()

        self.image_data_folder = image_data_folder 
        self.knowns = knowns
        self.unknowns = unknowns
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

        if resnet_out:
            self.image_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                GetResnetFeats()
                                                ])
        else:
            self.image_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
            # GetResnetFeatsGeneral(version=resnet_version, mode=resnet_mode)

        self.DF = DatasetFactory(knowns, unknowns,
                    object_data_path, action_data_path, class_key_path)
        self.dataset = self.DF.get_dataset()

        # self. dataset with clips -> self.dataset with frames
        self.unknown_lowest_level_label = survey_tree(self.knowns)

        self.frame_data = self.prune_to_frames(target=prune_target)
        random.shuffle(self.frame_data)
        if self.val_num_samples == 0:
            self.train_frame_set = self.frame_data
            self.val_frame_set = []
        else:
            self.train_frame_set = self.frame_data[:-self.val_num_samples]
            self.val_frame_set = self.frame_data[-self.val_num_samples:] 
        if model_saveloc is not None: 
            with open(os.path.join(model_saveloc, 'processing_params.pkl'),'wb') as f:
                pickle.dump({'output_cache_fullpath': self.output_cache_fullpath, 
                            'crop_type': self.crop_type, 
                            'f2bbox': self.dataset['known_frame2bbox'],
                            'image_data_folder': self.image_data_folder, 
                            'noun_dict': self.noun_dict, 
                            'knowns': self.knowns, 
                            'unknowns': self.unknowns, 
                            'unknown_lowest_level_label': self.unknown_lowest_level_label, 
                            'image_transform': self.image_transforms
                            }, f )

    def prune_to_frames(self, target='known_pretrain'):
        # check that 1: has a bounding box of the known class
        frames2bbox = self.dataset['known_frame2bbox' if target in ('known_pretrain', 'known') else 'unknown_frame2bbox']

        frames = []
        for clip in self.dataset[target]:
            vid_id = clip['video_id']
            part_id = clip['participant_id']
            start_frame = clip['start_frame']
            end_frame = clip['end_frame']
            
            a = int(start_frame)
            while a <= int(end_frame):
                # process
                key = '/'.join([part_id, vid_id, str(a)])
                if key in frames2bbox:
                    # check that the frame contains a bounding box of the correspoonding class
                    candidates = [bbox for bbox in frames2bbox[key] if bbox['noun_class']==clip['noun_class']]
                    # add bounding 
                    for element in candidates:
                        this_bbox = ast.literal_eval(element['bbox'])
                        if len(this_bbox) > 0:
                            # save: vid_id, part_id, a, this_bbox, class
                            frames.append({'video_id': vid_id,
                                            'participant_id': part_id,
                                            'frame_number': a,
                                            'bbox': this_bbox,
                                            'noun_class': element['noun_class']})
                a += 30

        return frames

    @staticmethod
    def process(sample, noun_dict, image_transforms, image_data_folder, known_classes, scaling = 0.5):
        # rescale crop implemented right here
        this_bbox = sample['bbox']
        assert len(this_bbox) >0,  'this bounding box is empty, and it is not supposed to be'

        # load the image
        file_path = sample['participant_id'] + '/' + sample['video_id'] + '/' + ('0000000000' + str(sample['frame_number']))[-10:]+'.jpg'
        image_path = os.path.join(image_data_folder, file_path)
        image = cv2.imread(image_path)

        y, x, yd, xd = this_bbox[0]
        rescaled_crop = cv2.resize(image[y:y+yd, x:x+xd, :], 
                            tuple([int(dim*scaling) for dim in image.shape[:2][::-1]]))
        encoding = get_tree_position(noun_dict[sample['noun_class']], known_classes)

        d = {'frame': rescaled_crop, # making rgb
            'noun_label': noun_dict[sample['noun_class']],
            'hierarchy_encoding': encoding
            }

        # BGR2RGB -> transforms.ToPILImage -> transforms.Resize((224, 224)) -> normalize -> resnet
        d['frame'] = image_transforms(d['frame'][:,:,[2,1,0]])
        d['hierarchy_encoding'] = torch.from_numpy(d['hierarchy_encoding'])

        return d

    def get_val_dataset(self):
        # just save valset. 

        val_set = []

        for frame in self.val_frame_set:
            val_set.append(EK_Dataset_pretrain_framewise_prediction.process(
                                    frame, self.noun_dict, self.image_transforms, 
                                    self.image_data_folder, self.knowns))

        return val_set
    
    def __len__(self):
        return len(self.train_frame_set)

    def __getitem__(self, idx):
        sample = self.train_frame_set[idx]
        processed_sample = EK_Dataset_pretrain_framewise_prediction.process(
            sample, self.noun_dict, self.image_transforms,
            self.image_data_folder, self.knowns)
        return processed_sample


class EK_Dataset_pretrain_batchwise(Dataset):
    def __init__(self, knowns, unknowns,
            object_data_path,
            action_data_path,
            class_key_path,
            image_data_folder,
            model_saveloc,
            batch_size = 8,
            training_num_batches=1000,
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
        self.train_num_batches = training_num_batches
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

        selector = Selector(self.training_data, option=sampling_mode, train_ratio=selector_train_ratio)
        self.rand_selection_indices = selector.get_minibatch_indices('train', self.batch_size, self.train_num_batches)
        self.val_indices = selector.get_minibatch_indices('val', self.batch_size, 40)

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
                        'batchwise_transform': self.batchwise_transform}, f )

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
    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    # with open('current_split.pkl','rb') as f:
    #     split = pickle.load(f)
    knowns = split['training_known']
    unknowns = split['training_unknown']

    train_ratio=0.51
    DF = DatasetFactory(knowns, unknowns, train_object_csvpath, train_action_csvpath, class_key_csvpath)
    dataset = DF.get_dataset()
    dataset2 = DF.get_dataset()

    random.shuffle(dataset['unknown'])
    training_unknown = dataset['unknown'][: int(train_ratio*len(dataset['unknown']))]
    training_known = dataset['known'][: int(train_ratio*len(dataset['known']))]
    training_dataset = dataset.copy()
    training_dataset['unknown'] = training_unknown 
    training_dataset['known'] = training_known

    validation_unknown = dataset2['unknown'][-int((1-train_ratio)*len(dataset2['unknown'])):]
    validation_known = dataset2['known'][-int((1-train_ratio)*len(dataset2['known'])):]
    validation_dataset = dataset2.copy()
    validation_dataset['unknown'] = validation_unknown
    validation_dataset['known'] = validation_known

    # train predictor for 3 classes: not object, known, unknown.
    inputlayer_rpn = InputLayer(max_num_boxes=None, rpn_conf_thresh=0.0, rpn_only=True)

    DF_train = EK_Dataset_detection(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, 
            image_data_folder, training_dataset, inputlayer_rpn,
            tree_encoder=None, 
            purpose='resnet101', max_num_boxes=50, max_gt_boxes=6)
    import ipdb; ipdb.set_trace()
    DF_train[2000]
    # DF_train.preprocess_and_cache()
    import ipdb; ipdb.set_trace()

    DF_val = EK_Dataset_detection(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, 
            image_data_folder, validation_dataset, inputlayer_rpn,
            tree_encoder=None, 
            purpose='resnet101', max_num_boxes=50, max_gt_boxes=6)
    DF_val.preprocess_and_cache()

    # for i in range(4):
    #     print(DF_pretrain[112+i])
    
    
