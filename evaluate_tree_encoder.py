"""
Scripts for evaluating the tree embeddings
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from torch.utils import data
import torch.optim as optim
from torchvision import transforms

from src.hierarchical_loss import *
from src.model import *
from src.input_layers import InputLayer

from data.EK_dataloader import *
from data.gt_hierarchy import *
from data.transforms import *
from tqdm import tqdm 

def evaluate_embeddings(model, classes, dataset):
    
    
    pass

def evaluate_encoding_model(model, dataset_path):

    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch hierarchy discovery model training')
    parser.add_argument('--data', type=str, default='/vision/group/EPIC-KITCHENS',
                        help='path to dataset directory')
    parser.add_argument('--model_folder', type=str, 
                        default='models/hierarchy_discovery/',
                        help='path to dataset directory')
    parser.add_argument('--run_number', type=int, default=0,
                        help='run number corresponding to the version of the model')
    parser.add_argument('--visual_encoder_path', type=str, 
                        default='models/pretraining_tree/framelevel_pred_run9/net_epoch99.pth',
                        help='path to tree encoder')

    args = parser.parse_args()

    # get the model you want to work with
    dataset_path = args.data
    annotations_foldername = 'annotations'
    annotations_folderpath = os.path.join(dataset_path, annotations_foldername)
    visual_dataset_path = os.path.join(dataset_path, 'EPIC_KITCHENS_2018.Bingbin')
    visual_images_foldername = 'object_detection_images'
    visual_images_folderpath = os.path.join(visual_dataset_path, visual_images_foldername)
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

    import pdb; pdb.set_trace()
    # get the classes that you wanna test on
    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    train_knowns = split['training_known']
    train_unknowns = split['training_unknown']
    test_knowns = list(set(train_knowns + train_unknowns))
    test_unknowns = split['testing_unknown']

    # get the dataset
    import pdb; pdb.set_trace()
    DF = EK_Dataset_pretrain_framewise_prediction(knowns, unknowns,
                train_object_csvpath, train_action_csvpath, 
                class_key_csvpath, image_data_folder,
                model_saveloc,
                crop_type='rescale',
                mode='resnet',
                validation_num_samples=0)
    # define the criteria by which it's measured

    # run the evaluation on baseline

    # run the evaluation on our model

    pass