"""
Pretraining of the autoencoder for the video data to embed into a latent space
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

from src.tree_encoder import *
from data.EK_dataloader import EK_Dataset, EK_Dataset_pretrain, EK_Dataset_pretrain_pairwise 
from data.gt_hierarchy import *
from data.transforms import *

from tqdm import tqdm

# dataloaders

DEBUG = True
USECUDA = True 
MODE = 'pairwise'

def pretrain_pairwise(net, dataloader, num_epochs=10, save_interval=1,
        model_saveloc='models/pretraining_pairwise'):
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), 0.01)

    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))
        for i, sample in enumerate(dataloader):
            print('on batch {}'.format(i))
            frames_a = sample['frames_a']
            frames_b = sample['frames_b']
            tree_distance = sample['dist']

            if USECUDA:
                frames_a = frames_a.type(torch.FloatTensor).to('cuda:0')
                frames_b = frames_b.type(torch.FloatTensor).to('cuda:0')
                tree_distance = tree_distance.type(torch.FloatTensor).to('cuda:0')
                net = net.to('cuda:0')

            optimizer.zero_grad()
            encoding_a = net(frames_a)
            encoding_b = net(frames_b)
            loss= criterion(torch.sqrt(torch.diag(torch.matmul(
                                            encoding_a - encoding_b,
                                            (encoding_a - encoding_b).t()))),
                            tree_distance)
            loss.backward()
            optimizer.step()
        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))

            torch.save(net, os.path.join(model_saveloc,
                'net_epoch{}.pth'.format(epoch)))
    return net

def pretrain(net, dataloader, num_epochs=10, save_interval=1, 
        model_saveloc='models/pretraining_single'):
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc)
    # define cost
    criterion = nn.MSELoss() 
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), 0.01)
    loss_per_sample = []
    # TODO iterate through the dataset, 10 epochs
    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))
        for i, sample in enumerate(dataloader):
            print('on batch {}'.format(i))
            frames = sample['frames']
            encoding = sample['hierarchy_encoding']
            if USECUDA:
                frames = frames.type(torch.FloatTensor).to('cuda:0')
                encoding = encoding.type(torch.FloatTensor).to('cuda:0')
                net = net.to('cuda:0')
            optimizer.zero_grad()
            pred_encoding = net(frames)
            loss = criterion(pred_encoding, encoding)
            loss.backward()
            # saving loss
            
            optimizer.step()

        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            # saving to the location
            torch.save(net, os.path.join(model_saveloc, 
                                        'net_epoch{}.pth'.format(epoch)))

    return net

if __name__=='__main__':
    # Setting up the paths
    dataset_path = '/vision/group/EPIC-KITCHENS/'
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

    image_normalized_dimensions = (150, 200)
    time_normalized_dimension = 16

    # splitting known and unknown data
    if DEBUG:
        if not os.path.exists('current_split.pkl'):
            split = get_known_unknown_split()
            # saving into 'current_split.pkl'
            with open('current_split.pkl', 'wb') as f:
                pickle.dump(split, f)

        else:
            # loading from the file
            with open('current_split.pkl', 'rb') as f:
                split = pickle.load(f)
    else:
        split = get_known_unknown_split()
    knowns = split['training_known']
    unknowns = split['training_unknown']
    # instantiating the dataloader
    composed_trans_indiv = transforms.Compose([Rescale(image_normalized_dimensions),
                                        Transpose(),
                                        TimeNormalize(time_normalized_dimension),
                                        ToTensor()])
    composed_trans_pair = transforms.Compose([ToTensor()])

    if MODE == 'individual':
        DF = EK_Dataset_pretrain(knowns, unknowns,
                train_object_csvpath, train_action_csvpath,
                class_key_csvpath, image_data_folder, 
                processed_frame_number=time_normalized_dimension,  
                transform=composed_trans_indiv) 
        train_dataloader = data.DataLoader(DF, batch_size=4, num_workers=0)
                                
        # model instatntiation and training
        model = C3D(input_shape=(3, time_normalized_dimension, image_normalized_dimensions[0] , image_normalized_dimensions[1]), 
                    embedding_dim=3) # TODO: replace these
        pretrain(model, train_dataloader, num_epochs=30)

    elif MODE == 'pairwise':
        DF = EK_Dataset_pretrain_pairwise(knowns, unknowns,
                train_object_csvpath, train_action_csvpath, 
                class_key_csvpath, image_data_folder,
                processed_frame_number=time_normalized_dimension, 
                individual_transform=composed_trans_indiv, 
                pairwise_transform=composed_trans_pair
                ) 
        train_dataloader = data.DataLoader(DF, batch_size=4, num_workers=0)

        model = C3D(input_shape=(3, time_normalized_dimension, image_normalized_dimensions[0] , image_normalized_dimensions[1]), 
                    embedding_dim=8) # TODO: replace these
        pretrain_pairwise(model, train_dataloader, num_epochs=30)
    else:
        raise ValueError('invalid mode')
