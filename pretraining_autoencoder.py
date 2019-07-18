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

import json
from tqdm import tqdm

# dataloaders

DEBUG = True
USECUDA = True 
MODE = 'pairwise'

def pretrain_pairwise(net, dataloader, num_epochs=10, save_interval=1, lr = 0.01, 
        model_saveloc='models/pretraining_tree/pairwise'):
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))
        for i, sample in enumerate(tqdm(dataloader)):
            # print('on batch {}'.format(i))
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

def pretrain(net, dataloader, num_epochs=10, save_interval=1, lr=0.01,
        model_saveloc='models/pretraining_tree/single'):
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)
    # define cost
    criterion = nn.MSELoss() 
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss_per_sample = []
    # TODO iterate through the dataset, 10 epochs
    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))
        for i, sample in enumerate(tqdm(dataloader)):
            # print('on batch {}'.format(i))
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
            loss_per_sample.append(list(loss.detach().cpu().numpy())[0])
            optimizer.step()

        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            # updating the loss file
            with open(os.path.join(model_saveloc, 'netloss_epoch{}.pth'.format(epoch)), 
                'wb') as f:
                pickle.dump(loss_per_sample, f)
            # saving to the location
            torch.save(net, os.path.join(model_saveloc, 
                                        'net_epoch{}.pth'.format(epoch)))

    return net

def save_training_config(path, args):
    config_dict = {'num_epochs': args.epochs,
                    'rescale_imwidth': args.rescale_imwidth,
                    'rescale_imheight': args.rescale_imheight,
                    'time_normalized_dimension': args.time_normalized_dimension,
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'run_num': args.run_num,
                    'embedding_dimension': args.embedding_dim}
    with open(path, 'w') as f:
        json.dump(config_dict, f)
    

if __name__=='__main__':

    # argparse
    parser = argparse.ArgumentParser(description='PyTorch hierarchy embedding pretraining')
    parser.add_argument('--data', type=str, default='/vision/group/EPIC-KITCHENS',
                        help='path to dataset directory')
    parser.add_argument('--model_folder', type=str, default='models/pretraining_tree',
                        help='path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--rescale_imwidth', type=int, default = 200, 
                        help='width of the image after rescaling')
    parser.add_argument('--rescale_imheight', type=int, default = 150,
                        help='height of the image after rescaling')
    parser.add_argument('--time_normalized_dimension', type=int, default = 16,
                        help='number of timesteps for which the normalization is done')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--use-cuda', default=False, type=bool, metavar='C',
                        help='use cuda')
    parser.add_argument('--run_num', type=int, default=0, 
                        help='the run number that will be included in all output files')
    parser.add_argument('--embedding_dim', type=int, default=3,
                        help='the dimensionality of the embedding, output from the tree encoder')
    args = parser.parse_args()

    # Setting up the paths
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

    image_normalized_dimensions = (args.rescale_imheight, args.rescale_imwidth)
    time_normalized_dimension = args.time_normalized_dimension

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
        model_saveloc = os.path.join(args.model_folder, 'individual_run{}'.format(args.run_num))
        if not os.path.exists(model_saveloc):
            os.makedirs(model_saveloc, exist_ok=True)

        save_training_config(os.path.join(model_saveloc, 'config.json'), args)

        DF = EK_Dataset_pretrain(knowns, unknowns,
                train_object_csvpath, train_action_csvpath,
                class_key_csvpath, image_data_folder, 
                processed_frame_number=time_normalized_dimension,  
                transform=composed_trans_indiv) 
        train_dataloader = data.DataLoader(DF, batch_size=args.batch_size, num_workers=0)
                                
        # model instatntiation and training
        model = C3D(input_shape=(3, time_normalized_dimension, image_normalized_dimensions[0] , image_normalized_dimensions[1]), 
                    embedding_dim=args.embedding_dim) # TODO: replace these
        pretrain(model, train_dataloader, num_epochs=args.epochs, model_saveloc=model_saveloc)

    elif MODE == 'pairwise':
        model_saveloc = os.path.join(args.model_folder, 'pairwise_run{}'.format(args.run_num))
        if not os.path.exists(model_saveloc):
            os.makedirs(model_saveloc, exist_ok=True)

        save_training_config(os.path.join(model_saveloc, 'config.json'), args)

        DF = EK_Dataset_pretrain_pairwise(knowns, unknowns,
                train_object_csvpath, train_action_csvpath, 
                class_key_csvpath, image_data_folder,
                processed_frame_number=time_normalized_dimension, 
                individual_transform=composed_trans_indiv, 
                pairwise_transform=composed_trans_pair
                ) 
        train_dataloader = data.DataLoader(DF, batch_size=args.batch_size, num_workers=0, num_samples=1000)

        model = C3D(input_shape=(3, time_normalized_dimension, image_normalized_dimensions[0] , image_normalized_dimensions[1]), 
                    embedding_dim=args.embedding_dim) # TODO: replace these
        pretrain_pairwise(model, train_dataloader, num_epochs=args.epochs, model_saveloc=model_saveloc)
    else:
        raise ValueError('invalid mode')
