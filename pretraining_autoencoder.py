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
from src.ltfb import *
from src.lifted_structure_loss import *

from data.EK_dataloader import EK_Dataset, EK_Dataset_pretrain, EK_Dataset_pretrain_pairwise, EK_Dataset_pretrain_batchwise
from data.gt_hierarchy import *
from data.transforms import *

import json
from tqdm import tqdm

# dataloaders

DEBUG = False 
USECUDA = True 
MODE = 'batchwise'
USERESNET = False # True

def pretrain_batchwise(net, dataloader, valset, optimizer_type='sgd', num_epochs=10, save_interval=1, lr=0.01,
        model_saveloc='models/pretraining_tree/pairwise'):

    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)
    
    criterion = HierarchicalLiftedStructureLoss(8, 'gpu' if USECUDA else 'cpu')
    if optimizer_type=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_per_sample = []
    val_losses_per10 = []
    counter = 0

    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))

        for i, sample in enumerate(tqdm(dataloader)):
            # sample shape: [MINI_BATCH_SIZE, BATCH_SIZE, 512, TIMESTEPS, 7, 7]
            # what this script is used to: [BATCH_SIZE, 512, TIMESTEPS, 7, 7]
            mini_batch_size = len(sample['batch_frames'])
            batch_size = sample['batch_frames'][0].shape[0]

            batch_stacked = torch.cat(sample['batch_frames'])

            # print('on batch {}'.format(i))
            tree_distance = sample['dist_matrix']

            if USECUDA:
                batch_stacked = batch_stacked.type(torch.FloatTensor).to('cuda:0')
                tree_distance = tree_distance.type(torch.FloatTensor).to('cuda:0')
                net = net.to('cuda:0')

            optimizer.zero_grad()
            encodings = net(batch_stacked)
            encodings = encodings.reshape((batch_size, mini_batch_size, -1))
            
            loss = criterion(encodings,tree_distance)

            loss_per_sample.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()
            # import ipdb; ipdb.set_trace()

            
            # if counter % 10 == 0: 
            #     with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
            #                 'wb') as f:
            #         pickle.dump(loss_per_sample, f)
                
            #     # validate on valset 
            #     val_losses = []
            #     for val_sample in valset:
            #         val_batch_stacked = torch.stack(val_sample['batch_frames'])
            #         # val_batch_size = val_batch_stacked.shape[0]
            #         # val_mini_batch_size = val_batch_stacked.shape[1]

            #         # val_batch_stacked = val_batch_stacked.reshape([val_batch_size*val_mini_batch_size]+list(val_batch_stacked.shape[2:]))
            #         val_tree_distance = val_sample['dist_matrix']
            #         if USECUDA:
            #             val_batch_stacked = val_batch_stacked.type(torch.FloatTensor).to('cuda:0')
            #             val_tree_distance = val_tree_distance.type(torch.FloatTensor).to('cuda:0')
            #             net = net.to('cuda:0')
            #         with torch.no_grad():
            #             val_encodings = net(val_batch_stacked)
            #             # val_encodings = val_encodings.reshape((val_batch_size, val_mini_batch_size, -1))
            #             val_encodings = val_encodings.unsqueeze(0)

            #             val_loss = criterion(val_encodings, val_tree_distance)
            #             val_losses.append(val_loss.data.cpu().numpy())
            #     val_losses_per10.append(np.mean(val_losses))
                
            #     with open(os.path.join(model_saveloc, 'validation_losses.pkl'.format(epoch)), 
            #                 'wb') as f:
            #         pickle.dump(val_losses_per10, f)
            #     # import ipdb; ipdb.set_trace()

            counter += 1

        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                'wb') as f:
                pickle.dump(loss_per_sample, f)
            torch.save(net, os.path.join(model_saveloc,
                'net_epoch{}.pth'.format(epoch)))
    return net

def pretrain_pairwise(net, dataloader, valset, optimizer_type='sgd', num_epochs=10, save_interval=1, lr = 0.01, 
        model_saveloc='models/pretraining_tree/pairwise'):
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)

    criterion = nn.MSELoss()
    if optimizer_type=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    counter = 0
    loss_per_sample = []
    val_losses_per10 = []
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
            
            loss = criterion(F.pairwise_distance(encoding_a, encoding_b),tree_distance)

            loss_per_sample.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

            if counter % 10 == 0: 
                with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                            'wb') as f:
                    pickle.dump(loss_per_sample, f)
                
                
                
                # validate on valset 
                val_losses = []
                for val_sample in valset:

                    val_frames_a = torch.stack([element['frames_a'] for element in val_sample])
                    val_frames_b = torch.stack([element['frames_b'] for element in  val_sample])
                    val_tree_distance = torch.stack([element['dist'] for element in val_sample])
                    if USECUDA:
                        val_frames_a = val_frames_a.type(torch.FloatTensor).to('cuda:0')
                        val_frames_b = val_frames_b.type(torch.FloatTensor).to('cuda:0')
                        val_tree_distance = val_tree_distance.type(torch.FloatTensor).to('cuda:0')
                        net = net.to('cuda:0')
                    with torch.no_grad():
                        val_encoding_a = net(val_frames_a)
                        val_encoding_b = net(val_frames_b)
                        val_loss = criterion(F.pairwise_distance(val_encoding_a, val_encoding_b),
                                val_tree_distance)
                        val_losses.append(val_loss.data.cpu().numpy())
                val_losses_per10.append(np.mean(val_losses))
                
                with open(os.path.join(model_saveloc, 'validation_losses.pkl'.format(epoch)), 
                            'wb') as f:
                    pickle.dump(val_losses_per10, f)

            counter += 1

        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                'wb') as f:
                pickle.dump(loss_per_sample, f)
            torch.save(net, os.path.join(model_saveloc,
                'net_epoch{}.pth'.format(epoch)))
    return net

def pretrain(net, dataloader, optimizer_type='sgd', num_epochs=10, save_interval=1, lr=0.01,
        model_saveloc='models/pretraining_tree/single'):
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)
    # define cost
    criterion = nn.MSELoss() 
    # optimizer
    
    if optimizer_type=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_per_sample = []
    counter = 0
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
            
            if counter % 10 == 0: 
                with open(os.path.join(model_saveloc, 'training_losses.pkl'),
                            'wb') as f:
                    pickle.dump(loss_per_sample, f)
            counter += 1

        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            # updating the loss file
            with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                'wb') as f:
                pickle.dump(loss_per_sample, f)
            # saving to the location
            torch.save(net, os.path.join(model_saveloc, 
                                        'net_epoch{}.pth'.format(epoch)))

    return net

def save_training_config(path, args, knowns, num_samples=None):
    config_dict = { 'run_num': args.run_num,
                    'max_training_knowns': args.max_training_knowns,
                    'known_classes': ','.join(knowns),
                    'model_mode': args.model_mode,
                    'num_epochs': args.epochs,
                    'num_samples': args.num_samples if num_samples is None else num_samples,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'rescale_imwidth': args.rescale_imwidth,
                    'rescale_imheight': args.rescale_imheight,
                    'time_normalized_dimension': args.time_normalized_dimension,
                    'embedding_dimension': args.embedding_dim,
                    'crop_mode': args.crop_mode,
                    'feature_extractor': args.feature_extractor,
                    'optimizer_type': args.optimizer,
                    'sampler_mode': args.sampler_mode,
                    'selector_train_ratio': args.selector_train_ratio}
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
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='The number of pairwise samples, applicable only if the mode is "pairwise"')
    parser.add_argument('--crop_mode', type=str, default='blackout',
                        help='The mode for the crop done around the object of interest. Can either be blackout or rescale.')
    parser.add_argument('--max_training_knowns', type=int, default=8,
                        help='Effectively the number of known classes in the pretraining.')
    parser.add_argument('--model_mode', type=str, default='simple',
                        help='complexity of the model')
    parser.add_argument('--feature_extractor', type=str, default='None',
                        help='image-level feature extraction')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='type of optimizer to use')
    parser.add_argument('--sampler_mode', type=str, default='equality',
                        help='type of sampling technique.')
    parser.add_argument('--selector_train_ratio', type=float, default=0.75,
                        help='Ratio of possible clips used for training')
    # parser.add_argument('--ltfb_stack_num', type=int, default=2,
    #                     help='number of layers in ltfb, if model_mode is ltfb')
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

    assert args.optimizer == 'sgd' or args.optimizer == 'adam'

    # splitting known and unknown data
    if DEBUG:
        if not os.path.exists('current_split.pkl'):
            split = get_known_unknown_split(max_training_knowns=args.max_training_knowns)
            # saving into 'current_split.pkl'
            with open('current_split.pkl', 'wb') as f:
                pickle.dump(split, f)

        else:
            # loading from the file
            with open('current_split.pkl', 'rb') as f:
                split = pickle.load(f)
    else:
        split = get_known_unknown_split(max_training_knowns=args.max_training_knowns)

    knowns = split['training_known']
    unknowns = split['training_unknown']
    # instantiating the dataloader
    
    # TOOD: transforms for the getting resnet features
    resnet_trans_indiv= transforms.Compose([Rescale((224, 224)),
                                        Transpose(),
                                        TimeNormalize(time_normalized_dimension),
                                        BGR2RGB(),
                                        ToTensor(),
                                        NormalizeVideo(),
                                        GetResnetFeats(),
                                        ToTensor() 
                                        ])

    composed_trans_indiv = transforms.Compose([Rescale(image_normalized_dimensions),
                                        Transpose(),
                                        TimeNormalize(time_normalized_dimension),
                                        ToTensor()])
    composed_trans_pair = transforms.Compose([ToTensor()])
    in_channels = 3 if not args.feature_extractor=='resnet' else 512
    
    if args.model_mode == 'simple':
        chosen_model_class = C3D_simplified
    elif args.model_mode == 'ltfb':
        chosen_model_class = LongTermFeatureBank
    else:
        chosen_model_class = C3D


    if MODE == 'individual':
        model_saveloc = os.path.join(args.model_folder, 'individual_run{}'.format(args.run_num))
        if not os.path.exists(model_saveloc):
            os.makedirs(model_saveloc, exist_ok=True)

        save_training_config(os.path.join(model_saveloc, 'config.json'), args, knowns)

        DF = EK_Dataset_pretrain(knowns, unknowns,
                train_object_csvpath, train_action_csvpath,
                class_key_csvpath, image_data_folder, 
                processed_frame_number=time_normalized_dimension,  
                transform=resnet_trans_indiv if args.feature_extractor=='resnet' else composed_trans_indiv,
                crop_type=args.crop_mode,
                mode='resnet' if args.feature_extractor=='resnet' else 'noresnet'
                ) 
        train_dataloader = data.DataLoader(DF, batch_size=args.batch_size, num_workers=0)
                                
        # model instatntiation and training
        model = chosen_model_class(input_shape=(in_channels, 
                    time_normalized_dimension, 
                    image_normalized_dimensions[0] if not args.feature_extractor=='resnet' else 7, 
                    image_normalized_dimensions[1] if not args.feature_extractor=='resnet' else 7), 
                    embedding_dim=args.embedding_dim) # TODO: replace these
        pretrain(model, train_dataloader, optimizer_type=args.optimizer, num_epochs=args.epochs, model_saveloc=model_saveloc, lr=args.lr)

    elif MODE == 'pairwise':
        model_saveloc = os.path.join(args.model_folder, 'pairwise_run{}'.format(args.run_num))
        if not os.path.exists(model_saveloc):
            os.makedirs(model_saveloc, exist_ok=True)


        DF = EK_Dataset_pretrain_pairwise(knowns, unknowns,
                train_object_csvpath, train_action_csvpath, 
                class_key_csvpath, image_data_folder,
                model_saveloc,
                processed_frame_number=time_normalized_dimension, 
                individual_transform=resnet_trans_indiv if args.feature_extractor=='resnet' else composed_trans_indiv, 
                pairwise_transform=composed_trans_pair,
                training_num_samples=args.num_samples, 
                crop_type=args.crop_mode,
                sampling_mode=args.sampler_mode,
                mode='resnet' if args.feature_extractor=='resnet' else 'noresnet',
                selector_train_ratio=args.selector_train_ratio
                ) 

        valset = DF.get_val_dataset()
        train_dataloader = data.DataLoader(DF, batch_size=args.batch_size, num_workers=0)
        # saving object DF in the model folder

        save_training_config(os.path.join(model_saveloc, 'config.json'), args, knowns, num_samples=len(DF))

        # now save DF.training_data, as well as DF.val_indices and DF.random_selection_indices
        with open(os.path.join(model_saveloc, 'data_info.pkl'), 'wb') as f:
            pickle.dump({'all_data': DF.training_data, 'train_indices': DF.rand_selection_indices, 'val_indices': DF.val_indices}, f)

        model = chosen_model_class(input_shape=(in_channels, 
                    time_normalized_dimension, 
                    image_normalized_dimensions[0] if not args.feature_extractor=='resnet' else 7, 
                    image_normalized_dimensions[1] if not args.feature_extractor=='resnet' else 7), 
                    embedding_dim=args.embedding_dim) # TODO: replace these
        pretrain_pairwise(model, train_dataloader, valset, optimizer_type=args.optimizer, num_epochs=args.epochs, model_saveloc=model_saveloc, lr=args.lr)

    elif MODE == 'batchwise':
        model_saveloc = os.path.join(args.model_folder, 'batchwise_run{}'.format(args.run_num))
        if not os.path.exists(model_saveloc):
            os.makedirs(model_saveloc, exist_ok=True)

        mini_batch_size = 20
        DF = EK_Dataset_pretrain_batchwise(knowns, unknowns,
                train_object_csvpath, train_action_csvpath, 
                class_key_csvpath, image_data_folder,
                model_saveloc,
                batch_size=mini_batch_size,
                processed_frame_number=time_normalized_dimension, 
                individual_transform=resnet_trans_indiv if args.feature_extractor=='resnet' else composed_trans_indiv, 
                batchwise_transform=composed_trans_pair,
                training_num_samples=args.num_samples, 
                crop_type=args.crop_mode,
                sampling_mode=args.sampler_mode,
                mode='resnet' if args.feature_extractor=='resnet' else 'noresnet',
                selector_train_ratio=args.selector_train_ratio)

        valset = DF.get_val_dataset()

        train_dataloader = data.DataLoader(DF, batch_size=args.batch_size, num_workers=0)
        save_training_config(os.path.join(model_saveloc, 'config.json'), args, knowns, num_samples=len(DF))

        # now save DF.training_data, as well as DF.val_indices and DF.random_selection_indices
        with open(os.path.join(model_saveloc, 'data_info.pkl'), 'wb') as f:
            pickle.dump({'all_data': DF.training_data, 'train_indices': DF.rand_selection_indices, 'val_indices': DF.val_indices}, f)

        model = chosen_model_class(input_shape=(in_channels, 
                    time_normalized_dimension, 
                    image_normalized_dimensions[0] if not args.feature_extractor=='resnet' else 7, 
                    image_normalized_dimensions[1] if not args.feature_extractor=='resnet' else 7), 
                    embedding_dim=args.embedding_dim) # TODO: replace these

        pretrain_batchwise(model, train_dataloader, valset, optimizer_type=args.optimizer, num_epochs=args.epochs, model_saveloc=model_saveloc, lr=args.lr)

    else:
        raise ValueError('invalid mode')
