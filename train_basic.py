"""
The training script of the network predictor
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
from src.model_basic import * 
from src.input_layers import InputLayer

from data.EK_dataloader import *
from data.gt_hierarchy import *
from data.transforms import *
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau


DEVICE = 'cuda:0'

def train_vanilla(net, train_dataloader, val_dataloader, optimizer_type, num_epochs, model_saveloc, lr,
    save_interval=1, class_freq_path=None):

    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)

    criterion = BasicLoss()
    
    if optimizer_type=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
        #                               verbose=False, threshold=1e-4, threshold_mode='rel',
        #                               cooldown=0, min_lr=0, eps=1e-8)

    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
        #                               verbose=False, threshold=1e-4, threshold_mode='rel',
        #                               cooldown=0, min_lr=0, eps=1e-8)

    loss_per_sample = []
    val_losses_per50 = []
    counter = 0 

    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))
        for i, sample in enumerate(tqdm(train_dataloader)):
            # dissect sample
            handpose = sample['handpose']
            handbbox = sample['handbbox']
            frames = sample['frames']
            unknown = sample['unknown']
            known = sample['known']
            hierarchy_encoding = sample['hierarchy_encoding']

            handpose = handpose.type(torch.FloatTensor).to(DEVICE)
            handbbox = handbbox.type(torch.FloatTensor).to(DEVICE)
            frames = frames.type(torch.FloatTensor).to(DEVICE)
            unknown = unknown.type(torch.FloatTensor).to(DEVICE)
            known = known.type(torch.FloatTensor).to(DEVICE)
            hierarchy_encoding = hierarchy_encoding.type(torch.FloatTensor).to(DEVICE)

            net = net.to(DEVICE)

            x = {'handpose': handpose, 'handbbox': handbbox,
                'frames': frames, 'unknown':unknown, 'known':known}
            
            optimizer.zero_grad()

            results = net(x)

            loss = criterion([results['tree_level_pred1'], results['tree_level_pred2'], results['tree_level_pred3']], hierarchy_encoding)
            print(loss.data.cpu().numpy())
            loss_per_sample.append(loss.data.cpu().numpy())
            loss.backward()

            # take optimizer step
            optimizer.step()

            if counter % 50 == 0:
                # do validation
                with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                            'wb') as f:
                    pickle.dump(loss_per_sample, f)
                val_losses = []
                for i, val_sample in enumerate(tqdm(val_dataloader)):
                    # print('GOT SAMPLE')
                    val_handpose = val_sample['handpose']
                    val_handbbox = val_sample['handbbox']
                    val_frames = val_sample['frames']
                    val_unknown = val_sample['unknown']
                    val_known = val_sample['known']
                    val_hierarchy_encoding = val_sample['hierarchy_encoding']
                    # print('DIVIDED SAMPLE')

                    val_handpose = val_handpose.type(torch.FloatTensor).to(DEVICE)
                    val_handbbox = val_handbbox.type(torch.FloatTensor).to(DEVICE)
                    val_frames = val_frames.type(torch.FloatTensor).to(DEVICE)
                    val_unknown = val_unknown.type(torch.FloatTensor).to(DEVICE)
                    val_known = val_known.type(torch.FloatTensor).to(DEVICE)
                    val_hierarchy_encoding = val_hierarchy_encoding.type(torch.FloatTensor).to(DEVICE)
                    # print('ALLOCATED SAMPLE TO GPU')

                    net = net.to(DEVICE)
                    # print('ALLOCATED NETWORK TO GPU')

                    x = {'handpose': val_handpose, 'handbbox': val_handbbox,
                        'frames': val_frames, 'unknown':val_unknown, 'known':val_known}

                    val_results = net(x)
                    # print('PUSHED THROUGH NETWORK')
                    val_loss = criterion([val_results['tree_level_pred1'], val_results['tree_level_pred2'], val_results['tree_level_pred3']], val_hierarchy_encoding)
                    # print('EVALUATED LOSSES')

                    val_losses.append(val_loss.data.cpu().numpy())
                    # print('APPENDED LOSSES')
                
                val_losses_per50.append(np.mean(val_losses))
                # scheduler.step(np.mean(val_losses))

                with open(os.path.join(model_saveloc, 'validation_losses.pkl'.format(epoch)), 
                            'wb') as f:
                    pickle.dump(val_losses_per50, f)

            counter += 1

        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                'wb') as f:
                pickle.dump(loss_per_sample, f)
            torch.save(net, os.path.join(model_saveloc,
                'net_epoch{}.pth'.format(epoch)))

    return net 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch hierarchy discovery model training')
    parser.add_argument('--data', type=str, default='/vision/group/EPIC-KITCHENS',
                        help='path to dataset directory')
    parser.add_argument('--model_folder', type=str, default='models/hierarchy_discovery/',
                        help='path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--rescale_imwidth', type=int, default = 200, 
                        help='width of the image after rescaling')
    parser.add_argument('--rescale_imheight', type=int, default = 150,
                        help='height of the image after rescaling')
    parser.add_argument('--time_normalized_dimension', type=int, default = 16,
                        help='number of timesteps for which the normalization is done')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for the model')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of training samples vs. validation samples')

    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--use-cuda', default=False, type=bool, metavar='C',
                        help='use cuda')
    parser.add_argument('--run_num', type=int, default=0, 
                        help='the run number that will be included in all output files')
    parser.add_argument('--max_training_knowns', type=int, default=None,
                        help='Effectively the number of known classes in the pretraining.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='type of optimizer to use')
    parser.add_argument('--visual_tree_encoder', type=str, default='models/pretraining_tree/framelevel_pred_run0/net_epoch0.pth',
                        help='path to static visual tree encoder')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for the training and validation dataloaders. Default 0.')
    parser.add_argument('--class_freq_path', type=str, default='',
                        help='path to the pkl file containing the number of appearances of each gt label in the training set')
    args = parser.parse_args()

    print('num_workers = {}'.format(args.num_workers))

    class_freq_path = None if len(args.class_freq_path)==0 else args.class_freq_path

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

    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    knowns = split['training_known']
    unknowns = split['training_unknown']

    # instantiate hand_pose_transforms: normalize every dimension wrt some batch mean and standard dev, time normalize, then into torch tensor
    handpose_transforms = transforms.Compose([
                                FeatureNormalize(means=[0]*126, stds=[1]*126), # TODO: find actual numbers
                                TimeStandardize(time_normalized_dimension),
                                # transforms.ToTensor()
                            ])
    # instantiate hand_bbox_transforms: scale everything down to range from 0 to 1, time normalize, then into torch tensor
    hand_bbox_transforms = transforms.Compose([
                                BboxUnitScale(image_height=1080, image_width=1920),
                                TimeStandardize(time_normalized_dimension),
                                # transforms.ToTensor()
                            ])
    # instantiate embedding_transforms: time normalize, just turn into torch tensor
    embedding_transforms = None
    #                      transforms.Compose([
    #                             TimeStandardize(time_normalized_dimension),
    #                             transforms.ToTensor()
    #                         ])
    # instantiate video_transforms: i3d, time normalize
    video_transforms = transforms.Compose([
                                TimeStandardize(time_normalized_dimension),
                                # I3D_feats(device='cpu' if args.num_workers else DEVICE),
                                # transforms.ToPILImage(),
                                # transforms.ToTensor()
                            ])
    video_feat_extract_transforms = transforms.Compose([
                                I3D_feats(device='cpu' if args.num_workers else DEVICE,
                                        cache_dir='i3d_cache', overwrite=False),
                            ])

    # instantiate label_transforms: just turninto torch tensor
    label_transforms = None
    # transforms.Compose([
    #                             transforms.ToTensor()
    #                         ])

    model_saveloc = os.path.join(args.model_folder, 'hierarchy_discovery_run{}'.format(args.run_num))

    model = VanillaEnd2End(frame_feat_shape= (1024, 1), tree_embedding_dim = 4, handpose_dim = 126,
             handbbox_dim = 8, timesteps=time_normalized_dimension, device=DEVICE)

    DF = DatasetFactory(knowns, unknowns, train_object_csvpath, train_action_csvpath, class_key_csvpath)
    dataset = DF.get_dataset()
    dataset2 = DF.get_dataset()

    random.shuffle(dataset['unknown'])
    training_data = dataset['unknown'][: int(args.train_ratio*len(dataset['unknown']))]
    training_dataset = dataset.copy()
    training_dataset['unknown'] = training_data 

    validation_data = dataset2['unknown'][int(args.train_ratio*len(dataset['unknown'])):]
    validation_dataset = dataset2.copy()
    validation_dataset['unknown'] = validation_data

    inputlayer = InputLayer()

    DF_train = EK_Dataset_discovery(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder,
            training_dataset, inputlayer, 
            tree_encoder=args.visual_tree_encoder,
            video_transforms=video_transforms, 
            video_feat_extract_transforms = video_feat_extract_transforms,
            hand_pose_transforms=handpose_transforms, 
            hand_bbox_transforms=hand_bbox_transforms, 
            embedding_transforms=embedding_transforms,
            label_transforms=label_transforms,
            device='cpu' if args.num_workers else DEVICE,
            snip_threshold=36)

    train_dataloader= data.DataLoader(DF_train, batch_size=args.batch_size, num_workers=args.num_workers)

    l1_keys, l2_keys, l3_keys =  get_tree_position_keys(knowns)

    print('Layer 1 keys: {}'.format(l1_keys))
    print('Layer 2 keys: {}'.format(l2_keys))
    print('Layer 3 keys: {}'.format(l3_keys))

    DF_val = EK_Dataset_discovery(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder,
            validation_dataset, inputlayer, 
            tree_encoder=args.visual_tree_encoder,
            video_transforms=video_transforms, 
            video_feat_extract_transforms = video_feat_extract_transforms,
            hand_pose_transforms=handpose_transforms, 
            hand_bbox_transforms=hand_bbox_transforms, 
            embedding_transforms=embedding_transforms,
            label_transforms=label_transforms,
            device='cpu' if args.num_workers else DEVICE,
            snip_threshold=36)
    val_dataloader= data.DataLoader(DF_val, batch_size=args.batch_size, num_workers=args.num_workers)
    

    train_vanilla(model, train_dataloader, val_dataloader, optimizer_type=args.optimizer, 
        num_epochs=args.epochs, model_saveloc=model_saveloc, lr=args.lr, class_freq_path=class_freq_path)


    pass
