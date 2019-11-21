"""
Training the known unknown detectors
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
from src.detector_models import *
from src.input_layers import InputLayer

from data.EK_dataloader import *
from data.gt_hierarchy import *
from data.transforms import *
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = 'cuda:0'

def train_vanilla(net, train_dataloader, val_dataloader, optimizer_type, num_epochs, model_saveloc, lr,
    save_interval=1):
     
    if not os.path.exists(model_saveloc):
        os.makedirs(model_saveloc, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                      verbose=False, threshold=1e-4, threshold_mode='rel',
                                      cooldown=0, min_lr=0, eps=1e-8)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                      verbose=False, threshold=1e-4, threshold_mode='rel',
                                      cooldown=0, min_lr=0, eps=1e-8)

    loss_per_sample = []
    val_losses_per1000 = []
    counter = 0

    for epoch in range(num_epochs):
        print('training on epoch {}'.format(epoch))
        prediction_counts = {class_: 0 for class_ in range(3)}
        # import ipdb; ipdb.set_trace()
        for i, sample in enumerate(tqdm(train_dataloader)):
            embeddings = sample['embeddings']
            labels = sample['labels']

            embeddings = embeddings.reshape((-1, embeddings.shape[-1])).type(torch.FloatTensor).to(DEVICE)
            labels = labels.reshape((-1)).type(torch.FloatTensor).to(DEVICE)
            net = net.to(DEVICE)

            pred = net(embeddings)
            pred_labels = torch.argmax(pred, -1).data

            prediction_counts[0] += torch.sum(pred_labels == 0).cpu().item() # background
            prediction_counts[1] += torch.sum(pred_labels == 1).cpu().item() # known
            prediction_counts[2] += torch.sum(pred_labels == 2).cpu().item() # unknown
            print('Distribution of predictions in this epoch: \n {}'.\
                format({element: prediction_counts[element]/sum(prediction_counts.values()) for element in prediction_counts}))

            loss = criterion(pred, labels.type(torch.long))
            loss_per_sample.append(loss.data.cpu().numpy())
            loss.backward()

            optimizer.step()

            if counter % 1000 == 0:
                val_losses = []
                # do validation
                for j, val_sample in enumerate(tqdm(val_dataloader)):
                    val_embeddings = val_sample['embeddings']
                    val_labels = val_sample['labels']

                    val_embeddings = val_embeddings.reshape((-1, val_embeddings.shape[-1])).type(torch.FloatTensor).to(DEVICE)
                    val_labels = val_labels.reshape((-1)).type(torch.FloatTensor).to(DEVICE)

                    # DOES SOMETHING
                    val_pred = net(val_embeddings)

                    val_loss = criterion(val_pred, val_labels.type(torch.long))
                    val_losses.append(val_loss.data.cpu().numpy())
                val_losses_per1000.append(np.mean(val_losses))
                with open(os.path.join(model_saveloc, 'validation_losses.pkl'.format(epoch)), 
                            'wb') as f:
                    pickle.dump(val_losses_per1000, f)

            counter += 1
        if epoch % save_interval == 0:
            print('current loss: {}'.format(str(loss)))
            with open(os.path.join(model_saveloc, 'training_losses.pkl'.format(epoch)), 
                'wb') as f:
                pickle.dump(loss_per_sample, f)
            torch.save(net, os.path.join(model_saveloc,
                'net_epoch{}.pth'.format(epoch)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch hierarchy discovery model training')
    parser.add_argument('--data', type=str, default='/vision/group/EPIC-KITCHENS',
                        help='path to dataset directory')
    parser.add_argument('--model_folder', type=str, default='models/hierarchy_discovery/',
                        help='path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for the model')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of training samples vs. validation samples')

    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--use-cuda', default=False, type=bool, metavar='C',
                        help='use cuda')
    parser.add_argument('--run_num', type=int, default=0, 
                        help='the run number that will be included in all output files')
    parser.add_argument('--max_training_knowns', type=int, default=None,
                        help='Effectively the number of known classes in the pretraining.')

    parser.add_argument('--max_num_boxes', type=int, default=50,
                        help='The maximum number of region proposals taken.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='type of optimizer to use')
    parser.add_argument('--visual_tree_encoder', type=str, default='models/pretraining_tree/framelevel_pred_run0/net_epoch0.pth',
                        help='path to static visual tree encoder')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for the training and validation dataloaders. Default 0.')
    parser.add_argument('--data_draw', type=float, default=1.0,
                        help='This allows training on only a fraction of the data, \
                        since training on the whole set can take forever. By default, this draw ratio is 1 \
                        (i.e. all data is used for training and validation)')

    parser.add_argument('--model_type', type=str, default='vanilla',
                        help='can be "vanilla" or "deep".')
    args = parser.parse_args()
    

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

    # get known unknown splits
    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    knowns = split['training_known']
    unknowns = split['training_unknown']

    train_ratio=args.train_ratio
    DF = DatasetFactory(knowns, unknowns, train_object_csvpath, train_action_csvpath, class_key_csvpath)
    dataset = DF.get_dataset()
    dataset2 = DF.get_dataset()

    unknown_keep_idx = np.random.choice(len(dataset['unknown']), round(len(dataset['unknown'])*args.data_draw), replace=False)
    known_keep_idx = np.random.choice(len(dataset['known']), round(len(dataset['known'])*args.data_draw), replace=False)
    dataset['unknown'] = [dataset['unknown'][idx] for idx in unknown_keep_idx]
    dataset['known'] = [dataset['known'][idx] for idx in known_keep_idx]
    #dataset2 is linked to dataset



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

    model_saveloc = os.path.join(args.model_folder, 'hierarchy_detection_run{}'.format(args.run_num))

    if args.model_type == 'vanilla':
        model = VanillaDetector()
    elif args.model_type == 'deep':
        model = DeeperDetector()

    # train predictor for 3 classes: not object, known, unknown.
    inputlayer_rpn = InputLayer(max_num_boxes=None, rpn_conf_thresh=0.0, rpn_only=True)

    # DF_train = EK_Dataset_detection(knowns, unknowns,
    #         train_object_csvpath, train_action_csvpath, class_key_csvpath, 
    #         image_data_folder, training_dataset, inputlayer_rpn,
    #         tree_encoder=args.visual_tree_encoder, purpose='train', max_num_boxes=args.max_num_boxes, max_gt_boxes=6)
    DF_train = EK_Dataset_detection(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, 
            image_data_folder, training_dataset, inputlayer_rpn,
            tree_encoder=None, purpose='resnet101', max_num_boxes=args.max_num_boxes, max_gt_boxes=6)
    DF_train.only_unique_frames() # filter out the repeated frames
    DF_train.restrict_to_cached() # filter out the ones that haven't been preprocessed.
    import ipdb; ipdb.set_trace()
    # now creating dataloaders
    train_dataloader= data.DataLoader(DF_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    DF_val = EK_Dataset_detection(knowns, unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, 
            image_data_folder, validation_dataset, inputlayer_rpn,
            tree_encoder=args.visual_tree_encoder, purpose='train', max_num_boxes=args.max_num_boxes, max_gt_boxes=6)
    DF_val.only_unique_frames() # filter out the repeated frames
    DF_val.restrict_to_cached() # filter out the ones that haven't been preprocessed.
    # now creating dataloaders
    val_dataloader= data.DataLoader(DF_val, batch_size=args.batch_size, num_workers=args.num_workers)

    # comparison of latent embeddings for g? Training g is all we do. g already has the visual embedding knowledge of "seen" classes
    # push through same g, then linear layers. output 3 classes. not object, known, unknown.

    train_vanilla(model, train_dataloader, val_dataloader, optimizer_type=args.optimizer, 
        num_epochs=args.epochs, model_saveloc=model_saveloc, lr=args.lr)
    
    # train on knowns and unknowns

    # plug in g already has visual embedding of "seen classes" (i.e.) the pretrained on the union of straining knowns and unknowns

    # test on testing unknowns
