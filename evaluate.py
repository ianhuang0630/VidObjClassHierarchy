from torchvision import transforms

from src.model import *
from src.input_layers import InputLayer

from data.EK_dataloader import *
from data.gt_hierarchy import *
from data.transforms import *
from tqdm import tqdm 

import pickle
from sklearn.metrics import classification_report

DEVICE='cuda:0'

def run_metrics(gt1, pred1, gt2, pred2):
    # input is 3 lists of the same length
    assert len(gt2) == len(gt1) and len(pred1) == len(pred2) and len(gt2) == len(pred2)

    print('Evaluating first-layer predictions')
    target_names = ['food', 'tools', 'others']
    print(classification_report(gt1, pred1, target_names=target_names))
    # Evaluating the second-layer predictions
    print('Evaluating second-layer predictions independent of first-layer predictions.')
    print(classification_report(gt2, pred2))

    # evaluating the second-layer predictions with consideration for the predictions made in the first layer. 
    print('Evaluating second-layer predictions w/ first-layer predictions')
    layer2_acc = np.sum(np.logical_and(np.array(gt1) == np.array(pred1), np.array(gt2) == np.array(pred2)))/len(gt2)
    print('Accuracy: {}'.format(layer2_acc))
    # analyzing the subset where first layer predictions agree

    gt2_prime = [gt2[i] for i in range(len(gt2)) if gt1[i] == pred1[i]]
    pred2_prime = [pred2[i] for i in range(len(gt2)) if gt1[i] == pred1[i]]
    print('Of the predictions where the first layer matches:')
    print(classification_report(gt2_prime, pred2_prime))

    # other metrics? marginal accuracy
    


def evaluate_model(model, dataloader, cache=None, overwrite=True):
    if cache is not None and os.path.exists(cache) and not overwrite:
        with open(cache, 'rb') as f:
            precomputed_results = pickle.load(f)
            all_gt1 = precomputed_results['gt1']
            all_preds1 = precomputed_results['preds1']
            all_gt2 = precomputed_results['gt2']
            all_preds2 = precomputed_results['preds2']

    else:        
        all_gt1 = []
        all_preds1 = []  
        all_gt2 = []
        all_preds2 = []

        for i, sample in enumerate(tqdm(dataloader)):
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

            model = model.to(DEVICE)

            x = {'handpose': handpose, 'handbbox': handbbox,
                'frames': frames, 'unknown':unknown, 'known':known}
            try:
                with torch.no_grad():
                    preds = model(x)
            except:
                continue

            level1_probs = preds['tree_level_pred1'].detach().cpu().numpy()
            level1_preds = np.argmax(level1_probs[:,:3], 1).tolist() # we only care about the first few
            level2_probs = preds['tree_level_pred2'].detach().cpu().numpy().tolist()
            level2_preds = np.argmax(level2_probs, 1).tolist()

            hierarchy_encoding = hierarchy_encoding.detach().cpu().numpy()[:, :2] # only caring about the first two 
            level1_gt = [int(element) for element in hierarchy_encoding[:,0].tolist()]
            level2_gt = [int(element) for element in hierarchy_encoding[:,1].tolist()]

            all_gt1.extend(level1_gt)
            all_gt2.extend(level2_gt)
            all_preds1.extend(level1_preds)
            all_preds2.extend(level2_preds)
            run_metrics(all_gt1, all_preds1, all_gt2, all_preds2)

        
        with open(cache, 'wb') as f:
            pickle.dump({'gt1': all_gt1,
                        'preds1': all_preds1,
                        'gt2': all_gt2,
                        'preds2': all_preds2}, f )

    run_metrics(all_gt1, all_preds1, all_gt2, all_preds2)


def evaluate_video_baseline(model, dataloader, cache=None, overwrite=True):
    if cache is not None and os.path.exists(cache) and not overwrite:
        with open(cache, 'rb') as f:
            precomputed_results = pickle.load(f)
            all_gt1 = precomputed_results['gt1']
            all_preds1 = precomputed_results['preds1']
            all_gt2 = precomputed_results['gt2']
            all_preds2 = precomputed_results['preds2']
    else:
        all_gt1 = []
        all_preds1 = []  
        all_gt2 = []
        all_preds2 = []

        for j, sample in enumerate(tqdm(dataloader)):
            unknown_predictions = sample['unknown']
            hierarchy_encoding = sample['hierarchy_encoding']
            level1_preds = [int(element) for element in unknown_predictions[:, 0].numpy().tolist()]
            level2_preds = [int(element) for element in unknown_predictions[:, 1].numpy().tolist()]
            level3_preds = [int(element) for element in unknown_predictions[:, 2].numpy().tolist()]

            level1_gt = [int(element) for element in hierarchy_encoding[:,0].tolist()]
            level2_gt = [int(element) for element in hierarchy_encoding[:,1].tolist()]

            all_gt1.extend(level1_gt)
            all_gt2.extend(level2_gt)
            all_preds1.extend(level1_preds)
            all_preds2.extend(level2_preds)

            if (j+1)%10 == 0: 
                run_metrics(all_gt1, all_preds1, all_gt2, all_preds2)

        with open(cache, 'wb') as f:
            pickle.dump({'gt1': all_gt1,
                        'preds1': all_preds1,
                        'gt2': all_gt2,
                        'preds2': all_preds2}, f )

    run_metrics(all_gt1, all_preds1, all_gt2, all_preds2)


def evaluate_baseline_model(model, dataloader, cache=None, overwrite=True):
    if cache is not None and os.path.exists(cache) and not overwrite:
        with open(cache, 'rb') as f:
            precomputed_results = pickle.load(f)
            all_gt1 = precomputed_results['gt1']
            all_preds1 = precomputed_results['preds1']
            all_gt2 = precomputed_results['gt2']
            all_preds2 = precomputed_results['preds2']
    else: 
        all_gt1 = []
        all_preds1 = []  
        all_gt2 = []
        all_preds2 = []

        for j, sample in enumerate(tqdm(dataloader)):
            frame = sample['frame']
            hierarchy_encoding = sample['hierarchy_encoding']

            frame = frame.type(torch.FloatTensor).to(DEVICE)
            hierarchy_encoding = hierarchy_encoding.type(torch.FloatTensor).to(DEVICE)

            model = model.to(DEVICE)
            with torch.no_grad():
                preds = model(frame)
            level1_probs = preds['tree_level_pred1'].detach().cpu().numpy()
            level1_preds = np.argmax(level1_probs[:,:3], 1).tolist()
            level2_probs = preds['tree_level_pred2'].detach().cpu().numpy().tolist()
            level2_preds = np.argmax(level2_probs, 1).tolist()

            hierarchy_encoding = hierarchy_encoding.detach().cpu().numpy()[:, :2] # only caring about the first two 
            level1_gt = [int(element) for element in hierarchy_encoding[:,0].tolist()]
            level2_gt = [int(element) for element in hierarchy_encoding[:,1].tolist()]

            all_gt1.extend(level1_gt)
            all_gt2.extend(level2_gt)
            all_preds1.extend(level1_preds)
            all_preds2.extend(level2_preds)

            if (j+1)%10 == 0: 
                run_metrics(all_gt1, all_preds1, all_gt2, all_preds2)

        with open(cache, 'wb') as f:
            pickle.dump({'gt1': all_gt1,
                        'preds1': all_preds1,
                        'gt2': all_gt2,
                        'preds2': all_preds2}, f )

    run_metrics(all_gt1, all_preds1, all_gt2, all_preds2)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch hierarchy discovery model training')
    parser.add_argument('--data', type=str, default='/vision/group/EPIC-KITCHENS',
                        help='path to dataset directory')
    # parser.add_argument('--model_folder', type=str, 
    #                     default='models/hierarchy_discovery/',
    #                     help='path to dataset directory')
    # parser.add_argument('--epoch_number', type=int, default=99,
    #                     help='Epoch number to load the hierarchy disocvery model')
    # parser.add_argument('--run_number', type=int, default=0,
    #                     help='run number corresponding to the version of the model')
    parser.add_argument('--our_model_path', type=str, 
                        default='models/hierarchy_discovery/hierarchy_discovery_run15/net_epoch49.pth')
    parser.add_argument('--visual_encoder_path', type=str, 
                        default='models/pretraining_tree/framelevel_pred_run9/net_epoch99.pth',
                        help='path to tree encoder')
    parser.add_argument('--oracle_visual_encoder_path', type=str,
                        default='models/pretraining_tree/framelevel_pred_run17/net_epoch5.pth')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers for the dataloader')
    parser.add_argument('--time_normalized_dimension', type=int, default=16,
                        help='time standardized dimension')
    parser.add_argument('--eval_cache_path', type=str, default='eval_classification_cache/V1')

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

    # get the classes that you wanna test on
    print('GETTING KNOWN UNKNOWN SPLITS FOR THIS TEST')
    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    train_knowns = split['training_known']
    train_unknowns = split['training_unknown']
    test_knowns = list(set(train_knowns + train_unknowns)) # training knowns and unknowns are the knowns during testing
    test_unknowns = split['testing_unknown']

    # get the dataset
    
    ## for static images
    print('CONSTRUCTING DATASET FOR THE STATIC MODEL')
    DF_static = EK_Dataset_pretrain_framewise_prediction(test_unknowns, test_knowns, # knowns, unknowns swapped
                train_object_csvpath, train_action_csvpath, 
                class_key_csvpath, image_data_folder,
                None,
                crop_type='rescale',
                mode='resnet',
                validation_num_samples=0)
    static_val_loader = data.DataLoader(DF_static, batch_size=64, num_workers=0)

    ## for clips
    # initializing the transforms
    print('CONSTRUCTING DATASET FOR THE CLIPS MODEL')
    handpose_transforms = transforms.Compose([
                                FeatureNormalize(means=[0]*126, stds=[1]*126), # TODO: find actual numbers
                                TimeStandardize(args.time_normalized_dimension),
                                # transforms.ToTensor()
                            ])
    # instantiate hand_bbox_transforms: scale everything down to range from 0 to 1, time normalize, then into torch tensor
    hand_bbox_transforms = transforms.Compose([
                                BboxUnitScale(image_height=1080, image_width=1920),
                                TimeStandardize(args.time_normalized_dimension),
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
                                TimeStandardize(args.time_normalized_dimension),
                                # I3D_feats(device='cpu' if args.num_workers else DEVICE),
                                # transforms.ToPILImage(),
                                # transforms.ToTensor()
                            ])
    video_feat_extract_transforms = transforms.Compose([
                                I3D_feats(device='cpu' if args.num_workers else DEVICE,
                                        cache_dir='i3d_cache', overwrite=False),
                            ])
    label_transforms = None

    DF = DatasetFactory(test_knowns, test_unknowns, train_object_csvpath, 
                        train_action_csvpath, class_key_csvpath)
    test_dataset = DF.get_dataset()
    import ipdb; ipdb.set_trace()
    random.shuffle(test_dataset['unknown'])
    inputlayer= InputLayer()
    DF_clip = EK_Dataset_discovery(test_knowns, test_unknowns,
                train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder,
                test_dataset, inputlayer,
                tree_encoder=args.visual_encoder_path,
                video_transforms=video_transforms, 
                video_feat_extract_transforms = video_feat_extract_transforms,
                hand_pose_transforms=handpose_transforms, 
                hand_bbox_transforms=hand_bbox_transforms, 
                embedding_transforms=embedding_transforms,
                label_transforms=label_transforms,
                device='cpu' if args.num_workers else DEVICE,
                snip_threshold=24) # changed to 24 to guard against crashes

    clip_val_loader = data.DataLoader(DF_clip, batch_size=8, num_workers=args.num_workers)

    DF_clip_vb = EK_Dataset_discovery(test_knowns, test_unknowns,
                train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder,
                test_dataset, inputlayer,
                tree_encoder=args.visual_encoder_path,
                video_transforms=video_transforms, 
                video_feat_extract_transforms = video_feat_extract_transforms,
                hand_pose_transforms=handpose_transforms, 
                hand_bbox_transforms=hand_bbox_transforms, 
                embedding_transforms=embedding_transforms,
                label_transforms=label_transforms,
                device='cpu' if args.num_workers else DEVICE,
                snip_threshold=24, is_video_baseline=True)
    
    DF_clip_or_vb = EK_Dataset_discovery(test_knowns, test_unknowns,
                train_object_csvpath, train_action_csvpath, class_key_csvpath, image_data_folder,
                test_dataset, inputlayer,
                tree_encoder=args.oracle_visual_encoder_path,
                video_transforms=video_transforms, 
                video_feat_extract_transforms = video_feat_extract_transforms,
                hand_pose_transforms=handpose_transforms, 
                hand_bbox_transforms=hand_bbox_transforms, 
                embedding_transforms=embedding_transforms,
                label_transforms=label_transforms,
                device='cpu' if args.num_workers else DEVICE,
                snip_threshold=24, is_video_baseline=True)
    

    clip_val_loader_vb = data.DataLoader(DF_clip_vb, batch_size=8, num_workers=0)
    clip_val_loader_or_vb = data.DataLoader(DF_clip_or_vb, batch_size=8, num_workers=0)

    # run the evaluation on baseline
    print('LOADING HIERARCHY DISCOVERY MODEL')
    # hd_model_path = os.path.join(
    #     os.path.join(args.model_folder, 'hierarchy_discovery_run{}'.format(args.run_number)), 
    #             'net_epoch{}.pth'.format(args.epoch_number)
    #             )
    hd_model_path = args.our_model_path
    hd_model = torch.load(hd_model_path)
    hd_model.eval()

    print('LOADING TREE ENCODER MODEL')
    ve_model = torch.load(args.visual_encoder_path)
    ve_model.eval()

    print('LOADING TREE ENCODER ORACLE MODEL')
    vo_model = torch.load(args.oracle_visual_encoder_path)
    vo_model.eval()
    
    oracle_cache_name  = \
        '-'.join(args.oracle_visual_encoder_path.split('/')[-2:])[:-len('.pth')] + '.pkl'
    baseline_cache_name = \
        '-'.join(args.visual_encoder_path.split('/')[-2:])[:-len('.pth')] + '.pkl'
    ourmodel_cache_name = \
        '-'.join(args.our_model_path.split('/')[-2:])[:-len('.pth')]+ '.pkl'
    video_baseline_cache_name = 'video_'+'-'.join(args.visual_encoder_path.split('/')[-2:])[:-len('.pth')] + '.pkl'
    oracle_video_baseline_cache_name =  'video_'+'-'.join(args.oracle_visual_encoder_path.split('/')[-2:])[:-len('.pth')] + '.pkl'

    if not os.path.exists(args.eval_cache_path):
        os.makedirs(args.eval_cache_path)
    oracle_cache_path = os.path.join(args.eval_cache_path, oracle_cache_name)
    baseline_cache_path = os.path.join(args.eval_cache_path, baseline_cache_name)
    ourmodel_cache_path = os.path.join(args.eval_cache_path, ourmodel_cache_name)
    video_baseline_cache_path = os.path.join(args.eval_cache_path, video_baseline_cache_name)
    oracle_video_baseline_cache_path = os.path.join(args.eval_cache_path, oracle_video_baseline_cache_name)

    print('ORACLE VIDEO BASELINE ACCURACY')
    print('---------------')
    evaluate_video_baseline(vo_model, clip_val_loader_or_vb, cache=oracle_video_baseline_cache_path, overwrite=False)
    print('---------------')

    print('VIDEO BASELINE ACCURACY')
    print('---------------')
    evaluate_video_baseline(ve_model, clip_val_loader_vb, cache=video_baseline_cache_path, overwrite=False)
    print('---------------')

    print('ORACLE ACCURACY')
    print('---------------')
    evaluate_baseline_model(vo_model, static_val_loader, cache=oracle_cache_path, overwrite=False)
    print('---------------')

    print('BASELINE ACCURACY')
    print('---------------')
    evaluate_baseline_model(ve_model, static_val_loader, cache=baseline_cache_path, overwrite=False)
    print('---------------')

    print('MODEL ACCURACY')
    print('---------------')
    evaluate_model(hd_model, clip_val_loader, cache=ourmodel_cache_path, overwrite=False)
    print('---------------')

    pass