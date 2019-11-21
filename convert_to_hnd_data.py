"""
"""
import os
import numpy as np
import scipy.io as sio
import multiprocessing

from torchvision import transforms
from torch.utils import data

from data.gt_hierarchy import *
from data.transforms import *
from data.EK_dataloader import *
from tqdm import tqdm
import random

import sys

if __name__=='__main__':

    # if sys.argv[1] == 'resnet101':
    #     hnd_data_dir = 'hnd_EK_res101'
    # elif sys.argv[1] == 'resnet18':
    #     hnd_data_dir = 'hnd_EK_res18'
    # else:
    #     raise ValueError
    hnd_data_dir = 'hnd_EK'
    if not os.path.exists(hnd_data_dir):
        os.makedirs(hnd_data_dir)

    with open('hierarchyV1.json', 'r') as f:
        hierarchy = json.load(f)

    ch_pa = []
    parents = []
    parents_id = []

    for l1_is_str in hierarchy:
        l1_is_str_ = '+'.join(l1_is_str.split(' '))
        parents.append(l1_is_str_)
        parents_id.append('l1_'+l1_is_str_)
        ch_pa.append(('l1_'+l1_is_str_, '_root_'))

        for l2_is_dict in hierarchy[l1_is_str]:
            l2_is_str = list(l2_is_dict.keys())[0]
            l2_is_str_ = '+'.join(l2_is_str.split(' '))

            parents.append(l2_is_str_)
            parents_id.append('l2_'+l2_is_str_)
            ch_pa.append(('l2_'+l2_is_str_, 'l1_'+l1_is_str_))
            for l3_is_str in l2_is_dict[l2_is_str]:
                l3_is_str_ = '+'.join(l3_is_str.split(' '))
                ch_pa.append(('l3_'+l3_is_str_, 'l2_'+l2_is_str_))
    parents.append('_root_')
    parents_id.append('_root_')

    # datafactory for splits and static frames.
    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    split_prime = {}
    split_prime['training_known'] = ['+'.join(element.split(' ')) for element in split['training_known']]
    split_prime['training_unknown'] = ['+'.join(element.split(' ')) for element in split['training_unknown']]
    split_prime['testing_unknown'] = ['+'.join(element.split(' ')) for element in split['testing_unknown']]

    train_knowns = np.unique(split_prime['training_known']).tolist()
    train_unknowns = np.unique(split_prime['training_unknown']).tolist()
    test_knowns = np.unique(train_knowns + train_unknowns).tolist() # training knowns and unknowns are the knowns during testing
    test_unknowns = np.unique(split_prime['testing_unknown']).tolist()
    test_unknowns.remove('support')
    # the leaf nodes
    trainval_classes = train_knowns.copy()
    trainval_classes.extend(np.unique(list(set(train_unknowns)-set(train_knowns))).tolist()) 
    test_classes = test_unknowns
    all_classes = trainval_classes + test_classes 

    # there is some random class "support"


    word2idx = {word:idx for idx, word in enumerate(all_classes)}

    ## classes and hierarchy
    # allclasses.txt (all leaf nodes in the hierarchy)
    with open(os.path.join(hnd_data_dir, 'allclasses.txt'), 'w') as f:
        f.write('\n'.join(all_classes))
    # trainvalclasses.txt (training knowns and unknowns)
    with open(os.path.join(hnd_data_dir, 'trainvalclasses.txt'), 'w') as f:
        f.write('\n'.join(trainval_classes))
    # testclasses.txt (testing unknowns)
    with open(os.path.join(hnd_data_dir, 'testclasses.txt'), 'w') as f:
        f.write('\n'.join(test_classes))

    ## the wordnet replacement
    # make your own taxonomy/words.txt (all parent nodes and leaf nodes)

    ids = ['l3_'+item for item in all_classes] # all wnids are just going to be capitalized
    ids.extend(parents_id) # _root_ included here
    nouns = [item for item in all_classes]
    nouns.extend(parents) # _root_ included here

    id2noun = dict(zip(ids, nouns))

    with open(os.path.join(hnd_data_dir, 'words.txt'), 'w') as f:
        # f.write('\n'.join(['{id}\t{word}'.format(id=noun2id[item], word=item) for item in all_classes]))
        f.write('\n'.join(['{id}\t{word}'.format(id=ids[i], word=nouns[i]) for i in range(len(ids))]))
        # TODO: write the parent classes into this too
    # awa_classes_offset_rev1.txt (from hierarchyV1.json)

    parents_of_leaves = []
    for element in all_classes:
        
        element_id = 'l3_'+element
        pa = [tup for tup in ch_pa if tup[0] == element_id]
        if len(pa) > 1:
            assert len(set([el[1] for el in pa])) == 1, '{class_} should have a unique parent'.format(class_=element)
        elif len(pa) < 0:
            raise ValueError('{class_} does not have a parent. Right now, {pa}'.format(class_=element, pa=pa))
        this_parent_id = pa[0][1] 
        parents_of_leaves.append(this_parent_id)
    
    with open(os.path.join(hnd_data_dir, 'awa_classes_offset_rev1.txt'), 'w') as f:
        f.write('\n'.join(parents_of_leaves))

    pa_papa_id = []
    for pa in parents_id:
        if pa != '_root_':
            an = [element[1] for element in ch_pa if element[0] == pa][0]
            pa_papa_id.append((an, pa))

    # taxonomy wordnet.is_a.txt (rest of the parent nodes go here)
    with open(os.path.join(hnd_data_dir, 'wordnet.is_a.txt'), 'w') as f:
        f.write('\n'.join([' '.join(element) for element in pa_papa_id]))


    # dataloader pretrain_framewise, with no featuree map transform (only rescaling and normalization)
    dataset_path = '/vision/group/EPIC-KITCHENS'
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

    feat_cache_path = os.path.join(hnd_data_dir, 'feat_cache')
    if not os.path.exists(feat_cache_path):
        os.makedirs(feat_cache_path, exist_ok=True)

    res101= GetResnetFeatsGeneral(version='resnet18', mode='map')
    g = torch.load('models/pretraining_tree/framelevel_pred_run15/net_epoch10.pth')
    g.eval()

    import ipdb; ipdb.set_trace()


    def res101_eq (x):
        with torch.no_grad():
            return g(res101(x))['embedding']
    # res101 = GetResnetFeatsGeneral(version='resnet18', mode='vec')
    # res101 = GetResnetFeatsGeneral(version='resnet101', mode='vec')
    
    print('TRAIN KNOWN')
    if not os.path.exists(os.path.join(feat_cache_path, 'dl_train_known_features.npy')) and \
        not os.path.exists(os.path.join(feat_cache_path, 'dl_train_known_labels.npy')):
        DF_train_known = EK_Dataset_pretrain_framewise_prediction(
                    split['training_known'], split['training_unknown'],
                    train_object_csvpath, train_action_csvpath, 
                    class_key_csvpath, image_data_folder,
                    model_saveloc = None,
                    crop_type='rescale',
                    mode='resnet',
                    prune_target='known',
                    resnet_out=False)
        dl_train_known = data.DataLoader(DF_train_known, 
                        batch_size=64, num_workers=5)
        dl_train_known_features = []
        dl_train_known_labels = []    
        import ipdb; ipdb.set_trace()
        for batch in tqdm(dl_train_known):
            dl_train_known_features.append(res101_eq(batch['frame']).squeeze().detach().cpu().numpy())
            dl_train_known_labels.extend(batch['noun_label'])
        dl_train_known_features = np.vstack(dl_train_known_features)
        np.save(os.path.join(feat_cache_path, 'dl_train_known_features.npy'), dl_train_known_features)
        np.save(os.path.join(feat_cache_path, 'dl_train_known_labels.npy'), dl_train_known_labels)
    else:
        print('Loading.')   
        dl_train_known_features = np.load(os.path.join(feat_cache_path, 'dl_train_known_features.npy'))
        dl_train_known_labels = np.load(os.path.join(feat_cache_path, 'dl_train_known_labels.npy'))

    print('TRAIN UNKNOWN')
    if not os.path.exists(os.path.join(feat_cache_path, 'dl_train_unknown_features.npy')) and \
        not os.path.exists(os.path.join(feat_cache_path, 'dl_train_unknown_labels.npy')):
        DF_train_unknown = EK_Dataset_pretrain_framewise_prediction(
                    split['training_known'], split['training_unknown'],
                    train_object_csvpath, train_action_csvpath, 
                    class_key_csvpath, image_data_folder,
                    model_saveloc = None,
                    crop_type='rescale',
                    mode='resnet',
                    prune_target='unknown',
                    resnet_out=False)
        dl_train_unknown = data.DataLoader(DF_train_unknown, 
                        batch_size=64, num_workers=5)
        dl_train_unknown_features = []
        dl_train_unknown_labels = []    
        for batch in tqdm(dl_train_unknown):
            dl_train_unknown_features.append(res101_eq(batch['frame']).squeeze().detach().cpu().numpy())
            dl_train_unknown_labels.extend(batch['noun_label'])
        dl_train_unknown_features = np.vstack(dl_train_unknown_features)
        np.save(os.path.join(feat_cache_path, 'dl_train_unknown_features.npy'), dl_train_unknown_features)
        np.save(os.path.join(feat_cache_path, 'dl_train_unknown_labels.npy'), dl_train_unknown_labels)
    else:
        print('Loading.')
        dl_train_unknown_features = np.load(os.path.join(feat_cache_path, 'dl_train_unknown_features.npy'))
        dl_train_unknown_labels = np.load(os.path.join(feat_cache_path, 'dl_train_unknown_labels.npy'))

    print('TEST UNKNOWN')
    if not os.path.exists(os.path.join(feat_cache_path, 'dl_test_unknown_features.npy')) and \
        not os.path.exists(os.path.join(feat_cache_path, 'dl_test_unknown_labels.npy')):
        DF_test_unknown = EK_Dataset_pretrain_framewise_prediction(
                    split['training_known']+split['training_unknown'], split['testing_unknown'],
                    train_object_csvpath, train_action_csvpath, 
                    class_key_csvpath, image_data_folder,
                    model_saveloc = None,
                    crop_type='rescale',
                    mode='resnet',
                    prune_target='unknown',
                    resnet_out=False)
        dl_test_unknown = data.DataLoader(DF_test_unknown, 
                        batch_size=64, num_workers=5)
        dl_test_unknown_features = []
        dl_test_unknown_labels = []    
        for batch in tqdm(dl_test_unknown):
            dl_test_unknown_features.append(res101_eq(batch['frame']).squeeze().detach().cpu().numpy())
            dl_test_unknown_labels.extend(batch['noun_label'])
        dl_test_unknown_features = np.vstack(dl_test_unknown_features)
        np.save(os.path.join(feat_cache_path, 'dl_test_unknown_features.npy'), dl_test_unknown_features)
        np.save(os.path.join(feat_cache_path, 'dl_test_unknown_labels.npy'), dl_test_unknown_labels)
    else:
        print('Loading.')
        dl_test_unknown_features = np.load(os.path.join(feat_cache_path, 'dl_test_unknown_features.npy'))
        dl_test_unknown_labels = np.load(os.path.join(feat_cache_path, 'dl_test_unknown_labels.npy'))

    # get all the images, concatenate into a single tensor, pass through resnet101. Get the penultimum layer of features
    # get the noun_label
    res101_feats = np.vstack([dl_train_known_features, dl_train_unknown_features, dl_test_unknown_features])
    word_labels = dl_train_known_labels.tolist() + dl_train_unknown_labels.tolist() + dl_test_unknown_labels.tolist()
    res101_labels = np.array([word2idx['+'.join(word.split(' '))] for word in word_labels]) + 1 # offset by 1
    # converting it into a label
    res101_mat = os.path.join(hnd_data_dir, 'res101.mat')
    # put into .mat, res101.mat
    sio.savemat(res101_mat, {'features': res101_feats.transpose(), 'labels':res101_labels})

    # splits into att_splits.mat
    all_seen = np.arange(len(dl_train_known_labels.tolist())+len(dl_train_unknown_labels.tolist()))
    all_unseen = np.arange(len(all_seen), len(word_labels))
    np.random.shuffle(all_seen)
    test_seen = all_seen[:round(0.25*len(all_seen))] + 1 # offset by 1
    trainval = all_seen[round(0.25*len(all_seen)):] + 1 # offset by 1
    np.random.shuffle(all_unseen)
    test_unseen = all_unseen +1 # offset by 1

    att_splits_mat = os.path.join(hnd_data_dir, 'att_splits.mat')
    sio.savemat(att_splits_mat, {'trainval_loc': trainval, 'test_seen_loc': test_seen, 'test_unseen_loc': test_unseen})

    print('Done.')





