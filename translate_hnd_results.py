""" translate hnd results with this script
"""
import os
import numpy as np

import argparse
from tqdm import tqdm

def read_hnd_results(taxonomy_path, dir_, result_name):
    # load the taxonomy
    assert os.path.exists(taxonomy_path) , '{path} does not exist'.format(path=args.taxonomy)
    taxonomy = np.load(taxonomy_path, allow_pickle=True).item()
    wnid_to_index = taxonomy['wnid_to_index']
    index_to_wnid = {wnid_to_index[wnid]: wnid for wnid in wnid_to_index}
    
    wnids_all = taxonomy['wnids_leaf'] + taxonomy['wnids_novel']
    known_to_parents = taxonomy['wnid_parents']
    known_to_parents = {element: known_to_parents[element][0] for element in known_to_parents if element != '_root_'}
    unknown_to_parents = taxonomy['wnid_novel_to_wnid_known']
    unknown_to_parents = {element: list(unknown_to_parents[element].keys())[0] for element in unknown_to_parents}

    files = os.listdir(dir_)
    try:
        labels_npy = [element for element in files if '_test_labels.npy' in element][0]
    except:
        FileNotFoundError('the test_labels file is not found.')
    try:
        preds_npy = [element for element in files if '_test_allpreds.npy' in element][0]
    except:
        FileNotFoundError('the test_allpreds file is not found.')
    labels_npy = np.load(os.path.join(dir_, labels_npy), allow_pickle = True).item()
    preds_npy = np.load(os.path.join(dir_, preds_npy), allow_pickle=True).item()

    # focusing on the novel
    novel_preds = preds_npy['novel']
    novel_gt = labels_npy['novel']
    # converting novel_gt into a node in the hierarchy
    # TODO: elements in novel_gt are l3 labels, not l2 labels

    l3_gt_wnids = [wnids_all[element] for element in novel_gt] # this is the l2 layer of the ground truth

    keep_index = np.where(np.array(l3_gt_wnids) != 'l3_support')[0]
    novel_gt = novel_gt[keep_index]
    if result_name != 'TD':
        novel_preds = novel_preds[:, keep_index]
        l2_preds_wnids = np.array([[index_to_wnid[element] for element in element2] for element2 in novel_preds]) # this is the l2 layer of the predictions
    else:
        novel_preds = novel_preds[keep_index]
        l2_preds_wnids = np.array([index_to_wnid[element] for element in novel_preds])

    l3_gt_wnids = np.array(l3_gt_wnids)[keep_index].tolist()

    l2_gt_wnids = np.array([unknown_to_parents[l3] for l3 in l3_gt_wnids])
    

    # take the predictions with the best novel accuracy
    if result_name != 'TD':
        layer2_accuracy = [np.mean(l2_gt_wnids==l2_pred) for l2_pred in l2_preds_wnids]
        max_idx = np.argmax(layer2_accuracy)
        l2_preds_wnids = l2_preds_wnids[max_idx]
    else:
        l2_preds_wnids = l2_preds_wnids

    l1_gt_wnids = [known_to_parents[l2] for l2 in l2_gt_wnids]
    l1_preds_wnids = [known_to_parents[l2] for l2 in l2_preds_wnids]

    return l1_gt_wnids, l1_preds_wnids, l2_gt_wnids, l2_preds_wnids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument for evaluation against CVPR-hnd baseline')
    parser.add_argument('--taxonomy', type=str, default='utilities/cvpr2018-hnd/taxonomy/EK/taxonomy.npy') 
    parser.add_argument('--hnd_results_tdloo', type=str, default='utilities/cvpr2018-hnd/train/EK/resnet101/TD+LOO/TD_-1_1e+00_0e+00_cw_1e-02_1e-02/')
    parser.add_argument('--hnd_results_loo', type=str, default='utilities/cvpr2018-hnd/train/EK/resnet101/LOO')
    parser.add_argument('--hnd_results_td', type=str, default='utilities/cvpr2018-hnd/train/EK/resnet101/TD')
    args = parser.parse_args()

    result_dirs = []
    result_names = []
    if args.hnd_results_tdloo is not None:
        result_dirs.append(args.hnd_results_tdloo)
        result_names.append('TD+LOO')
    if args.hnd_results_loo is not None:
        result_dirs.append(args.hnd_results_loo)
        result_names.append('LOO')
    if args.hnd_results_td is not None:
        result_dirs.append(args.hnd_results_td)
        result_names.append('TD')

    # load the training results for the different experiments

    for j, dir_ in enumerate(result_dirs):
        print('----------------------------')
        print('RESULTS FOR {name}'.format(name=result_names[j]))
        print('----------------------------')

        l1_gt_wnids, l1_preds_wnids, l2_gt_wnids, l2_preds_wnids = read_hnd_results(args.taxonomy, dir_, result_names[j])

        layer1_accuracy = np.mean([l1_gt_wnids[i] == l1_preds_wnids[i] for i in range(len(l1_preds_wnids))])
        layer2_accuracy = np.mean([l2_gt_wnids[i] == l2_preds_wnids[i] for i in range(len(l2_preds_wnids))])
        layer1_correct = np.where([l1_gt_wnids[i] == l1_preds_wnids[i] for i in range(len(l1_preds_wnids))])
        layer2_marginal_accuracy = np.mean([l2_gt_wnids[index] == l2_preds_wnids[index] for index in layer1_correct])
        
        print('layer1 accuracy: {}'.format(layer1_accuracy))
        print('layer2 marginal accuracy: {}'.format(layer2_marginal_accuracy))
        print('layer2 accuracy: {}'.format(layer2_accuracy))

        # Layer 1 accuracy

        # Layer 2 accuracy


    pass
