"""
Code to output the embeddings of a model
"""
import os
import torch
import json
from tqdm import tqdm
from sklearn.manifold import TSNE
import pickle
import numpy as np

try:
    from data.EK_dataloader import EK_Dataset_pretrain_pairwise
except:
    from EK_dataloader import EK_Dataset_pretrain_pairwise

USECUDA = True 

def viz_pretraining_pairwise_embeddings(model, dataset_path, visualize_section='training'):
    model_dir = os.path.dirname(dataset_path)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(model_dir, 'processing_params.pkl'), 'rb') as f:
        processing_params = pickle.load(f)

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    assert 'all_data' in dataset and 'train_indices' in dataset and 'val_indices' in dataset
    assert visualize_section=='training' or visualize_section=='validation'

    all_data = dataset['all_data']
    indices = dataset['train_indices' if visualize_section=='training' else 'val_indices']

    processing_params = [processing_params[element] for element in ["output_cache_fullpath", "crop_type", "processed_frame_number","f2bbox", 
                                                                    "image_data_folder", "noun_dict", "knowns", "unknowns", "unknown_lowest_level_label", 
                                                                    "individual_transform", "pairwise_transform"]]

    # loading the model
    net = torch.load(model)
    net.eval()

    embeddings = []
    unique_indices = list(set(np.array(indices).flatten().tolist()))
    print('{} unique clips'.format(len(unique_indices)))
    # running model through the batches
    for idx in tqdm(indices):
        sample_a = all_data[idx[0]]
        sample_b = all_data[idx[1]]
        processed_pair = EK_Dataset_pretrain_pairwise.process(*([sample_a, sample_b] + processing_params))
        frames_a = torch.stack([processed_pair['frames_a']])
        frames_b = torch.stack([processed_pair['frames_b']])

        label_a = processed_pair['noun_label_a']
        label_b = processed_pair['noun_label_b']

        if USECUDA:
            frames_a = frames_a.type(torch.FloatTensor).to('cuda:0')
            frames_b = frames_b.type(torch.FloatTensor).to('cuda:0')
            net = net.to('cuda:0')
        with torch.no_grad():
            encoding_a=net(frames_a)
            encoding_b=net(frames_b)

        embeddings.append([(encoding_a.data.cpu().numpy()[0], label_a),
                            (encoding_b.data.cpu().numpy()[0], label_b)])
    return embeddings

def apply_TSNE(embeddings, output_dimensions=2, perplexity=30.0):
    """
    Args:
        embeddings: list of pairs of embeddings
        output_dimensions: number of dimensions
    Returns:
        lower dimensional representation
    """

    # check if there are already output_dimensions 
    assert output_dimensions in {2,3}, 'output_dimensions neeed to be either 2 or 3'

    uniques = []
    seen = {}
    reference_indices = []
    for pair in embeddings:
        ref_indices = []
        for i in range(2):
            if tuple(pair[i][0].tolist()) not in seen:
                # record index len(uniques)
                ref_indices.append(len(uniques))
                # add to unique and to seen
                seen[tuple(pair[i][0].tolist())] = len(uniques)
                uniques.append((tuple(pair[i][0].tolist()),pair[i][1]))
 
            else:
                # find where it was 
                idx = seen[tuple(pair[i][0].tolist())]
                ref_indices.append(idx)
        reference_indices.append(ref_indices)

    # stacked = []
    # for element in embeddings:
    #     stacked.append(tuple(element[0][0].tolist()))
    #     stacked.append(tuple(element[1][0].tolist()))
    # stacked = np.array(stacked)
    # pass into TSNE

    uniques_vecs = np.array([element[0] for element in uniques])

    if not all([element.size==output_dimensions for element in uniques_vecs]):
        embedded = TSNE(n_components=output_dimensions, perplexity=perplexity, verbose=10).fit_transform(uniques_vecs)
        embedded = [(embedded[idx], vec[1]) for idx, vec in enumerate(uniques)]
    else:
        print('Already in required dimensionality, just choosing these.')
        embedded = uniques_vecs
        embedded = [(embedded[idx], vec[1]) for idx, vec in enumerate(uniques)]

    # embeddings_out = []
    # for i, pair in enumerate(embeddings):
    #     idx_a = seen[tuple(pair[0][0].tolist())]
    #     idx_b = seen[tuple(pair[1][0].tolist())]
    #     embeddings_out.append([(embedded[idx_a], pair[0][1]), 
    #                            (embedded[idx_b], pair[1][1])])

    return embedded, reference_indices

if __name__=='__main__':

    training_embeddings = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run28/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run28/data_info.pkl', visualize_section='training')
    val_embeddings  = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run28/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run28/data_info.pkl', visualize_section='validation')
    import ipdb; ipdb.set_trace()
    print(apply_TSNE(training_embeddings))
