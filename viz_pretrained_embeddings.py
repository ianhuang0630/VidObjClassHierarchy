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
    for idx in tqdm(unique_indices):
        sample = all_data[idx]
        processed_pair = EK_Dataset_pretrain_pairwise.process(*([sample, sample] + processing_params))
        frames_a = torch.stack([processed_pair['frames_a']])
        label_a = processed_pair['noun_label_a']

        if USECUDA:
            frames_a = frames_a.type(torch.FloatTensor).to('cuda:0')
            net = net.to('cuda:0')
        with torch.no_grad():
            encoding_a=net(frames_a)
        
        embeddings.append((encoding_a.data.cpu().numpy()[0], label_a))
    return embeddings

def apply_TSNE(embeddings, output_dimensions=2):
    """
    Args:
        embeddings: list of embeddings
        output_dimensions: number of dimensions
    Returns:
        lower dimensional representation
    """

    # check if there are already output_dimensions 
    assert output_dimensions in {2,3}, 'output_dimensions neeed to be either 2 or 3'
    embedding_matrix = np.stack([element[0] for element in embeddings])

    # pass into TSNE
    embedded = TSNE(n_components=output_dimensions, verbose=10).fit_transform(embedding_matrix)
    embeddings_out = [(vec, embeddings[idx][1]) for idx, vec in enumerate(list(embedded))]

    return embeddings_out

if __name__=='__main__':

    training_embeddings = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run20/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run20/data_info.pkl', visualize_section='training')
    val_embeddings  = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run20/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run20/data_info.pkl', visualize_section='validation')
    import ipdb; ipdb.set_trace()

