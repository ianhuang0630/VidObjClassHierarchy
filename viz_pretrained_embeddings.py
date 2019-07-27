"""
Code to output the embeddings of a model
"""
import os
import torch
import json
from tqdm import tqdm
import pickle

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
        
        embeddings.append([(encoding_a.data.cpu().numpy(), label_a), (encoding_b.data.cpu().numpy(), label_b)])
    return embeddings

if __name__=='__main__':

    training_embeddings = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run20/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run20/data_info.pkl', visualize_section='training')
    val_embeddings  = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run20/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run20/data_info.pkl', visualize_section='validation')
    import ipdb; ipdb.set_trace()

