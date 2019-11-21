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
from torch.nn.modules.pooling import AvgPool3d

from data.transforms import *

try:
    from data.EK_dataloader import *
except:
    from EK_dataloader import *

USECUDA = True 

# JUST FOR EXPERIMENTATION!
def viz_resnet_frames(dataset_path, visualize_section='training'):
    model_dir =  os.path.dirname(dataset_path)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        assert config['feature_extractor'] == 'resnet'

    with open(os.path.join(model_dir, 'processing_params.pkl'), 'rb') as f:
        processing_params = pickle.load(f)

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    assert 'all_data' in dataset and 'train_indices' in dataset and 'val_indices' in dataset
    assert visualize_section == 'training' or visualize_section == 'validation'

    all_data = dataset['all_data']
    indices = dataset['train_indices' if visualize_section=='training' else 'val_indices']

    processing_params = [processing_params[element] for element in ["output_cache_fullpath", "crop_type", "processed_frame_number","f2bbox", 
                                                                    "image_data_folder", "noun_dict", "knowns", "unknowns", "unknown_lowest_level_label", 
                                                                    "individual_transform", "pairwise_transform"]]
    # hijacking pairwise transforms
    processing_params[-2] = transforms.Compose([Rescale((224, 224)),
                                        Transpose(),
                                        TimeNormalize(16),
                                        BGR2RGB(),
                                        ToTensor()
                                        ])

    unique_indices = list(set(np.array(indices).flatten().tolist()))
    print('{} unique clips'.format(len(unique_indices)))

    avgpool = None
    frames = []
    for idx in tqdm(unique_indices):
        sample_a = all_data[idx]
        processed_pair = EK_Dataset_pretrain_pairwise.process(*([sample_a, sample_a] + processing_params), overwrite=True)
        frames_a = processed_pair['frames_a']

        # processed_pair_frames = EK_Dataset_pretrain_pairwise.process(*([sample_a, sample_a] + processing_params2), overwrite=True)
        # if avgpool is None:
        #     avgpool = AvgPool3d(frames_a.shape[1:])
        # result = avgpool(frames_a).squeeze().data.cpu().numpy()

        frames.append(frames_a)

    return frames 

def viz_resnet_embeddings(dataset_path, visualize_section='training'):
    model_dir =  os.path.dirname(dataset_path)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        assert config['feature_extractor'] == 'resnet'

    with open(os.path.join(model_dir, 'processing_params.pkl'), 'rb') as f:
        processing_params = pickle.load(f)

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    assert 'all_data' in dataset and 'train_indices' in dataset and 'val_indices' in dataset
    assert visualize_section == 'training' or visualize_section == 'validation'

    all_data = dataset['all_data']
    indices = dataset['train_indices' if visualize_section=='training' else 'val_indices']

    processing_params = [processing_params[element] for element in ["output_cache_fullpath", "crop_type", "processed_frame_number","f2bbox", 
                                                                    "image_data_folder", "noun_dict", "knowns", "unknowns", "unknown_lowest_level_label", 
                                                                    "individual_transform", "pairwise_transform"]]
    # hijacking pairwise transforms
    processing_params[-2] = transforms.Compose([Rescale((224, 224)),
                                        Transpose(),
                                        TimeNormalize(16),
                                        BGR2RGB(),
                                        ToTensor(),
                                        NormalizeVideo(),
                                        GetResnetLastLayerFeats(),
                                        # ToTensor() 
                                        ])

    # processing_params2 = processing_params
    # processing_params2[-2] = transforms.Compose([Rescale((224, 224)),
    #                                     Transpose(),
    #                                     TimeNormalize(16),
    #                                     BGR2RGB(),
    #                                     ToTensor(),
    #                                     ])


    unique_indices = list(set(np.array(indices).flatten().tolist()))
    print('{} unique clips'.format(len(unique_indices)))

    avgpool = None
    embeddings = []
    frames = []
    for idx in tqdm(unique_indices):
        sample_a = all_data[idx]
        # import ipdb; ipdb.set_trace()
        processed_pair = EK_Dataset_pretrain_pairwise.process(*([sample_a, sample_a] + processing_params), overwrite=True)
        frames_a = processed_pair['frames_a']
        label_a = processed_pair['noun_label_a']

        # processed_pair_frames = EK_Dataset_pretrain_pairwise.process(*([sample_a, sample_a] + processing_params2), overwrite=True)
        # if avgpool is None:
        #     avgpool = AvgPool3d(frames_a.shape[1:])
        # result = avgpool(frames_a).squeeze().data.cpu().numpy()

        result = np.max(frames_a.data.cpu().numpy(), axis=0)
        # result = frames_a.data.cpu().numpy()
        embeddings.append([result, label_a ])

        # frames.append(processed_pair_frames['frames_a'])

    return embeddings#, frames 

def viz_pretraining_framelevel_pred_embeddings(model, dataset_path, visualize_section='training', num_samples=1000):
    model_dir = os.path.dirname(dataset_path)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(model_dir, 'processing_params.pkl'), 'rb') as f:
        processing_params = pickle.load(f)

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    data = dataset['train_set' if visualize_section=='training' else 'val_set']

    if len(data) > num_samples:
        data = np.random.choice(data, num_samples, replace=False)
    net = torch.load(model)
    net.eval()

    embeddings = []

    l1_keys, l2_keys, l3_keys = get_tree_position_keys(processing_params['knowns']) # assuming tree_file='hierarchyV1.json'

    for sample in tqdm(data):
        processed_frames = EK_Dataset_pretrain_framewise_prediction.process(sample, 
                            processing_params['noun_dict'], processing_params['image_transform'], 
                            processing_params['image_data_folder'], processing_params['knowns'])
        frame = processed_frames['frame']
        label = processed_frames['noun_label']

        if USECUDA:
            frame = frame.type(torch.FloatTensor).to('cuda:0')
            net = net.to('cuda:0')

        with torch.no_grad():

            encoding = net(frame.unsqueeze(0))['embedding'][0].data.cpu().numpy()
            hierarchy_encoding = processed_frames['hierarchy_encoding'].data.cpu().numpy()
            # l1 and l2 labels
            l1_label = l1_keys[(hierarchy_encoding[0])]
            l2_label = l2_keys[(hierarchy_encoding[0], hierarchy_encoding[1])]

            embeddings.append((encoding, label, l2_label, l1_label))

    return embeddings

def viz_pretraining_batchwise_embeddings(model, dataset_path, visualize_section='training'):
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

    net = torch.load(model)
    net.eval()

    processing_params = [processing_params[element] for element in ["output_cache_fullpath", "crop_type", "processed_frame_number","f2bbox", 
                                                                    "image_data_folder", "noun_dict", "knowns", "unknowns", "unknown_lowest_level_label", 
                                                                    "individual_transform", "batchwise_transform"]]
    unique_indices = list(set(np.hstack(indices).tolist()))
    embeddings = []
    for idx in tqdm(unique_indices):
        sample = all_data[idx]
        processed_frames = EK_Dataset_pretrain_batchwise.process(*([[sample]]+processing_params), overwrite=True)
        frames = torch.stack(processed_frames['batch_frames'])
        label = processed_frames['noun_labels'][0] # there is only one frame

        if USECUDA:
            frames = frames.type(torch.FloatTensor).to('cuda:0')
            net = net.to('cuda:0')
        with torch.no_grad():
            encoding = net(torch.cat([frames,frames], 0))[0] # duplication because of squeeze
        embeddings.append((encoding.data.cpu().numpy(), label))

    return embeddings

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
            encoding_a=net(torch.cat([frames_a, frames_a]))
            encoding_b=net(torch.cat([frames_b, frames_b]))

        embeddings.append([(encoding_a.data.cpu().numpy()[0], label_a),
                            (encoding_b.data.cpu().numpy()[0], label_b)])
    return embeddings

def apply_TSNE(embeddings, output_dimensions=2, perplexity=30.0):
    assert output_dimensions in {2,3}, 'output_dimensions neeed to be either 2 or 3'


    uniques = embeddings
    uniques_vecs = np.array([element[0] for element in uniques])
    if np.all([element.size == output_dimensions for element in uniques_vecs]):
        print('Already in required dimensionality, just choosing these.')
        return embeddings

    if not all([element.size==output_dimensions for element in uniques_vecs]):
        embedded = TSNE(n_components=output_dimensions, perplexity=perplexity, verbose=10).fit_transform(uniques_vecs)
        embedded = [(embedded[idx], vec[1]) for idx, vec in enumerate(uniques)]
    else:
        print('Already in required dimensionality, just choosing these.')
        embedded = uniques_vecs
        embedded = [(embedded[idx], vec[1]) for idx, vec in enumerate(uniques)]
    return embedded 
          

def apply_TSNE_pairs(embeddings, output_dimensions=2, perplexity=30.0):
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

    b = viz_pretraining_framelevel_pred_embeddings('models/pretraining_tree/framelevel_pred_run15/net_epoch10.pth',
                            'models/pretraining_tree/framelevel_pred_run15/data_info.pkl', visualize_section = 'training', 
                            num_samples=100000)
    import pickle
    with open('g_embeddings.pkl', 'wb') as f:
        pickle.dump(b, f)

    import ipdb; ipdb.set_trace()

    a = viz_pretraining_batchwise_embeddings('models/pretraining_tree/batchwise_run12/net_epoch0.pth',
                            'models/pretraining_tree/batchwise_run12/data_info.pkl', visualize_section = 'training')

    import ipdb; ipdb.set_trace()


    #resnet_training_embeddings = viz_resnet_embeddings('models/pretraining_tree/pairwise_run0/data_info.pkl', visualize_section='training')
    resnet_validation_input = viz_resnet_embeddings('models/pretraining_tree/pairwise_run0/data_info.pkl', visualize_section='validation')

    import ipdb; ipdb.set_trace()

    training_embeddings = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run0/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run0/data_info.pkl', visualize_section='training')
    val_embeddings  = viz_pretraining_pairwise_embeddings('models/pretraining_tree/pairwise_run0/net_epoch0.pth', 
                        'models/pretraining_tree/pairwise_run0/data_info.pkl', visualize_section='validation')
    import ipdb; ipdb.set_trace()
    print(apply_TSNE(training_embeddings))
