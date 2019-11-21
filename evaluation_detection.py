"""
Evaluation script for the detection
"""
import torch.optim as optim
from torchvision import transforms

from src.hierarchical_loss import *
from src.model import *
from src.input_layers import InputLayer

from data.EK_dataloader import *
from data.gt_hierarchy import *
from data.transforms import *
from tqdm import tqdm 

import pickle
from sklearn.metrics import classification_report

DEVICE = 'cuda:0'

# do_coco_evaluation(bounding_box_datasets, predictions, box_only=True,
#                   output_folder='coco_eval_output/', expected_results=None,
#                      expected_results_sigma_tol=None)


def run_metrics(gt_bbox, gt_labels, pred_bbox, pred_labels):

        
    pass 

def evaluate_model(model, dataloader, cache=None, overwrite=True):

    if cache is not None and os.path.exists(cache) and not overwrite:
        with open(cache, 'rb') as f:
            precomputed_results = pickle.load(f)
            pred_bbox = precomputed_results['pred_bbox']
            pred_bbox_class = precomputed_results['pred_bbox_class']
            pred_bbox_gt_class = precomputed_results['pred_bbox_gt_class']
            gt_bbox = precomputed_results['gt_bbox']
            gt_bbox_class = precomputed_results['gt_bbox_class']

    else:

        # all bounding boxes are in corner format
        pred_bbox = []
        pred_bbox_class = []

        pred_bbox_gt_class = [] # just for intermediary evaluations of accuracy, not mAP

        gt_bbox = []
        gt_bbox_class = []

        for i,sample in enumerate(tqdm(dataloader)):
            embeddings = sample['embeddings']
            batch_size = embeddings.shape[0]

            embeddings_original_shape = embeddings.shape
            embeddings = embeddings.reshape((-1, embeddings.shape[-1])).type(torch.FloatTensor).to(DEVICE)
            model = model.to(DEVICE)
            with torch.no_grad():   
                pred = model(embeddings).detach().cpu().numpy()

            # predictions embedildng
            pred = pred.reshape((embeddings_original_shape[0], embeddings_original_shape[1], -1))
            # pred: BATCH X NUM_PRED_BBOXES X 3
            bbox_class_pred = np.argmax(pred, -1)
            import ipdb; ipdb.set_trace()
            pred_bbox_class.extend(bbox_class_pred.tolist())
            # bbox_proposals: 
            bbox_proposals = sample['pred_bboxes']
            pred_bbox.extend(bbox_proposals.numpy().tolist())

            pred_bbox_gt_class.extend(sample['labels'].numpy().tolist())

            ###################### ground truth stuff ########################

            # gt_bboxes
            gt_bboxes = sample['gt_bboxes'].numpy() # 8,6,4 # 6 ground truth bounding boxes, but some are duplicates
            gt_bboxes_corners = gt_bboxes[:, [1, 0, 3, 2]] #  getting the x to the front
            gt_bboxes_corners = np.hstack([gt_bboxes_corners[:, :2],
                gt_bboxes_corners[:, 2:] + gt_bboxes_corners[:, :2]])


            gtc = list(gt_bboxes_corners) # gives list of np_arrays
            unique_gt_bboxes = []
            uniques_idx = []
            for in_batch in gtc:
                unique_gt_bboxes_corners, unique_indices = np.unique(in_batch, axis=0, return_index=True)
                unique_gt_bboxes.append(unique_gt_bboxes_corners.tolist())
                uniques_idx.append(unique_indices.tolist())

            gt_labels = sample['gt_bboxes_label'].numpy()
            gtl = list(gt_labels)
            unique_gt_labels = []
            for idx, in_batch in enumerate(gtl):
                uniques_labels = in_batch[uniques_idx[idx]]
                unique_gt_labels.append(uniques_labels.tolist())

            gt_bbox.extend(unique_gt_bboxes)
            gt_bbox_class.extend(unique_gt_labels)
            if (i+1) % 100 == 0:

                print(classification_report(np.array(pred_bbox_gt_class).flatten(), 
                                            np.array(pred_bbox_class).flatten(),
                                        target_names=['background', 'known', 'unknown']))
                # doing a preliminary save
                precomputed_results = {"pred_bbox": pred_bbox,
                                "pred_bbox_class": pred_bbox_class,
                                "pred_bbox_gt_class": pred_bbox_gt_class,
                                "gt_bbox": gt_bbox,
                                "gt_bbox_class": gt_bbox_class}
                with open(cache, 'wb') as f:
                    pickle.dump(precomputed_results, f)

        import ipdb; ipdb.set_trace()

        precomputed_results = {"pred_bbox": pred_bbox,
                                "pred_bbox_class": pred_bbox_class,
                                "pred_bbox_gt_class": pred_bbox_gt_class,
                                "gt_bbox": gt_bbox,
                                "gt_bbox_class": gt_bbox_class}
        with open(cache, 'wb') as f:
            pickle.dump(precomputed_results, f)


    # now doing mAP analysis

    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch hierarchy discovery model training')
    parser.add_argument('--data', type=str, default='/vision/group/EPIC-KITCHENS',
                        help='path to dataset directory')
    parser.add_argument('--model', type=str, default='models/hierarchy_discovery/hierarchy_detection_run1/net_epoch4.pth',
                        help='path to dataset directory')
    parser.add_argument('--visual_tree_encoder', type=str, default='models/pretraining_tree/framelevel_pred_run15/net_epoch10.pth',
                        help='path to static visual tree encoder')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for the training and validation dataloaders. Default 0.')
    parser.add_argument('--eval_cache_path', type=str, default='eval_detection_cache/V1')

    args = parser.parse_args()

    # loading the dataset of unknown objects
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

    # get known unknown splits. Here, we've trained a model on training_knowns and training_unknowns
    # now we test it on training_knowns and testing_unknowns.
    split = get_known_unknown_split(required_training_knowns='EK_COCO_Imagenet_intersection.txt')
    training_knowns = split['training_known']
    training_unknowns = split['training_unknown']
    testing_unknowns = split['testing_unknown'] # we're testing on this set of unknowns
    
    # loading the tree encoder (now trained wth the unino of training knowns and training unknowns)
    assert os.path.exists(args.model)
    detection_mod = torch.load(args.model)
    detection_mod.eval()
    
    # DF = DatasetFactory(training_knowns, testing_unknowns, train_object_csvpath, 
    #                     train_action_csvpath, class_key_csvpath)
    
    # test_dataset = DF.get_dataset()



    DF = DatasetFactory(training_knowns, training_unknowns, train_object_csvpath, train_action_csvpath, class_key_csvpath)
    training_dataset = DF.get_dataset()


    # making relevant transforms
    inputlayer_rpn = InputLayer(max_num_boxes=None, rpn_conf_thresh=0.0, rpn_only=True)

    # DF_test = EK_Dataset_detection(training_knowns, testing_unknowns,
    #         train_object_csvpath, train_action_csvpath, class_key_csvpath, 
    #         image_data_folder, test_dataset, inputlayer_rpn,
    #         tree_encoder=args.visual_tree_encoder, purpose='test', max_num_boxes=50, max_gt_boxes=6)

    DF_check = EK_Dataset_detection(training_knowns, training_unknowns,
            train_object_csvpath, train_action_csvpath, class_key_csvpath, 
            image_data_folder, training_dataset, inputlayer_rpn,
            tree_encoder=args.visual_tree_encoder, purpose='train', max_num_boxes=25, max_gt_boxes=6)

    # TODO: run this first, then at some point, comment it out
    # DF_test.preprocess_and_cache()
    # TODO: after the previous line is commented out, uncomment the following:

    # DF_test.only_unique_frames()
    # DF_test.restrict_to_cached()

    DF_check.only_unique_frames()
    DF_check.restrict_to_cached()

    # test_dataloader = data.DataLoader(DF_test, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = data.DataLoader(DF_check, batch_size=args.batch_size, num_workers=args.num_workers)

    if not os.path.exists(args.eval_cache_path):
        os.makedirs(args.eval_cache_path)

    # evaluate the model   
    cache_name = \
        '-'.join(args.model.split('/')[-2:])[:-len('.pth')]+ '.pkl'
    cache_path = os.path.join(args.eval_cache_path, cache_name)

    evaluate_model(detection_mod, test_dataloader, cache=cache_path, overwrite=False)

    pass
