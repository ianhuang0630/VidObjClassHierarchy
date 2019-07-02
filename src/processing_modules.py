import os
import torch
import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.misc import imread, imresize
import json
import pickle
from tqdm import tqdm
# necessary tools for hand position estimation
# import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'utilities/hand3d/')

from mpl_toolkits.mplot3d import Axes3D
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork

# necessary tools for hand mesh prediction
sys.path.insert(0, 'utilities/obman_train')
import cv2
from PIL import Image

from handobjectdatasets.queries import TransQueries, BaseQueries
from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.preprocess import prepare_input, preprocess_frame


# running from the top level directory

class HandPositionEstimator(object):
    
    def __init__(self, 
            model_weight_files=['./utilities/hand3d/weights/handsegnet-rhd.pickle', 
            './utilities/hand3d/weights/posenet3d-rhd-stb-slr-finetuned.pickle'], 
            visualize=False, 
            visualize_save_loc='visualize/handpose_estimation',
            cache_loc='cache/handpose_estimation', image_extension='.jpg',
            overwrite=False):

        self.extension_length = len(image_extension)
        self.model_weight_files = model_weight_files 
        self.visualize = visualize
        self.visualize_save_loc = visualize_save_loc
        if self.visualize:
            if not os.path.exists(self.visualize_save_loc):
                os.makedirs(self.visualize_save_loc, exist_ok=True) 
        self.cache_loc = cache_loc
        if not os.path.exists(self.cache_loc):
            os.makedirs(self.cache_loc, exist_ok=True)
        self.overwrite = overwrite 

        # input place holders 
        self.image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
        self.hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
        self.evaluation = tf.placeholder_with_default(True, shape=())
        
        # building network      
        self.net = ColorHandPose3DNetwork()
        self.hand_scoremap_tf, self.image_crop_tf, self.scale_tf, self.center_tf,\
        self.keypoints_scoremap_tf, self.keypoint_coord3d_tf = \
        self.net.inference(self.image_tf, self.hand_side_tf, self.evaluation)

        # Start TF
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))

        self.net.init(self.sess, weight_files=self.model_weight_files)

    def process(self, image_list):
        """
        Args:
            image_list: list of tuples, first item being the name/path of the
                image, and the second item being a RGB matrix.
        """
        results = []
        print('Extracting masks...')
        for image_name, image_raw in tqdm(image_list):
            
            save_name = os.path.join(self.cache_loc,
                    ('#'.join(image_name.split('/')[-3:])[:-self.extension_length])+'.pkl')
            
            if os.path.exists(save_name) and not self.overwrite:
                # loading directly from cache
                with open(save_name, 'rb') as f:
                    results.append(pickle.load(f))

            else:
                image_raw_shape = image_raw.shape[:2] 
                image_raw = imresize(image_raw, (240, 320))
                image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
                
                hand_scoremap_v, image_crop_v, scale_v, center_v,\
                keypoints_scoremap_v, keypoint_coord3d_v = \
                self.sess.run([self.hand_scoremap_tf, self.image_crop_tf, self.scale_tf, 
                        self.center_tf, self.keypoints_scoremap_tf, self.keypoint_coord3d_tf],
                                        feed_dict={self.image_tf: image_v})

                hand_scoremap_v = np.squeeze(hand_scoremap_v)
                image_crop_v = np.squeeze(image_crop_v)
                keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
                keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
                
                # post processing
                image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
                coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
                coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

                # return
                # TODO: these coordinates are all normalized with respect to
                # the rescaled image. aprameters would have to be scaled back.
                image_result = {'image_name': image_name,
                                'original_shape': image_raw_shape,
                                'confidence': hand_scoremap_v,
                                'binary_mask': np.argmax(hand_scoremap_v, 2),
                                'hand_joints_2d': coord_hw_crop,
                                'hand_joints_3d': keypoint_coord3d_v}
                
                # assuming that there is a 4-character extension like .jpg for the
                # image name
                with open(save_name, 'wb') as f:
                    pickle.dump(image_result, f)

                results.append(image_result)
                # if self.visualize:
                #     # visualize
                #     fig = plt.figure(1, figsize=(10,10))
                #     ax1 = fig.add_subplot(221)
                #     ax2 = fig.add_subplot(222)
                #     ax3 = fig.add_subplot(223)
                #     ax4 = fig.add_subplot(224, projection='3d')
                #     ax1.imshow(image_raw)
                #     plot_hand(coord_hw, ax1)
                #     ax2.imshow(image_crop_v)
                #     plot_hand(coord_hw_crop, ax2)
                #     ax3.imshow(np.argmax(hand_scoremap_v, 2))
                #     plot_hand_3d(keypoint_coord3d_v, ax4)
                #     ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
                #     ax4.set_xlim([-3, 3])
                #     ax4.set_ylim([-3, 1])
                #     ax4.set_zlim([-3, 3])
                #     
                #     image_save_name = os.path.join(self.visualize_save_loc, 
                #             os.path.basename(image_name))
                #     plt.savefig(image_save_name)
        return results

class HandPositionEstimator2(object):
    """ Implementation using different package than HandPositionEstimator2
    """
    def __init__(self):
        pass

    def process(self, image_list):
        pass

class HandDetector(object):
    def __init__(self, margin = [20, 20], orderby = 'total_confidence',
                    area_threshold = 0, average_confidence_threshold = 0.0,
                    cache_loc='cache/hand_bounding_boxes', image_extension='.jpg',
                    overwrite=False):
        """
        Args:
            margin: tuple of the margin expected around each contguous section
        """
        self.margin = margin
        self.orderby = orderby
        assert self.orderby in ['total_confidence', 'area'], \
                'orderby option is invalid, please either choose "total_confidence" or area'
        self.area_threshold = area_threshold
        self.average_confidence_threshold = average_confidence_threshold
        self.cache_loc = cache_loc
        if not os.path.exists(self.cache_loc):
            os.makedirs(self.cache_loc, exist_ok=True)
        self.overwrite = overwrite 
        self.extension_length = len(image_extension) 
        self.recursion_depth = 0
        self.max_recursion_depth = sys.getrecursionlimit()

    def find_contiguous_areas(self, mask, position):
        """
        Args:
            mask: boolean np array
            position: list of indices
        """
        # TODO: keep track of number of layers
        self.recursion_depth += 1
        followup = None

        assert len(position) ==2, 'only two coordinates allowed.'
        this_island = []
        if not mask[position[0], position[1]]:
            self.visited[position[0], position[1]] = 1
        elif mask[position[0], position[1]] and not self.visited[position[0], position[1]]:
            self.visited[position[0], position[1]] = 1 # we've now visited this 
            this_island.append(position)
            # now we look at the neighbors
            for i in range(max(0, position[0]-1), min(mask.shape[0]-1, position[0]+1)+1):
                for j in range(max(0, position[1]-1), min(mask.shape[1]-1, position[1]+1)+1):
                    if i != position[0] and j != position[1] and self.recursion_depth < self.max_recursion_depth:
                        neighbor_islands, followup = self.find_contiguous_areas(mask, (i,j))
                        this_island.extend(neighbor_islands)
                    elif self.recursion_depth == self.max_recursion_depth:
                        # return this current position as a follow-up later on,
                        if followup is None:
                            followup = (position[0], position[1])
                        self.visited[followup[0], followup[1]] = 0 # set as unvisited so that followup is valid
                        # stop iterating and return this_island as is.
                        return this_island, followup
        return this_island, followup 

    def process(self, masks):
        """
        Very naive crop of the hand region based on binary masks provided
        Args:
            masks: list of tuple, first element being the image name, and
                second element being the binary_mask, third element being
                the confidence matrix for hand vs. non-hand.
        """
        hands = [] 
        # finding the two largest contiguous area
        print('extracting bounding boxes')
        for mask_tuple in tqdm(masks):
            image_name = mask_tuple[0]
            mask = mask_tuple[1]

            confidence = np.exp(mask_tuple[2][:,:,1])/ np.sum(np.exp(mask_tuple[2]),2)
            image_raw_shape = mask_tuple[3] 
            # save_name
            save_name = os.path.join(self.cache_loc,
                    ('#'.join(image_name.split('/')[-3:])[:-self.extension_length])+'.pkl')
            # check if file already exists
            if os.path.exists(save_name) and not self.overwrite:
                with open(save_name, 'rb') as f:
                    hands.append(pickle.load(f)) 
            else:
                # se tall unvisisted
                self.visited = np.zeros_like(mask)
                contiguous_sets = [] 
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if not self.visited[i,j]:
                            contiguous_set, followup = self.find_contiguous_areas(mask, (i,j))
                            followup_sets = []
                            self.recursion_depth = 0
                            while followup is not None:
                                followup_set, followup = self.find_contiguous_areas(mask, followup) 
                                self.recursion_depth = 0
                                followup_sets.append(followup_set)
                            if len(followup_sets)>0: 
                                contiguous_set = list(
                                        set.union(*([set(element) for element in followup_sets]+[set(contiguous_set)])))
                            # resetting recursion_depth
                            if len(contiguous_set) > 0:
                                contiguous_sets.append(contiguous_set)
                
                bounding_boxes = []

                for set_ in contiguous_sets:
                    bottom_y = max([element[0] for element in set_])
                    top_y = min([element[0] for element in set_])
                    assert bottom_y >= top_y, 'bottom_y < top_y'
                    right_x = max([element[1] for element in set_])
                    left_x = min([element[1] for element in set_])
                    assert right_x >= left_x, 'right_x < left_x'
                    
                    x_center = np.mean([left_x, right_x])
                    y_center = np.mean([bottom_y, top_y])
                    
                    margin_bottom_y = min(bottom_y +  self.margin[0], mask.shape[0]-1)
                    margin_top_y = max(top_y - self.margin[0], 0)
                    margin_right_x= min(right_x + self.margin[1], mask.shape[1]-1)
                    margin_left_x = max(left_x - self.margin[1], 0)
                    assert margin_right_x >= margin_left_x

                    # calculating overall confidence 
                    total_confidence = sum([confidence[element[0], element[1]] for element in set_])
                    area = len(set_)
                    
                    # verify that the object is at least above some area threshold
                    if area > self.area_threshold:    
                        bounding_boxes.append(((margin_bottom_y, margin_top_y, 
                            margin_right_x, margin_left_x), total_confidence, area,
                            x_center, y_center))
                    
                # depending on the option in the ordering, we can either order by
                # total_confidence or area.
                if self.orderby == 'area':
                    # order by area, second element in the tuples
                    sorted_bounding_boxes = sorted(bounding_boxes, key=lambda x: x[2], reverse=True)
                elif self.orderby == 'total_confidence':
                    # order by total_confidence, second element in the tuples
                    sorted_bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True) 
                     
                if len(sorted_bounding_boxes) >= 2:
                    hand_bounding_boxes = sorted_bounding_boxes[:2]
                    left_to_right = sorted(hand_bounding_boxes, key=lambda x: x[3])
                    
                    hand = {'left': self.rescale(self.format_bounding_box(left_to_right[0]), image_raw_shape),
                            'right': self.rescale(self.format_bounding_box(left_to_right[1]), image_raw_shape)}
                elif len(sorted_bounding_boxes) == 1:
                    assert len(sorted_bounding_boxes[0]) ==5, 'image: {} , {}'.format(image_name, sorted_bounding_boxes)
                    if sorted_bounding_boxes[0][3] >= (mask.shape[1]-1)/2.0: # if the midpoint is more than halfway
                        hand = {'right': self.rescale(self.format_bounding_box(sorted_bounding_boxes[0]), image_raw_shape)}
                    else:
                        hand = {'left': self.rescale(self.format_bounding_box(sorted_bounding_boxes[0]), image_raw_shape)}
                else:
                    hand = {}
                # TODO: rescaling back to original image 

                hand_result = {'image_name': image_name, 'hand': hand}
                # appending to hands
                hands.append(hand_result)
                
                # save into cache
                with open(save_name, 'wb') as f:
                    pickle.dump(hand_result, f)
                
        return hands

    def format_bounding_box(self, tup):
        return {'bottom_y': tup[0][0], 'top_y': tup[0][1], 'right_x': tup[0][2], 'left_x': tup[0][3],
                'total_confidence': tup[1], 'boolean_area': tup[2],
                'boolean_x_center': tup[3], 'boolean_y_center': tup[4]}
    
    def rescale(self, dict_, image_raw_shape):
        dict_2 = dict_.copy()
        dict_2['bottom_y'] = np.round(dict_['bottom_y']/ 240.0 * image_raw_shape[0])
        dict_2['top_y'] = np.round(dict_['top_y']/ 240.0 * image_raw_shape[0])
        dict_2['left_x'] = np.round(dict_['left_x']/ 320.0 * image_raw_shape[1])
        dict_2['right_x'] = np.round(dict_['right_x']/ 320.0 * image_raw_shape[1])
        
        return dict_2

class HandMeshPredictor(object):
    def __init__(self, 
            resume_checkpoint='utilities/obman_train/release_models/obman/checkpoint.pth.tar',
            mano_root='utilities/obman_train/misc/mano',
            no_beta=True,cache_loc='cache/hand_mesh', image_extension='.jpg',
            overwrite=False ):
    
        self.overwrite = overwrite
        self.extension_length = len(image_extension)
        self.cache_loc = cache_loc
        if not os.path.exists(self.cache_loc):
            os.makedirs(self.cache_loc, exist_ok=True)

        self.resume = resume_checkpoint
        self.checkpoint = os.path.dirname(self.resume)
        with open(os.path.join(self.checkpoint, 'opt.pkl'), 'rb') as opt_f:
            self.opts = pickle.load(opt_f)
        self.no_beta = no_beta 
        self.mano_root = mano_root
        self.model = reload_model(self.resume, self.opts, 
                mano_root=self.mano_root, no_beta=self.no_beta)
        self.model.eval()
        # model should be loaded now
    

    def forward_pass_3d(self, model, input_image, pred_obj=True):
        sample = {}
        sample[TransQueries.images] = input_image
        sample[BaseQueries.sides] = ["left"]
        sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
        sample["root"] = "wrist"
        if pred_obj:
            sample[TransQueries.objpoints3d] = input_image.new_ones(
                (1, 600, 3)
            ).float()
        _, results, _ = model.forward(sample, no_loss=True)

        return results

    def process (self, image_list):
        """
        Args:
            image_list: list of tuples, where first element is an image_name,
                second element is a dictionary with the hand bounding_boxes 
                as well as other information
        """
        hand_mesh_list = []
        print('Extracting hand pose and hand mesh...')
        for image_name, hand_info in tqdm(image_list):
            save_name = os.path.join(self.cache_loc,
                    ('#'.join(image_name.split('/')[-3:])[:-self.extension_length])+'.pkl')
            
            if os.path.exists(save_name) and not self.overwrite:
                with open(save_name, 'rb')  as f :
                    hand_mesh_list.append(pickle.load(f))   
            else:
                hand_mesh = {}
                # crop the image for the left hand
                # pass through the model
                for which_hand in ['left', 'right']:
                    if which_hand in hand_info:
                        image_raw = cv2.imread(image_name)
                        # cropping the hand
                        crop = image_raw[int(hand_info[which_hand]['top_y']):int(hand_info[which_hand]['bottom_y'])+1, 
                                    int(hand_info[which_hand]['left_x']):int(hand_info[which_hand]['right_x'])+1, :]
                        frame= preprocess_frame(crop)
                        img = Image.fromarray(frame.copy())
                        hand_crop = cv2.resize(np.array(img), (256, 256)) 
                        
                        if which_hand == 'left':
                            hand_image = prepare_input(hand_crop, flip_left_right=False)
                        elif which_hand == 'right':
                            hand_image= prepare_input(hand_crop, flip_left_right=True)

                        output = self.forward_pass_3d(self.model, hand_image)
                        verts = output['verts'].cpu().detach().numpy()[0]
                        joints = output['joints'].cpu().detach().numpy()[0]
                        hand_mesh[which_hand] = {'verts': verts, 'joints': joints}
                
                hand_mesh_list.append(hand_mesh)
                # save into cache
                with open(save_name, 'wb') as f:
                    pickle.dump(hand_mesh, f)
        return hand_mesh_list                

if __name__=='__main__':
    # load the hand positions and estimate rough hand pose
    HPE = HandPositionEstimator(overwrite=True)
    test_images = [['viz/viz_data/tmp_dataset/P01/P01_01/0000024871.jpg', 
        imread('viz/viz_data/tmp_dataset/P01/P01_01/0000024871.jpg')] ] 
    results = HPE.process(test_images)
    input_ = [(element['image_name'], element['binary_mask'], element['confidence'], 
        element['original_shape']) for element in results]
    # generate bounding boxes for each image
    HD = HandDetector(overwrite=True)
    hand_bounding_box_results = HD.process(input_)
    # more find grained hand mesh prediction
    input_mesh = [(element['image_name'], element['hand']) for element in hand_bounding_box_results]

    HMP = HandMeshPredictor()
    HMP.process(input_mesh)

