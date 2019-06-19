import os
import torch
import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.misc import imread
import json

# necessary tools for hand position estimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import import Axes3D
from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from hand3d.utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

# running from the top level directory

class HandPositionEstimator(object):
    def __init__(self, model_weight_files=['./utils/hand3d/weights/handsegnet-rhd.pickle', 
            './utils/hand3d/weights/posenet3d-rhd-stb-slr-finetuned.pickle'], visualize=True, 
            visual_save_loc='visualize/handpose_estimation',
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
            os.makedirs(self.cache_loc)
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

        for image_name, image_raw in image_list:
            save_name = os.path.join(self.cache_loc, os.path.basename(image_name)[:-self.extension_length])+'.json'
            if os.path.exists(save_name) and not self.overwrite:
                # loading directly from cache
                with open(save_name, 'r') as f:
                    results.append(json.load(f))

            else:
                image_raw = scipy.misc.imresize(image_raw, (240, 320))
                image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
                
                hand_scoremap_v, image_crop_v, scale_v, center_v,\
                keypoints_scoremap_v, keypoint_coord3d_v = \
                sess.run([self.hand_scoremap_tf, self.image_crop_tf, self.scale_tf, 
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
                image_result = {'image_name': image_name,
                                'binary_mask': np.argmax(hand_scoremap_v, 2),
                                'hand_joints_2d': coord_hw_crop,
                                'hand_joints_3d': keypoint_coord3d_v})
                
                # assuming that there is a 4-character extension like .jpg for the
                # image name
                with open(save_name, 'w') as f:
                    json.dump(image_result, f)

                results.append(image_result)
                if self.visualize:
                    # visualize
                    fig = plt.figure(1, figsize=(10,10))
                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224, projection='3d')
                    ax1.imshow(image_raw)
                    plot_hand(coord_hw, ax1)
                    ax2.imshow(image_crop_v)
                    plot_hand(coord_hw_crop, ax2)
                    ax3.imshow(np.argmax(hand_scoremap_v, 2))
                    plot_hand_3d(keypoint_coord3d_v, ax4)
                    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
                    ax4.set_xlim([-3, 3])
                    ax4.set_ylim([-3, 1])
                    ax4.set_zlim([-3, 3])
                    
                    image_save_name = os.path.join(self.visualize_save_loc, 
                            os.path.basename(image_name))
                    plt.savefig(image_save_name)

class HandDetector(object):
    def __init__(self):
        pass

class HandMeshPredictor(object):
    def __init__(self):
        pass


if __name__=='__main__':
    # load the hand positions and estimate rough hand pose
    HPE = HandPositionEstimator()
    test_images = [['/viz/viz_data/tmp_dataset/P01/P01_01/0000024871.jpg', imread('viz/viz_data/tmp_dataset/P01/P01_01/0000024871.jpg')] ] 
    HPE.process(test_images)
    # generate bounding boxes for each image

    # more find grained hand mesh prediction
