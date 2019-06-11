from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
from scipy.misc import imread
import matplotlib.pyplot as plt

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()
        
    # cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    # get image
    image_np = imread(args.video_source) # RGB matrix
    num_hands_detect = 2
    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
    im_height, im_width = image_np.shape[0], image_np.shape[1]
    
    plt.imshow(image_np)
    for i in range(num_hands_detect):
        if scores[i] > args.score_thresh:
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            plt.scatter([left, left, right, right],[top, bottom, top, bottom])
            print(left, right, top, bottom)
    plt.show()
