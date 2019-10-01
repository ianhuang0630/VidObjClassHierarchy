"""
Implementations of different preprocessing transforms given to the dataloader
"""
import torch.nn as nn 
from torchvision import transforms
import torchvision.models as models
import torch
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, 'src')
from i3d import InceptionI3d

class Transpose(object):
    """ TUrns H W 3 T  --> 3 T 1080 1920
    """
    def __init__(self):
        pass
    def __call__(self, d):
        # 1080, 1920, 3, 11 -->  3, 11, 1080, 1920
        assert 'frames' in d, 'need "frames" in d'
        new_d = d.copy()
        new_d['frames'] = new_d['frames'].transpose([2, 3, 0, 1])
        return new_d

class Rescale(object):
    def __init__(self, output_size):
        """
        Args: 
            output_size: size per frame
        """
        self.output_size = output_size

    def __call__(self, d):
        assert 'frames' in d, 'need "frames" in d'
        new_d = d.copy()
        # transpose first
        # 1080, 1920, 3, 11 -->  11, 1080, 1920, 3
        frames = new_d['frames']
        frames = frames.transpose([3,0,1,2])
        frames_resized = []
        for frame in frames:
            frames_resized.append(cv2.resize(frame, self.output_size[::-1]))
        frames_resized = np.stack(frames_resized, axis=3)
        new_d['frames'] = frames_resized
        # output; 1080, 1920, 3, 11
        
        # import ipdb; ipdb.set_trace()
        return new_d


class FeatureNormalize(object):
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        if len(self.means) != len(self.stds):
            raise ValueError('Mean vector and std vector should have same length')

    def __call__(self, arr):
        assert arr.shape[-1] == len(self.means)
        return (arr - self.means)/self.stds 


class TimeStandardize(object):
    def __init__(self, target_time_dim):
        self.target_time_dim = target_time_dim

    def __call__(self, arr):
        """
        Args :
            arr is a numpy array with dimension 0 being the temporal dimension
        """
        indices = np.round(np.linspace(0, len(arr)-1, num=self.target_time_dim)).astype(np.int)
        return arr[indices]

class BboxUnitScale(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

        self.largest_values = np.array([image_width, image_height, image_width, image_height]*2)

    def __call__(self, bboxes):
        # every timestep has a feature of size 8
        # left_x, bottom_y, right_x, top_y (bottom_y >= top_y)
        return bboxes/self.largest_values
        
class I3D_feats(object):
    def __init__(self, weights_loc = 'models/i3d_pretrained/rgb_i3d_pretrained.pt',
                    device='cuda:0'):
        self.device = device
        self.weights_loc = weights_loc

        i3d  = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(157)
        i3d.load_state_dict(torch.load('models/i3d_pretrained/rgb_i3d_pretrained.pt'))
        self.i3d = i3d.to(self.device)

        
    def __call__(self, frames):
        # frames: T, W, H, 3
        # TO: 1, 3, T, 224, 224
        element2s = []
        for element in frames:
            element2 = cv2.resize(element, (224, 224))
            element2s.append(element2)
        
        frames2 = np.array([np.stack(element2s).transpose([3, 0, 1, 2])])

        with torch.no_grad():
            frames2 = torch.Tensor(frames2)
            frames2 = frames2.to(self.device)
            i3d_processed = self.i3d(frames2)
            if len(i3d_processed.shape) == 2:
                i3d_processed = i3d_processed.unsqueeze(2)

        return i3d_processed

class TimeNormalize(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, d):
        # reformatting the d['frames'] and normalizing it 
        # assuming tha tth frames are of shape
        # 11 1080 1920 3
        new_d = d.copy()
        frames = new_d['frames'].transpose([1, 0, 2, 3])
        new_d['frames']= frames[np.round(np.linspace(0,len(frames)-1, num=self.num_frames)).astype(np.int), :, :, :]
        new_d['frames'] = new_d['frames'].transpose([1, 0, 2, 3])
        # import ipdb; ipdb.set_trace()

        return new_d

class ToTensor(object):
    def __init__(self):
        self.tf = transforms.ToTensor()

    def __call__(self, d):
        new_d = d.copy()
        for key_ in new_d:
            if type(new_d[key_]) is np.ndarray: 
                # import ipdb; ipdb.set_trace()

                if key_ == 'hierarchy_encoding' or key_=='dist_matrix':
                    new_d[key_] = torch.from_numpy(new_d[key_])

                else:
                    pil_tf = []

                    for timestep in range(new_d[key_].shape[1]):
                        pil_tf.append(self.tf(Image.fromarray(new_d[key_][:,timestep,:,:].transpose([1,2,0]))))

                    new_d[key_] = torch.Tensor(np.stack(pil_tf).transpose([1,0,2,3]))

            elif type(new_d[key_]) is int:
                new_d[key_] = torch.Tensor([new_d[key_]])

        # import ipdb; ipdb.set_trace()
        return new_d

class NormalizeVideo(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def __call__(self, d):
        # import ipdb; ipdb.set_trace()
        assert 'frames' in d, 'need "frames" in d'
        new_d = d.copy() # 3, 11, H, W 
        frames = new_d['frames']
        frames = frames.transpose(0,1).type(torch.FloatTensor)
        normalized_frames = []
        for frame in frames:
            normalized = self.normalize(frame)
            normalized_frames.append(normalized)
        normalized_frames= torch.stack(normalized_frames, dim=3).transpose(2,3).transpose(1,2)

        new_d['frames'] = normalized_frames # 3, 11, H, W 
        # import ipdb; ipdb.set_trace()
        return new_d
        
class BGR2RGB(object):
    def __init__ (self):
        pass

    def __call__(self, d):
        assert 'frames' in d, 'need "frames" in d'
        new_d = d.copy()
        new_d['frames'] = new_d['frames'][[2, 1, 0], :, :, :]
        return new_d

class GetResnetFeats(object):
    def __init__(self):
        # instantiate network
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.eval()

    def __call__(self, d):
        if type(d) is dict:
            assert 'frames' in d, 'need "frames" in d' 
            new_d = d.copy()
            frames = new_d['frames'].transpose(0,1)
            with torch.no_grad():
                new_d['frames']=self.model(frames)
            new_d['frames']=new_d['frames'].transpose(0,1)
            return new_d

        elif type(d) is torch.Tensor:
            with torch.no_grad():
                return self.model(d.unsqueeze(0))[0]

class GetResnetLastLayerFeats(object):
    def __init__(self):
        # instantiate network
        self.model = models.resnet18(pretrained=True)
        self.model.eval() 
        # self.model = nn.Sequential(*list(self.model.children())[:-2])

    def __call__(self, d):
        # import ipdb; ipdb.set_trace()
        if type(d) is dict:
            assert 'frames' in d, 'need "frames" in d' 
            new_d = d.copy()
            frames = new_d['frames'].transpose(0,1)
            with torch.no_grad():
                new_d['frames']=self.model(frames)
            # new_d['frames']=new_d['frames'].transpose(0,1)
            # import ipdb;ipdb .set_trace() 
            return new_d

        elif type(d) is torch.Tensor:
            with torch.no_grad():
                return self.model(d.unsqueeze(0))[0]

if __name__=='__main__':
    # first rescale
    # transpose 
    # timenormalize
    composed_transfs = transforms.Compose([Rescale((200,200)),
                                            Transpose(),
                                            TimeNormalize(10),
                                            ToTensor()])

    gr=GetResnetFeats()

