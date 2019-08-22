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
        assert 'frames' in d, 'need "frames" in d' 
        new_d = d.copy()
        frames = new_d['frames'].transpose(0,1)
        with torch.no_grad():
            new_d['frames']=self.model(frames)
        new_d['frames']=new_d['frames'].transpose(0,1)
        return new_d

class GetResnetLastLayerFeats(object):
    def __init__(self):
        # instantiate network
        self.model = models.resnet18(pretrained=True)
        self.model.eval() 
        # self.model = nn.Sequential(*list(self.model.children())[:-2])

    def __call__(self, d):
        # import ipdb; ipdb.set_trace()

        assert 'frames' in d, 'need "frames" in d' 
        new_d = d.copy()
        frames = new_d['frames'].transpose(0,1)
        with torch.no_grad():
            new_d['frames']=self.model(frames)
        # new_d['frames']=new_d['frames'].transpose(0,1)
        # import ipdb;ipdb .set_trace() 
        return new_d

if __name__=='__main__':
    # first rescale
    # transpose 
    # timenormalize
    composed_transfs = transforms.Compose([Rescale((200,200)),
                                            Transpose(),
                                            TimeNormalize(10),
                                            ToTensor()])

    gr=GetResnetFeats()

