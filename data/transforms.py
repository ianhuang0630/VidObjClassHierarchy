"""
Implementations of different preprocessing transforms given to the dataloader
"""
from torchvision import transforms

class Transpose(object):
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
	    	frames_resized.append(cv2.resize(frame, self.output_size))
	    frames_resized = np.stack(frames_resized, axis=3)

	    # TODO
	    # reshaping and resizing every single image
	    new_d['frames'] = frames_resized
	    return new_d

class TimeNormalize(object):
	def __init__(self, num_frames):
		self.num_frames = num_frames

	def __call__(self, d):
		# reformatting the d['frames'] and normalizing it 
		# assuming tha tth frames are of shape
		# 11 1080 1920 3
		new_d = d.copy()
		new_d['frames']= new_d['frames'][np.round(np.linspace(0,len(new_d['frames'])-1, num=self.num_frames)), :, :, :]
		return new_d

class ToTensor(object):
	def __init__(self):
		pass
	def __call__(self, d):
		new_d = d.copy()
		for key_ in new_d:
			if type(new_d[key_]) is np.ndarray:
				new_d[key_] = torch.from_numpy(new_d[key_])
		return new_d

if __name__=='__main__':
    # first rescale
	# transpose 
	# timenormalize
	composed_transfs = transforms.Compose([Rescale((200,200)),
											Transpose(),
											TimeNormalize(10),
											ToTensor()])

