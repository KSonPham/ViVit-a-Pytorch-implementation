import decord
from decord import cpu, gpu
import numpy as np
import random
import data_transforms as T
# import torchvision
import os
import torch
import pandas as pd

class DecordInit(object):
	"""Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

	def __init__(self, num_threads=1, **kwargs):
		self.num_threads = num_threads
		self.ctx = decord.cpu(0)
		self.kwargs = kwargs
		
	def __call__(self, filename):
		"""Perform the Decord initialization.
		Args:
			results (dict): The resulting dict to be modified and passed
				to the next transform in pipeline.
		"""
		reader = decord.VideoReader(filename,
									ctx=self.ctx,
									num_threads=self.num_threads)
		return reader

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'num_threads={self.num_threads})')
		return repr_str


class ZALO(torch.utils.data.Dataset):
	"""Load the video files
	
	Args:
		annotation_path (string): Annotation file path.
		num_class (int): The number of the class.
		num_samples_per_cls (int): the max samples used in each class.
		target_video_len (int): the number of video frames will be load.
		align_transform (callable): Align different videos in a specified size.
		temporal_sample (callable): Sample the target length of a video.
	"""

	def __init__(self,
				 data_path,
				 label_path,
				 num_frames,
				 objective,
				 transform=None,
				 temporal_sample=None):
		# self.configs = configs
		self.labels = pd.read_csv(label_path)
		self.data = os.listdir(data_path)
		self.annotation_path = data_path
		self.transform = transform
		self.temporal_sample = temporal_sample
		self.target_video_len = num_frames
		self.objective = objective
		self.v_decoder = DecordInit()

		# mask
		# if self.objective == 'mim':
		# 	self.mask_generator = CubeMaskGenerator(input_size=(self.target_video_len//2,14,14),min_num_patches=16)

	def __getitem__(self, index):
		while True:
			try:
				vid = self.data[index]
				path = os.path.join(self.annotation_path, vid)
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)
				
				# Sampling video frames
				start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
				assert end_frame_ind-start_frame_ind >= self.target_video_len
				frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
				video = v_reader.get_batch(frame_indice).asnumpy()
				del v_reader
				break
			except Exception as e:
				print(e)
				index = random.randint(0, len(self.data) - 1)
		
		# Video align transform: T C H W
		with torch.no_grad():
			video = torch.from_numpy(video).permute(0,3,1,2)
			if self.transform is not None:
				if self.objective == 'mim':
					pre_transform, post_transform = self.transform
					video = pre_transform(video) # align shape
				else:
					video = self.transform(video)

		# Label (depends)
		if self.objective == 'mim':
			# old version
			'''
			mask, cube_marker = self.mask_generator() # T' H' W'
			label = np.stack(list(map(extract_hog_features, video.permute(0,2,3,1).numpy())), axis=0) # T H W C -> T H' W' C'
			'''
			# new version
			pass
			# mask, cube_marker = self.mask_generator() # T' H' W'
			# hog_inputs = video.permute(0,2,3,1).numpy()
			# hog_features = np.zeros((self.target_video_len,14,14,2*2*3*9))
			# # speed up the extraction of hog features
			# for marker in cube_marker: # [[start, span]]
			# 	start_frame, span_frame = marker
			# 	center_index = start_frame*2 + span_frame*2//2 # fix the temporal stride to 2
			# 	hog_features[center_index] = extract_hog_features(hog_inputs[center_index])
			# label = hog_features
		else:
			label = self.labels[self.labels["fname"] == vid]["liveness_score"].item()
		
		if self.objective == 'mim':
			pass
			# if self.transform is not None:
			# 	video = post_transform(video) # to tensor & norm
			# return video, numpy2tensor(label), numpy2tensor(mask), cube_marker
		else:
			return video, label

	def __len__(self):
		return len(self.data)
	
	def collate_fn(self, batch):
		return tuple(zip(*batch))

video_path = "train/videos"
label_path = "train/label.csv"
num_frames = 32
frame_interval = num_frames // 2
video_decoder = DecordInit()
temporal_sample = T.TemporalRandomCrop(num_frames*frame_interval)

mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
train_transform = T.create_video_transform(
	objective="supervised",
	input_size=224,
	is_training=True,
	hflip=0.5,
	interpolation='bicubic',
	mean=mean,
	std=std)
dataset = ZALO(video_path, label_path, num_frames, "supervised", train_transform, temporal_sample)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=dataset.collate_fn)
for i in range(len(dataset)):
	print(dataset[i][1])
