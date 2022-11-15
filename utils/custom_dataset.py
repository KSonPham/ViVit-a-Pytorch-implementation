import decord
from decord import cpu, gpu
import numpy as np
import random

# import torchvision
import os
import torch
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as F



class DecordInit(object):
	"""Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

	def __init__(self, num_threads=2, **kwargs):
		self.num_threads = num_threads
		self.ctx = decord.cpu(0)
		#decord.bridge.set_bridge('torch')
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

class CustomDataset(torch.utils.data.Dataset):
	"""Load the video files
	
	Args:
		data_path (string): Path to dataset.
  		label_path (string): Path to label
		num_frames: Number of input frame to be extracted per video
		transform: data augmentation
		sample_method: tubelet or uniform sampling
  		blackbar_check: check for existance of black bar in input video
	"""

	def __init__(self,
				 data_path,
				 label_path=None,
				 num_frames=64,
				 tubelet_size=2,
				 transform=None,
				 sample_method="tubelet",
     			 blackbar_check=None):
		# self.configs = configs
		self.labels = pd.read_csv(label_path) if label_path != None else None
		self.label_path = label_path
		self.data = sorted(os.listdir(data_path))
		self.data_path = data_path
		self.tubelet_size = tubelet_size
		self.transform = transform
		self.sample_method = sample_method
		self.num_frames = num_frames
		self.v_decoder = DecordInit()
		self.blackbar_check = blackbar_check

	def __getitem__(self, index):
		while True:
			try:
				vid = self.data[index]
				path = os.path.join(self.data_path, vid)
				blackbak_crop = None
				if self.blackbar_check != None:
					blackbak_crop = self.blackbar_check(path)
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)
				sample_length = self.tubelet_size * self.num_frames
				# Sampling video frames
				if self.sample_method == "tubelet":
					rand_end = max(0, total_frames - sample_length - 1)
					begin_index = random.randint(0, rand_end)
					end_index = min(begin_index + sample_length, total_frames)
					assert end_index-begin_index >= sample_length
					frame_indice = np.linspace(begin_index, end_index-1, self.num_frames, dtype=int)
				elif self.sample_method == "uniform_sampling":
					frame_indice = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
				video = v_reader.get_batch(frame_indice).asnumpy()
				del v_reader
				break
			except Exception as e:
				print(e)
				index = random.randint(0, len(self.data) - 1)
	
		# Video align transform: T C H W
		with torch.no_grad():
			if self.label_path != None:
				label = self.labels[self.labels["fname"] == vid]["liveness_score"].item()
				video = torch.tensor(video, dtype=torch.float16)
			else:
				label = None
				video = torch.tensor(video)
			video = video.permute(0,3,1,2)
			video = torch.div(video, 255)
			if blackbak_crop is not None:
				video = F.crop(video,blackbak_crop[3],blackbak_crop[2],blackbak_crop[1],blackbak_crop[0])
			if self.transform is not None:
				video = self.transform(video)

		data_out = (video,label) if label != None else video
		return  data_out
		

	def __len__(self):
		return len(self.data)

	def set_transform(self,transform):
		self.transform = transform
	
	def collate_fn(self, batch):
		return tuple(zip(*batch))






