from decord import VideoReader
from decord import cpu, gpu
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import subprocess
import re
from utils.custom_dataset import CustomDataset
from utils.data_utils import data_transforms
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm, BCEWithLogitsLoss,BCELoss

"ffmpeg  -i train/videos/1070.mp4 -vframes 2 -vf cropdetect -f null -"
# p1 = subprocess.run(['ls'],capture_output=True, text=True)
# print(p1.stdout)
# def check_crop(fpath):
#     CROP_DETECT_LINE = b'x1:(\d+)\sx2:(\d+)\sy1:(\d+)\sy2:(\d+)'
#     fpath = 'train/videos/2.mp4'
#     p = subprocess.Popen(["ffmpeg", "-i", fpath, "-vf", "cropdetect", "-vframes", "2", "-f", "rawvideo", "-y", "/dev/null"]
#                         , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     infos = p.stderr.read()
    
#     allCrops = re.findall(CROP_DETECT_LINE , infos)
#     print(allCrops)
#     print(int(allCrops[0][0].decode("utf-8")))
#     return allCrops

# CROP_DETECT_LINE = b'w:(\d+)\sh:(\d+)\sx:(\d+)\sy:(\d+)'
# test = b'\d+x\d+'
# fpath = 'train/videos/1873.mp4'
# p = subprocess.Popen(["ffmpeg", "-i", fpath, "-vf", "cropdetect", "-vframes", "2", "-f", "rawvideo", "-y", "/dev/null"]
#                     , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# infos = p.stderr.read()
# allCrops = re.findall(test , infos)
# allCrops = re.findall(CROP_DETECT_LINE , infos) #y1,y2,x1,x2
# print(allCrops)
# output = [int(crop.decode('utf8')) for crop in allCrops[0]]
# #frame = frame[x1:x2,y1:y2,:]
# print(output) #=> 290,560,0,461
# print(infos)


# vr = VideoReader('train/videos/1873.mp4', ctx=cpu(0))

# for i in range(len(vr)):
#     # the video reader will handle seeking and skipping in the most efficient manner
#     frame = vr[i].asnumpy()
#     frame = torch.from_numpy(frame).permute(2,0,1)
#     frame = frame.unsqueeze(0)
#     frame = frame.expand(2,-1,-1,-1)
#     frame = F.crop(frame, 8,298,448,256)
#     frame = frame.permute(1,2,0)
#     plt.figure()
#     plt.imshow(frame) 
#     plt.show() 
#     print(frame.shape)
# print('video frames:', len(vr))

# video_path = "train/videos"
# label_path = "train/label.csv"
# num_frames = 64



# dataset = CustomDataset(video_path, label_path, num_frames, 2, data_transforms["train"], "tubelet")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
# for data in dataset:
#     x,y = data
#     print(y)
# for data in dataloader:
#     x,y = data
#     image = x[0][0,:,:,:]
#     transform = T.ToPILImage()
#     image = transform(image)
#     plt.figure()
#     plt.imshow(image) 
#     plt.show() 
a = torch.tensor([10],dtype=torch.float32)
b = torch.tensor([0], dtype=torch.float32)

loss_fct = BCEWithLogitsLoss()   
normal = BCELoss()
#a = torch.sigmoid(a)
t = loss_fct(a,b)
print(type(t))