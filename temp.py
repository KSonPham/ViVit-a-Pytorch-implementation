from decord import VideoReader
from decord import cpu, gpu
import torchvision.transforms as T
import matplotlib.pyplot as plt

import subprocess
import re
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

CROP_DETECT_LINE = b'x1:(\d+)\sx2:(\d+)\sy1:(\d+)\sy2:(\d+)'
fpath = 'train/videos/1873.mp4'
p = subprocess.Popen(["ffmpeg", "-i", fpath, "-vf", "cropdetect", "-vframes", "2", "-f", "rawvideo", "-y", "/dev/null"]
                    , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
infos = p.stderr.read()

allCrops = re.findall(CROP_DETECT_LINE , infos) #y1,y2,x1,x2
#frame = frame[x1:x2,y1:y2,:]
print(allCrops) #=> 290,560,0,461
print(infos)


vr = VideoReader('train/videos/1873.mp4', ctx=cpu(0))
# a file like object works as well, for in-memory decoding
for i in range(len(vr)):
    # the video reader will handle seeking and skipping in the most efficient manner
    frame = vr[i].asnumpy()
    frame = frame[0:422,290:560,:]
    plt.figure()
    plt.imshow(frame) 
    plt.show() 
    print(frame.shape)
print('video frames:', len(vr))


