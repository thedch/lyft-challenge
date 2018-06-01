import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

from fastai.conv_learner import *
from fastai.transforms import *
import torch
import cv2

class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)
        
    def forward(self, x): return self.bn(F.relu(self.conv(x)))

def grab_last_channel(x): return x[:,0]
def not_lambda(x): return x[:,0] # TODO: Rename this please

flatten_channel = Lambda(grab_last_channel)

simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel
)

aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05)]
small_sz = 128
big_sz = 512
big_transforms = tfms_from_model(resnet34, big_sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
small_transforms = tfms_from_model(resnet34, small_sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
# train_tfms, val_tfms = transforms

road_learn = torch.load('road-fullmodel.pt')
car_learn = torch.load('car-fullmodel-3.pt')

# Frame numbering starts at 1
frame = 1

for im in video:
    im = im.astype(np.float32)/255
    
    # Road section (128)
    aug_t, aug_v = small_transforms[1](im, im)
    im_with_batch = V(aug_t).unsqueeze_(0)
    pred = road_learn(im_with_batch)
    pred_np = to_np(pred)
    pred_big = cv2.resize(pred_np[0], (800, 600)) 
    binary_road_result = np.where(pred_big>3,1,0).astype('uint8')
    
    # Car section (512)
    aug_t, aug_v = big_transforms[1](im, im)
    im_with_batch = V(aug_t).unsqueeze_(0)
    pred = car_learn(im_with_batch)
    pred_np = to_np(pred)
    pred_big = cv2.resize(pred_np[0], (800, 600)) 
    binary_car_result = np.where(pred_big>-2,1,0).astype('uint8')

    # Save answers
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]        
    frame += 1

# Print output in proper json format
print (json.dumps(answer_key))
