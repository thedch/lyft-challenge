import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from fastai.conv_learner import *
from fastai.transforms import *
import torch
import cv2
import pdb
from lyft_helpers import *

file = sys.argv[-1]

# Thanks to jaycode and phmagic!
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

answer_key = {}

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
frame_num = 1
    
video = skvideo.io.vread(file)        
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
    answer_key[frame_num] = [encode(binary_car_result), encode(binary_road_result)]        
    frame_num += 1

# Print output in proper json format
print (json.dumps(answer_key))
