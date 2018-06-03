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

sz = 448
transforms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)

road_learn = torch.load('road-fullmodel-2.pt')
car_learn = torch.load('car-fullmodel-2kds-2.pt')

ROAD_THRESH = 3
CAR_THRESH = -2

frame_num = 1 # Frame numbering starts at 1

video = cv2.VideoCapture(file)
while video.isOpened():    
    ret, frame = video.read()
    if frame == None:
        break # end of video
        
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      
    im = im.astype(np.float32)/255
    
    aug_t, aug_v = transforms[1](im, im)
    im_with_batch = V(aug_t).unsqueeze_(0)
    
    # Road section    
    pred = road_learn(im_with_batch)
    pred_np = to_np(pred)
    pred_big = cv2.resize(pred_np[0], (800, 600)) 
    binary_road_result = np.where(pred_big>ROAD_THRESH,1,0).astype('uint8')
    
    # Car section        
    pred = car_learn(im_with_batch)
    pred_np = to_np(pred)
    pred_big = cv2.resize(pred_np[0], (800, 600)) 
    binary_car_result = np.where(pred_big>CAR_THRESH,1,0).astype('uint8')

    # Save answers
    answer_key[frame_num] = [encode(binary_car_result), encode(binary_road_result)]        
    frame_num += 1

# Print output in proper json format
print (json.dumps(answer_key))
