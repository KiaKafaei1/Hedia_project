import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
from IPython.display import clear_output
from skimage.io import imread
from skimage.transform import resize
from google.colab import drive
import sys

# run GPU .... 
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device= torch.device("cpu")
    print(device)

size = (7, 7)

class Detection_Network(nn.Module):
  def __init__(self):
    super(Detection_Network, self).__init__()

    self.adaptive_max_pool = nn.AdaptiveMaxPool2d(size[0], size[1])
    self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)]).to(device)
    self.cls_loc = nn.Linear(4096, 11 * 4).to(device) # (10 classes + 1 background. Each will have 4 co-ordinates)
    self.score = nn.Linear(4096, 11).to(device) # (10 classes, + 1 background)
    
    self.cls_loc.weight.data.normal_(0, 0.01)
    self.cls_loc.bias.data.zero_()
    
    self.score.weight.data.normal_(0, 0.01)
    self.score.bias.data.zero_()
    

  def forward(self,sample_roi, out_map):
    # 7x7x512 = 25088 this is the input size and 4096 is the output size which could be an image of 64x64.
    # we first have a layer that reduces the input and the second layer doesn't reduce the input.
    # The third layer reduces the image to an ouput list of size 8. Consiting of 2 classes (forground and background) and 4 coordinates which
    # is the coordinate and a height and width.

    # Get RoIs
    rois = torch.from_numpy(sample_roi).float()
    roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
    roi_indices = torch.from_numpy(roi_indices).float()
    indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    indices_and_rois = xy_indices_and_rois.contiguous()

    size = (7, 7)
    # get RoI maxpooling
    output = []
    rois = indices_and_rois.data.float()
    rois[:, 1:].mul_(1/16.0) # Subsampling ratio
    rois = rois.long()
    num_rois = rois.size(0)
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        tmp = self.adaptive_max_pool(im)
        output.append(tmp[0])
    output = torch.cat(output, 0)
    outputs = output.clone().detach()
    # Reshape the tensor so that we can pass it through the feed forward layer.
    # This is the output after pooling
    out_maxPool = outputs.view(outputs.size(0), -1)

    # run the detection network
    k = self.roi_head_classifier(out_maxPool.to(device))
    # It classifies the location and the score of the location
    roi_cls_loc = self.cls_loc(k)
    roi_cls_score = self.score(k)
    return roi_cls_loc, roi_cls_score, rois
