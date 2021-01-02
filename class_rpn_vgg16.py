# -*- coding: utf-8 -*-
"""Class_RPN_vgg16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uJBYdBT_esJKE2_aWIfszJwrL0qwOc47
"""

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

# run GPU .... 
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device= torch.device("cpu")
    print(device)

#RPN
in_channels = 512 # depends on the output feature map. in vgg 16 it is equal to 512
mid_channels = 512
n_anchor = 9  # Number of anchors at each location

class vgg16(nn.Module):
  def __init__(self):
    super(vgg16, self).__init__()
    # List all the layers of VGG16
    # input can be smaller than 800 according to torchvision
    model = torchvision.models.vgg16(pretrained=True).to(device)
    fe = list(model.features)

    # collect layers with output feature map size (W, H) < 50
    dummy_img = torch.zeros((1, 3, 800, 800)).float() # test image array [1, 3, 800, 800] 

    #go through each layer and save output of layer into req_features
    req_features = []
    k = dummy_img.clone().to(device)
    for i in fe:
        k = i(k)
        if k.size()[2] < 800//16:   #800/16=50
            break
        req_features.append(i)
        out_channels = k.size()[1]

    # Convert this list into a Sequential module
    self.faster_rcnn_fe_extractor = nn.Sequential(*req_features)

  def forward(self, img):
    #feature extractor
    transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    imgTensor = transform(img).to(device) 
    imgTensor = imgTensor.unsqueeze(0)
    #use vgg16 network on our image
    out_map = self.faster_rcnn_fe_extractor(imgTensor)
    return out_map



class ClassRPN(nn.Module):
  def __init__(self):
    super(ClassRPN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(device)
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv1.bias.data.zero_()


    #the reason for n_anchor*4 is that it correspond to the position offset of each bounding box relative to the preset anchor box (1 region needs to predict 4 values of the prediction area Tx, Ty, Tw, Th) .
    #https://medium.com/@nabil.madali/demystifying-region-proposal-network-rpn-faa5a8fb8fce
    self.reg_layer = nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0).to(device)
    self.reg_layer.weight.data.normal_(0, 0.01)
    self.reg_layer.bias.data.zero_()


    #the reason for n_anchor*2 is that it correspond to the foreground and background probabilities of k regions at each pixel in Feature Map (1 region and 2 scores)
    self.cls_layer = nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0).to(device) ## I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
    self.cls_layer.weight.data.normal_(0, 0.01)
    self.cls_layer.bias.data.zero_()


  def forward(self, out_map):
    #apply definened layers on image
    x = self.conv1(out_map.to(device)) # out_map = faster_rcnn_fe_extractor(imgTensor)
    pred_anchor_locs = self.reg_layer(x)
    pred_cls_scores = self.cls_layer(x)
    
    # Rearrange arrays
    pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
    pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
    objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
    pred_cls_scores  = pred_cls_scores.view(1, -1, 2)
    return pred_anchor_locs, objectness_score, pred_cls_scores

