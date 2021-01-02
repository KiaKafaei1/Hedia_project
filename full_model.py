# -*- coding: utf-8 -*-
"""full_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17dGF-3pWd20tJGqa7lorEYZ0n5NRp-j4
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

# Self defined models and functions
from prepare_img import prepare_img
from anchorbox_generator import Anchorbox_generator
from class_rpn_vgg16 import ClassRPN, vgg16
from detection import Detection_Network
from nms_detection_loss import nms, sampleRoi
from bbox_predict import bbox_predict

# run GPU .... 
if (torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device= torch.device("cpu")
    print(device)

class full_model(nn.Module):
  def __init__(self, VGG16_t=True):
    super(full_model, self).__init__()
    if VGG16_t:
      self.vgg16 = vgg16()
    self.ClassRPN = ClassRPN()
    self.Detection_Network = Detection_Network()
    

  def forward(self, img0, tt , out_map=[], anc_n_sample = 256, n_sample = 128, anc_pos_ratio = 0.5, pos_ratio = 0.25, pos_iou_thresh = 0.5, anc_pos_iou_thresh = 0.7, anc_neg_iou_thresh = 0.3, neg_iou_thresh_hi = 0.5, neg_iou_thresh_lo = 0.0, nms_thresh = 0.7, n_train_pre_nms = 12000, n_train_post_nms = 2000, min_size = 16):
    out = {}
    # prepare image and labels for use
    labels, bbox, img = prepare_img(img0, tt)
    # get feature map with vgg16 network
    if isinstance(out_map, list):
      out_map = self.vgg16(img)
    # run the region proposal network
    pred_anchor_locs, objectness_score, pred_cls_scores = self.ClassRPN(out_map)
    out['RPN'] = pred_anchor_locs, objectness_score, pred_cls_scores
    # get anchors
    anchor_labels, anchor_locations, anchors = Anchorbox_generator(bbox, labels, anc_pos_iou_thresh, anc_neg_iou_thresh, anc_pos_ratio, anc_n_sample)
    out['anchor'] = anchor_labels, anchor_locations, anchors
    # Get the regions of interest
    roi = nms(anchors, pred_anchor_locs, objectness_score, nms_thresh, n_train_pre_nms, n_train_post_nms, min_size)
    out['nms'] = roi
    # Get sample rois for training
    pos_index, neg_index, gt_roi_locs, gt_roi_labels, sample_roi, pos_roi_per_this_image = sampleRoi(roi, bbox, labels, n_sample, pos_ratio, pos_iou_thresh, neg_iou_thresh_hi, neg_iou_thresh_lo)
    out['sampleROI'] = pos_index, neg_index, gt_roi_locs, gt_roi_labels, sample_roi, pos_roi_per_this_image
    # run the detection network
    roi_cls_loc, roi_cls_score, rois = self.Detection_Network(sample_roi, out_map)
    out['roi'] = roi_cls_loc, roi_cls_score, rois

    return out


  def predict(self, img0, out_map=[], nms_thresh=0.7, n_test_pre_nms=100, n_test_post_nms=30, min_size=16, remove_background=True):
    # prepare image for use
    img = prepare_img(img0)
    # get feature map with vgg16 network
    if isinstance(out_map, list):
      out_map = self.vgg16(img)
    # run the region proposal network
    pred_anchor_locs, objectness_score, pred_cls_scores = self.ClassRPN(out_map)
    # get anchors
    anchors = Anchorbox_generator()
    # Get the regions of interest
    roi = nms(anchors, pred_anchor_locs, objectness_score, n_train_pre_nms = 6000, n_train_post_nms = 300)
    # run the detection network
    roi_cls_loc, roi_cls_score, rois = self.Detection_Network(roi, out_map)
    # get predicted bbox, predicted labels and score
    roi_loc_pred, score, labels = bbox_predict(roi, roi_cls_loc, roi_cls_score, nms_thresh, n_test_pre_nms, n_test_post_nms, min_size, remove_background)

    return img, roi_loc_pred, score, labels, roi, anchors
