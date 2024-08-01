from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
import sys
from ..generic_dataset_rgbt import GenericDataset

class MOT_RGBT(GenericDataset):
  num_categories = 2
  default_resolution = [544, 960]
  class_name = ['']
  max_objs = 256
  cat_ids = {1: 1, 2: 2, -1: -1}
  def __init__(self, opt, split):


    self.dataset_version = opt.dataset_version
    
    print('Using MOT_RGBT')
    data_dir = os.path.join("/data1/Datasets/Tracking/MOT/VTMOT/",'images')
    #--dataset_version mot(rgbt)
    if opt.dataset_version in ['mot_rgbt']:
      ann_file = '{}.json'.format('train' if split == 'train' else \
        'test')
    img_dir = os.path.join(data_dir, '{}'.format(
      'test' if 'test' in ann_file else 'train'))
    
    print('ann_file', ann_file)
    ann_path = os.path.join(data_dir, 'annotations', ann_file)

    self.images = None
    # load image list and coco
    super(MOT_RGBT, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded MOT {} {} {} samples'.format(
      self.dataset_version, split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_{}'.format(self.dataset_version))
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      tracks = defaultdict(list)
      for image_info in images:
        if not (image_info['id'] in results):
          continue
        result = results[image_info['id']]
        frame_id = image_info['frame_id']
        for item in result:
          if not ('tracking_id' in item):
            item['tracking_id'] = np.random.randint(100000)
          if item['active'] == 0:
            continue
          tracking_id = item['tracking_id']
          bbox = item['bbox']
          bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
          tracks[tracking_id].append([frame_id] + bbox)
      rename_track_id = 0
      for track_id in sorted(tracks):
        rename_track_id += 1
        for t in tracks[track_id]:
          f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
            t[0], rename_track_id, t[1], t[2], t[3]-t[1], t[4]-t[2]))
      f.close()
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    gt_type_str = '{}'.format(
                'mot_rgbt' if 'mot_rgbt' in self.opt.dataset_version \
                else '')
    # gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
    gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
      ''
    os.system('python tools/eval_motchallenge.py ' + \
              '../data/mot{}/{}/ '.format(self.year, 'train') + \
              '{}/results_mot{}/ '.format(save_dir, self.dataset_version) + \
              gt_type_str + ' --eval_official')
