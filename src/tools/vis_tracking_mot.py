import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

GT_PATH = "/data1/Datasets/Tracking/MOT/VTMOT/images/test/"
# GT_PATH = "/home/zhuyabin/MOT/CenterTrack/output1/pre/"
IMG_PATH = GT_PATH
SAVE_VIDEO = True
RESIZE = 1
IS_GT = True

def draw_bbox(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
      c, 2, lineType=cv2.LINE_AA)
    # ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    # txt = '{}'.format(bbox[4])
    # cv2.putText(img, txt, (int(ct[0]), int(ct[1])), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
    #             (255,0,0), thickness=1, lineType=cv2.LINE_AA)
    
def draw_bbox1(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
      c, 2, lineType=cv2.LINE_AA)
    # ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    # txt = '{}'.format(bbox[4])
    # cv2.putText(img, txt, (int(bbox[0]+1), int(bbox[1]+1)), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
    #             (0,0,255), thickness=1, lineType=cv2.LINE_AA)

if __name__ == '__main__':
  result = {
    # 'fairmot_rgbt': "/data1/Datasets/Tracking/MOT/VTMOT/images/results/FairMOT/VTMOT_RGBT_9/",
    'ours': "/home/zhuyabin/MOT/CenterTrack/exp/tracking/VTMOT_fusion_early_V3_m/results_mot_rgbt_7",
    # # 'transtrack_rgbt': "/data1/Datasets/Tracking/MOT/VTMOT/images/results/TransTrack/RGBT/RGB-T_result_28/",
    # 'bytetrack_rgbt': "/home/zhuyabin/wangtao/ByteTrack-main/YOLOX_outputs/yolox_m_rgbt/track_results_VT4",
    # 'trades':"/home/zhuyabin/MOT/TraDeS/exp/tracking/VTMOT/results_motmot_rgbt_test_6",
    # 'oc_sort_rgbt': "/home/zhuyabin/MOT/OC_SORT/VT/VT_6",
    # "centertrack":"/home/zhuyabin/MOT/CenterTrack/exp/tracking/VTMOT_RGBT/results_mot_rgbt_7"
  }

  color = {
    'fairmot_rgbt':(0,255,255),
    'ours':(0,0,255),
    'transtrack_rgbt':(0,255,255),
    'trades':(255,128,0),
    'bytetrack_rgbt':(255,0,0),
    'oc_sort_rgbt':(0,255,255),
    'centertrack':(153,0,204)
    



  }

  flag = 0
  # image_to_anns = defaultdict(list)
  search_seq = 'photo-0318-46'


  for tracker, result_root in result.items():
    print(tracker)
    print(result_root)

    pre_root = result_root
    # "/data1/Datasets/Tracking/MOT/MOT_RGBT_all/images/results/MOT_visible/"
    seqs = os.listdir(GT_PATH)
    # if SAVE_VIDEO:
    #   save_path = sys.argv[1][:sys.argv[1].rfind('/res')] + '/video'
    #   if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #   print('save_video_path', save_path)
    for seq in sorted(seqs):
      if search_seq in seq:  #RGBT234-49   Vtuav-13   photo-0310-33  photo-0310-46 photo-0318-40 photo-0318-42  qiuxing-0306-28

        print('seq', seq)
      #   # if len(sys.argv) > 2 and not sys.argv[2] in seq:
      #   #   continue
      #   if '.DS_Store' in seq:
      #     continue
      #   # if SAVE_VIDEO:
      #   #   fourcc = cv2.VideoWriter_fourcc(*'XVID')
      #   #   video = cv2.VideoWriter(
      #   #     '{}/{}.avi'.format(save_path, seq),fourcc, 10.0, (1024, 750))
        seq_path = '{}/{}/visible/'.format(GT_PATH, seq)
        if IS_GT:
          ann_path = seq_path.replace('visible/','') + 'gt/gt.txt'
        # else:
        #   ann_path = seq_path + 'det/det.txt'
          image_to_anns = defaultdict(list)
          anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
          print('anns shape', anns.shape)
          if flag == 0:
            print('load gt anns ................................................')
            for i in range(anns.shape[0]):
              if anns[i][6] == 0:
                continue
              if (not IS_GT) or (int(anns[i][6]) == 1 and float(anns[i][8]) >= 0.25):
                frame_id = int(anns[i][0])
                track_id = int(anns[i][1])
                bbox = (anns[i][2:6] / RESIZE).tolist()
                image_to_anns[frame_id].append(bbox + [track_id])
                flag = 0

          images = sorted(os.listdir(seq_path))
          num_imgs = len([image for image in images if 'jpg' in image])
        
        # image_to_preds = {}
        for K in range(1, num_imgs+1):
          K = str(K).zfill(6)
          #image_to_preds[K] = defaultdict(list)
          image_to_preds = defaultdict(list)
          pred_path = pre_root + '/{}.txt'.format(seq)
          try:
            preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=',')
          except:
            preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=' ')
          for i in range(preds.shape[0]):
            frame_id = int(preds[i][0])
            track_id = int(preds[i][1])
            bbox = (preds[i][2:6] / RESIZE).tolist()
            # image_to_preds[K][frame_id].append(bbox + [track_id])
            image_to_preds[frame_id].append(bbox + [track_id])
        
        img_path = seq_path
        images = sorted(os.listdir(img_path))
        num_images = len([image for image in images if 'jpg' in image])
        
        for i ,image in enumerate(images):
          frame_id = i + 1
          v_file_name = '{}/visible/{}'.format(seq, image)
          r_file_name = '{}/infrared/{}'.format(seq, image)
          v_file_path = IMG_PATH + v_file_name
          r_file_path = IMG_PATH + r_file_name
          v_img = cv2.imread(v_file_path)
          r_img = cv2.imread(r_file_path)
          print(v_file_path)
          print(r_file_path)
          if RESIZE != 1:
            v_img = cv2.resize(v_img, (v_img.shape[1] // RESIZE, v_img.shape[0] // RESIZE))
            r_img = cv2.resize(r_img, (r_img.shape[1] // RESIZE, r_img.shape[0] // RESIZE))
          v_img_pred = v_img.copy()
          r_img_pred = r_img.copy()
          # for K in range(1, num_images+1):
          #   K = str(K).zfill(6)
            
          # draw_bbox(img_pred, image_to_preds[K][frame_id])
          draw_bbox(v_img_pred, image_to_anns[frame_id],(0,255,0))
          # draw_bbox1(v_img_pred, image_to_preds[frame_id], color[tracker])
          draw_bbox(r_img_pred, image_to_anns[frame_id],(0,255,0))
          # draw_bbox1(r_img_pred, image_to_preds[frame_id], color[tracker])
          if not os.path.exists(f'/home/zhuyabin/MOT/CenterTrack/output1/gt/{seq}/visible'):
            os.makedirs(f'/home/zhuyabin/MOT/CenterTrack/output1/gt/{seq}/visible')
          if not os.path.exists(f'/home/zhuyabin/MOT/CenterTrack/output1/gt/{seq}/infrared'):
            os.makedirs(f'/home/zhuyabin/MOT/CenterTrack/output1/gt/{seq}/infrared')
          
          frame_id = str(frame_id).zfill(6)
          cv2.imwrite(f'/home/zhuyabin/MOT/CenterTrack/output1/gt/{seq}/visible/{frame_id}.jpg', v_img_pred)
          cv2.imwrite(f'/home/zhuyabin/MOT/CenterTrack/output1/gt/{seq}/infrared/{frame_id}.jpg', r_img_pred)
          # cv2.imshow('gt', img)
          # cv2.waitKey()
          # if SAVE_VIDEO:
          #   video.write(img_pred)
        # if SAVE_VIDEO:
        #   video.release()
