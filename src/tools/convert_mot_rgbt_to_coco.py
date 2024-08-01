"""
https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun
"""
import os
import numpy as np
import json
import cv2
from natsort import ns, natsorted



DATA_PATH = "/data1/Datasets/Tracking/MOT/VTMOT/images/"
OUT_PATH = os.path.join(DATA_PATH, 'annotations2')
SPLITS = ['train','test']

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:

        data_path = os.path.join(DATA_PATH, split)                   #"/home/oil/Wangqianwu/MOTDataset/MOT(RGBT)/images/train"
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'},{'id': 2, 'name': 'car'}]}#{'id': 1, 'name': 'pedestrian'},{'id': 2, 'name': 'car'}
        seqs = os.listdir(data_path)                                 #[baby、bike、bikeman、biketwo、blueCar]
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq or '.ipy' in seq:
                continue

            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            vi_img_path = os.path.join(seq_path, 'visible')
            ir_img_path = os.path.join(seq_path, 'infrared')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            #natsorted(os.listdir(vi_dir_path), alg=ns.PATH)
            vi_images = natsorted(os.listdir(vi_img_path),alg=ns.PATH)
            ir_images = natsorted(os.listdir(ir_img_path),alg=ns.PATH)

            #判断两个文件夹下面的图片数量是否相等
            if len(vi_images) != len(ir_images):
                print("error！")
            num_images = len([image for image in vi_images if 'jpg' in image])  # half and half

            #for i in range(num_images):
            for i in range(len(vi_images)):
                
                vi_img = cv2.imread(os.path.join(vi_img_path, vi_images[i]))
                height, width = vi_img.shape[:2]
                image_info = {'vi_file_name': '{}/visible/{}'.format(seq,vi_images[i]),  # image name.
                              'ir_file_name': '{}/infrared/{}'.format(seq,ir_images[i]),
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height,
                              'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))

            if split != 'test':
                
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                for i in range(anns.shape[0]):
                    if anns[i][6] ==0:
                        continue
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    #category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': cat_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                    out['annotations'].append(ann)
                print('{}: {} ann images'.format(seq, int(anns[:, 0].max())))

            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
        
