'''
用于显示mot17数据集于图片上，便于观察,代码路径文件可参考以下格式：
mot17数据格式：
MOT17:
    train:
        MOT17-02-FRCNN:
            det:
                det.txt
            gt:
                gt.txt
            MOT17-02-FRCNN:
                000001.jpg
                000002.jpg
                000003.jpg
            seqinfo.ini
        MOT17-02-DPM:
            与上面格式类似
    test:
        MOT17-01-DPM:
            det:
                det.txt
            img1:
                000001.jpg
                000002.jpg
                000003.jpg
            seqinfo.ini
        ......

'''


from utils import read_txt,chinese2img,show_img,build_dir
import os
import numpy as np
import cv2
from tqdm import tqdm

def draw_mot(img, info_lst,label_size=20,label_color=(0, 255, 20)):

    for i, f in enumerate(info_lst):

        track_id=int(f[1])
        x1, y1, w,h = f[2:6]
        x2,y2=x1+w,y1+h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 画大矩形
        if len(f)==9:#表示gt
            act = str(int(f[6]))
            cls_id = str(int(f[7]))
            vision_ratio = str(np.round(float(f[8]), 3))
            label = 'track_id:{}\ncls_id:{}\nact:{}\nratio:{}'.format(str(track_id), cls_id, act, vision_ratio)
        else: # 表示det 因为det有时候没有后三个-1，为10或7
            conf = str(np.round(float(f[6]), 3))
            label = 'track_id:{}\nconf:{}'.format(str(track_id), conf)

        img = chinese2img(img, label, coord=(int(x1 + 2), int(y1 + 6)),label_size=label_size,label_color=label_color)

    return img


def mot_vision_demo():

    '''
    mot17数据集的det.txt与gt.txt信息画图显示
    root:需给到这样路径下-->D:\tracker\data\MOT17\train\MOT17-02-DPM


    '''
    root='/data1/Datasets/Tracking/MOT/MOT_RGBT_all/images/test/'
    video_name = 'LasHeR-004'

    results_dir_path = ''
    path_img=os.path.join(root,video_name,'visible')

    out_root = '/home/zhuyabin/MOT/FairMOT/output'


    out_dir=build_dir(os.path.join(out_root,video_name))
    # out_dir_det=build_dir(os.path.join(root, 'out_dir_draw', 'draw_det_imgs'))

    # path_det_txt=os.path.join(root,'det\det.txt')
    path_result_txt = os.path.join(results_dir_path, video_name+'.txt')
    

    # det_info=read_txt(path_det_txt)
    result_info=read_txt(path_result_txt)
    # det_val_lst=np.array([[float(v) for v in det.split(',')] for det in det_info ])
    gt_val_lst = np.array([[float(v) for v in gt.split(',')] for gt in result_info])

    for img_name in tqdm(os.listdir(path_img)):
        name_float=float(img_name[:-4])
        # det_lst=det_val_lst[det_val_lst[:,0]==name_float]
        gt_lst=gt_val_lst[gt_val_lst[:,0]==name_float]
        img=cv2.imread(os.path.join(path_img,img_name))
        img_gt=draw_mot(img.copy(),gt_lst)
        # img_det=draw_mot(img.copy(),det_lst)
        cv2.imwrite(os.path.join(out_dir_gt,img_name),img_gt)
        # cv2.imwrite(os.path.join(out_dir_det, img_name), img_det)


        # show_img(img_gt)
        # show_img(img_det)





if __name__ == '__main__':
    mot_vision_demo()










