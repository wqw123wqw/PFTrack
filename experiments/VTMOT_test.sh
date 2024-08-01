cd src
CUDA_VISIBLE_DEVICES=3 python test_rgbt.py tracking --modal RGB-T --test_mot_rgbt True --exp_id VTMOT_fusion_mid --dataset mot_rgbt --dataset_version mot_rgbt --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model /home/zhuyabin/MOT/CenterTrack/exp/tracking/VTMOT_fusion_mid/model_5.pth
cd ..