cd src
CUDA_VISIBLE_DEVICES=5  python  test_rgbt.py tracking --modal RGB-T --test_mot_rgbt True --exp_id VTMOT_fusion_M_V3_all --dataset mot_rgbt --dataset_version mot_rgbt --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model /home/zhuyabin/MOT/CenterTrack/exp/tracking/VTMOT_fusion_early_M_V3_all/model_9.pth
cd ..
