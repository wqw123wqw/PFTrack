cd src
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py tracking --modal RGB-T --save_all  --exp_id VTMOT_fusion_mid  --dataset mot_rgbt --dataset_version mot_rgbt --load_model "/home/zhuyabin/MOT/CenterTrack/models/mot17_fulltrain.pth" --batch_size 10 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 1 >> "../CenterTrack_fusion_mid04.18.log" 2>&1 
cd ..