# Scribblekitti_sp2_ddp

to run step 1: training:

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config_path config/training.yaml --dataset_config_path config/semantickitti.yaml
