#!/bin/bash
devices=(0 1 2 3 4 5)
root_dir=~/SWIM-ASAD/logs
save_dir=logs0
path=all_subject_leave_story_0
run=3
config=all_subject_leave_story
seed=(42 43 44 45 46)
version=0

python -m src.main \
    --batch_size 64 \
    --num_workers 8 \
    --max_epochs 1 \
    --min_epochs 0 \
    --patience 0 \
    --seed 42 \
    --lr 0.001 \
    --weight_decay 0.001 \
    --pin_memory \
    --es_monitor val/accuracy \
    --model_name cnn \
    --kernel_size 5 \
    --batch_norm \
    --out_channels 16 \
    --EEG_channels 64 \
    --patch_size 128 \
    --train_overlapping_ratio 0.75 \
    --val_overlapping_ratio 0.875 \
    --data_aug_funcs mask_time \
    --mask_time_ratio 1 \
    --subject_loss_weight 0.05 \
    --dataset_split_config $config \
    --split_file split_subject \
    --leave_story story1 \
    --val_ratio 0.15 \
    --speaker_num 3 \
    --subject_num 16 \
    --dataset_path ~/KUL \
    --raw_path ~/KUL/download \
    --preprocessed_path ~/KUL/normalize_std_channel \
    --log_dir $path \
    --devices 0 \
    --version $version \
    --root_dir $root_dir \
    --save_dir $save_dir \
    --fast_dev_run \

# In ~/KUL/download, there are S1.mat - S16.mat, please download them from the official KUL dataset website.