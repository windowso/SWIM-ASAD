#!/bin/bash
devices=(0 1 2 3 4 5)
root_dir=/home/zhangzy/SWIM-ASAD/logs
save_dir=logs0
path=all_subject_leave_story_0
run=3
config=all_subject_leave_story
seed=(42 43 44 45 46)
version=0

for ((i=1; i<=2; i++)); do
    for ((j=1; j<=$run; j++)); do
        python -m src.main \
            --batch_size 64 \
            --num_workers 8 \
            --max_epochs 100 \
            --min_epochs 50 \
            --patience 20 \
            --seed ${seed[$j-1]} \
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
            --leave_story story$i \
            --val_ratio 0.15 \
            --speaker_num 3 \
            --subject_num 16 \
            --dataset_path /home/zhangzy/KUL \
            --raw_path /home/zhangzy/KUL/download \
            --preprocessed_path /home/zhangzy/KUL/normalize_std_channel \
            --log_dir $path \
            --devices ${devices[$((version%${#devices[@]}))]} \
            --version $version \
            --root_dir $root_dir \
            --save_dir $save_dir \
        &
        version=$((version+1))
        if [ $((version%${#devices[@]})) -eq 0 ]
        then
            wait
        fi
    done
done

wait
python src/utils.py --root_dir $root_dir --save_dir $save_dir --path $path --config $config --run $run