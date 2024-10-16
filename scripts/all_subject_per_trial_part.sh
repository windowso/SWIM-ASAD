#!/bin/bash

trial=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
start=0
end=1

for ((i=0; i<=6; i++)); do
    devices=(0 1 2 3 4 5 6 7)
    path=all_subject_per_trial_$i
    run=3
    config=all_subject_per_trial
    seed=(42 43 44 45 46)
    root_dir=~/SWIM-ASAD/logs
    save_dir=logs1
    version=0
    for ((j=0; j<$run; j++)); do
        python -m src.main \
            --batch_size 64 \
            --num_workers 8 \
            --max_epochs 100 \
            --min_epochs 50 \
            --patience 20 \
            --seed ${seed[$j]} \
            --lr 0.001 \
            --weight_decay 0.001 \
            --pin_memory \
            --es_monitor val/accuracy \
            --model_name cnn \
            --out_channels 16 \
            --kernel_size 5 \
            --batch_norm \
            --EEG_channels 64 \
            --patch_size 128 \
            --train_overlapping_ratio 0.75 \
            --data_aug_funcs mask_time \
            --mask_time_ratio 1 \
            --subject_loss_weight 0.05 \
            --val_overlapping_ratio 0.875 \
            --dataset_split_config $config \
            --split_file split_subject \
            --val_ratio 0.15 \
            --test_ratio 0.15 \
            --trial_train ${trial[$start]} ${trial[$end]} \
            --speaker_num 3 \
            --subject_num 16 \
            --dataset_path ~/KUL \
            --raw_path ~/KUL/download \
            --preprocessed_path ~/KUL/normalize_std_channel \
            --log_dir $path \
            --devices ${devices[$((j%${#devices[@]}))]} \
            --version $j \
            --root_dir $root_dir \
            --save_dir $save_dir \
        &
        version=$((version+1))
        if [ $((version%${#devices[@]})) -eq 0 ]
        then
            wait
        fi
    done

    wait
    python src/utils.py --root_dir $root_dir --save_dir $save_dir --path $path --config $config --run $run
    start=$((start+1))
    end=$((end+1))
    wait

done