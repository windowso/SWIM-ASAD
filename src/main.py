import json
import torch
import numpy as np
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from .model import MInterface
from .kul_dataset import KULDataset


def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor=args.es_monitor,
        mode='min' if 'loss' in args.es_monitor else 'max',
        patience=args.patience,
        min_delta=0.0001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val/accuracy',
        filename='best-epoch={epoch}-val_acc={val/accuracy:.2f}',
        save_top_k=1,
        mode='max',
        auto_insert_metric_name=False
    ))

    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'))
    
    return callbacks


def save_results(log_dir, results):
    if len(results) > 1:
        acc = [list(i.values())[0] for i in results]
        acc_dict = {f'S{i+1}': acc[i] for i in range(len(acc))}
        acc_dict.update({'mean': np.mean(acc)})
        acc_dict.update({'median': np.median(acc)})
        with open(f'{log_dir}/test_accuracy.json', 'w') as f:
            json.dump(acc_dict, f, indent=4)
    else:
        with open(f'{log_dir}/test_accuracy.json', 'w') as f:
            json.dump(results[0], f, indent=4)


def main(args):
    pl.seed_everything(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = KULDataset(**vars(args))

    model = MInterface(**vars(args))

    logger = TensorBoardLogger(
        save_dir=f'{args.root_dir}/{args.save_dir}', name=args.log_dir, version=args.version)

    callbacks = load_callbacks(args)

    trainer = Trainer(
        accelerator='gpu',
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        fast_dev_run=args.fast_dev_run,
        devices=[args.devices],
        enable_model_summary=True,
        enable_progress_bar=False,
    )
    if args.test_only:
        print("Begin testing")
        map_location = {'cuda:0': f'cuda:{args.devices}',
                        'cuda:1': f'cuda:{args.devices}',
                        'cuda:2': f'cuda:{args.devices}',
                        'cuda:3': f'cuda:{args.devices}',
                        'cuda:4': f'cuda:{args.devices}',
                        'cuda:5': f'cuda:{args.devices}',
                        'cuda:6': f'cuda:{args.devices}',
                        'cuda:7': f'cuda:{args.devices}', }
        model = MInterface.load_from_checkpoint(
            args.ckpt_path, map_location=map_location, **vars(args))
        results = trainer.test(
            model=model, datamodule=dataset)
        save_results(trainer.log_dir, results)
    elif args.finetune:
        print("Begin finetuning")
        map_location = {'cuda:0': f'cuda:{args.devices}',
                        'cuda:1': f'cuda:{args.devices}',
                        'cuda:2': f'cuda:{args.devices}',
                        'cuda:3': f'cuda:{args.devices}',
                        'cuda:4': f'cuda:{args.devices}',
                        'cuda:5': f'cuda:{args.devices}',
                        'cuda:6': f'cuda:{args.devices}',
                        'cuda:7': f'cuda:{args.devices}', }
        model = MInterface.load_from_checkpoint(
            args.ckpt_path, map_location=map_location, strict=False, **vars(args))
        trainer.fit(model, dataset)
        results = trainer.test(ckpt_path='best', datamodule=dataset)
        save_results(trainer.log_dir, results)
    else:
        print("Begin training")
        trainer.fit(model, dataset)
        if not args.fast_dev_run:
            results = trainer.test(ckpt_path='best', datamodule=dataset)
            save_results(trainer.log_dir, results)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Basic Training Control
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--min_epochs', default=50, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--seed', default=46, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--es_monitor', default='val/accuracy', type=str)
    parser.add_argument('--subject_loss_weight', type=float)
    parser.add_argument('--finetune', action='store_true')

    # model config
    parser.add_argument('--model_name', default='cnn', type=str)
    parser.add_argument('--EEG_channels', default=64, type=int)
    # CNN config
    parser.add_argument('--out_channels', type=int)
    parser.add_argument('--kernel_size', type=int)
    parser.add_argument('--dropout_p', type=float)
    parser.add_argument('--batch_norm', action='store_true')

    # dataset config
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--cnn_patch_size', default=128, type=int)
    parser.add_argument('--cnn_step', default=128, type=int)
    parser.add_argument('--train_overlapping_ratio', default=0, type=float)
    parser.add_argument('--val_overlapping_ratio', default=0, type=float)
    parser.add_argument('--test_overlapping_ratio', default=0, type=float)
    parser.add_argument('--dataset_split_config', default='all_subject_leave_story', type=str)
    parser.add_argument('--split_file', default='split_subject',
                        choices=['split_subject', 'split_hrtf', 'split_dry'], type=str)
    parser.add_argument('--leave_subject', type=int, nargs='+')
    parser.add_argument('--subject', type=int)
    parser.add_argument('--leave_story', type=str)
    parser.add_argument('--leave_story_completely', action='store_true')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float)
    parser.add_argument('--trial_train', type=float, nargs='+')
    parser.add_argument('--std_by_window', action='store_true')
    parser.add_argument('--excluded_channels', type=int, nargs='+')
    parser.add_argument('--used_channels', type=int, nargs='+')
    parser.add_argument('--save_val', action='store_true')
    parser.add_argument('--target_resample_rate', type=int)

    # data augmentation config
    parser.add_argument('--data_aug_funcs', type=str, nargs='+')
    parser.add_argument('--mask_time_ratio', type=float)

    # xf and subject id config
    parser.add_argument('--speaker_num', default=3, type=int)
    parser.add_argument('--subject_num', default=16, type=int)

    # Path Info
    parser.add_argument('--dataset_path', default='KUL', type=str)
    parser.add_argument('--raw_path', default='/KUL_v1.0/download', type=str)
    parser.add_argument('--preprocessed_path', default='KUL_v1.0/normalize_std_channel', type=str)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--root_dir', default='logs', type=str)
    parser.add_argument('--save_dir', default='logs', type=str)
    parser.add_argument('--version', type=int)
    parser.add_argument('--ckpt_path', type=str)

    parser.add_argument('--devices', default=0, type=int)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--fast_dev_run', action='store_true')

    args = parser.parse_args()

    main(args)
