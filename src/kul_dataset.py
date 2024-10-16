import os
import json
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
from mat4py import loadmat
from torch.utils.data import ConcatDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class SequenceDataset(data.Dataset):

    def __init__(self, eegdata, label, speaker, subject, patch_size, overlapping_ratio,  sub_mean=False, divide_std=False):
        self.eegdata = eegdata
        self.label = label
        self.speaker = speaker
        self.subject = subject
        self.eeglen = len(self.eegdata)
        self.eeg_patch_size = patch_size
        self.eegstep = np.floor(patch_size * (1 - overlapping_ratio)).astype(int)
        self.sub_mean = sub_mean
        self.divide_std = divide_std

    def __len__(self):
        return (self.eeglen - self.eeg_patch_size) // self.eegstep + 1

    def __getitem__(self, idx):
        start = idx * self.eegstep
        end = start + self.eeg_patch_size
        # patch_size * 64
        normalized_data = self.eegdata[start:end]
        if self.sub_mean:
            normalized_data -= torch.mean(normalized_data, dim=0)
        if self.divide_std:
            var = torch.square(torch.std(normalized_data, dim=0)) + 1e-8
            normalized_data /= torch.sqrt(var)
        return normalized_data, self.label, self.subject


class KULDataset(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_path = kwargs['dataset_path']
        self.raw_path = kwargs['raw_path']
        self.preprocessed_path = kwargs['preprocessed_path']
        self.patch_size = kwargs['patch_size']
        self.train_overlapping_ratio = kwargs['train_overlapping_ratio']
        self.val_overlapping_ratio = kwargs['val_overlapping_ratio']
        self.test_overlapping_ratio = kwargs['test_overlapping_ratio']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']
        self.dataset_split_config = kwargs['dataset_split_config']
        self.split_file = kwargs['split_file']
        self.leave_subject = kwargs['leave_subject']
        self.subject = kwargs['subject']
        self.leave_story = kwargs['leave_story']
        self.leave_story_completely = kwargs['leave_story_completely']
        self.val_ratio = kwargs['val_ratio']
        self.test_ratio = kwargs['test_ratio']
        self.std_by_window = kwargs['std_by_window']
        self.speaker_num = kwargs['speaker_num']
        self.subject_num = kwargs['subject_num']
        self.excluded_channels = kwargs['excluded_channels']
        self.used_channels = kwargs['used_channels']
        self.trial_train = kwargs['trial_train']
        self.save_val = kwargs['save_val']

    def prepare_data(self) -> None:
        def _split_dataset():
            split_subject = []
            split_hrtf = []
            split_dry = []
            part_story_dict = {
                "part1_track1": "story1_part1",
                "part1_track2": "story2_part1",
                "part2_track1": "story1_part2",
                "part2_track2": "story2_part2",
                "part3_track1": "story3_part1",
                "part3_track2": "story4_part1",
                "part4_track1": "story3_part2",
                "part4_track2": "story4_part2",
            }
            for i in range(1, 17):
                subject_info = []
                hrtf_info = []
                dry_info = []
                raw_file = loadmat(f"{self.raw_path}/S{i}.mat")['trials']
                # pre subject has 1-20 segments
                # 9-20 are repetitions of 1-8, so don't use them as dataset
                for j in range(1, 9):
                    segment = raw_file[j-1]
                    attended_ear = segment['attended_ear']  # L or R
                    stimuli = []
                    for _stimuli in segment['stimuli']:
                        name = _stimuli[0]
                        names = name.rsplit('_', 1)
                        names[0] = part_story_dict[names[0]]
                        is_hrtf = names[1].startswith('hrtf')
                        stimuli.append('_'.join(names))
                    segment_info = {'file_name': f"S{i}/{j}.npy", 'attended_ear': attended_ear,
                                    'label': 0 if attended_ear == 'L' else 1, 'stimuli': stimuli}
                    subject_info.append(segment_info)
                    if is_hrtf:
                        hrtf_info.append(segment_info)
                    else:
                        dry_info.append(segment_info)
                split_subject.append(subject_info)
                split_hrtf.append(hrtf_info)
                split_dry.append(dry_info)
            with open(f"{self.dataset_path}/split_subject.json", 'w') as f:
                json.dump(split_subject, f, indent=4)
            with open(f"{self.dataset_path}/split_hrtf.json", 'w') as f:
                json.dump(split_hrtf, f, indent=4)
            with open(f"{self.dataset_path}/split_dry.json", 'w') as f:
                json.dump(split_dry, f, indent=4)

        def _process_data(data_path, do_normalize=False):
            os.makedirs(f"{data_path}")
            for i in range(1, 17):
                if not os.path.exists(f"{data_path}/S{i}"):
                    os.makedirs(f"{data_path}/S{i}")
                raw_file = loadmat(f"{self.raw_path}/S{i}.mat")['trials']
                for j in range(1, 9):
                    EEG_data = np.array(raw_file[j-1]['RawData']['EegData'])
                    if do_normalize:
                        EEG_data /= np.std(EEG_data, axis=0)
                    np.save(f"{data_path}/S{i}/{j}.npy",
                            EEG_data.astype(np.float32))

        if not os.path.exists(f"{self.dataset_path}/split_subject.json") or not os.path.exists(f"{self.dataset_path}/split_hrtf.json") or not os.path.exists(f"{self.dataset_path}/split_dry.json"):
            _split_dataset()
        if not os.path.exists(f"{self.preprocessed_path}"):
            _process_data(self.preprocessed_path, do_normalize=True)
        if not os.path.exists(f"{self.raw_path}_npy"):
            _process_data(f"{self.raw_path}_npy", do_normalize=False)

    def setup(self, stage: str) -> None:

        def _get_id(segment):
            speaker = int(segment['stimuli']
                          [segment['label']].split('_')[0][-1])
            if speaker == 4:
                speaker = 3
            speaker = F.one_hot(torch.tensor(speaker-1),
                                num_classes=self.speaker_num).float()
            subject = int(segment['file_name'].split('/')[0][1:])
            subject = F.one_hot(torch.tensor(subject-1),
                                num_classes=self.subject_num).float()
            return speaker, subject

        def _get_eeg_data(segment):
            eegdata = np.load(f"{self.preprocessed_path}/{segment['file_name']}")
            eegdata = torch.from_numpy(eegdata)

            if self.used_channels:
                excluded_channels = list(
                    set(range(0, 64)).difference(set(self.used_channels)))
                eegdata[:, excluded_channels] = 0
            elif self.excluded_channels:
                eegdata[:, self.excluded_channels] = 0
            return eegdata

        def _get_test_data(segment, overlapping_ratio):
            eegdata = _get_eeg_data(segment)
            speaker, subject = _get_id(segment)
            dataset = SequenceDataset(
                eegdata, segment['label'], speaker, subject, self.patch_size,
                overlapping_ratio, sub_mean=True, divide_std=self.std_by_window)
            return dataset

        def _split_data(data_list, split: int):
            train_data_list = []
            val_data_list = []
            test_data_list = []
            if split == 3:
                for data in data_list:
                    split1 = int(len(data) * (1 - self.val_ratio - self.test_ratio))
                    split2 = int(len(data) * (1 - self.test_ratio))
                    if self.trial_train:
                        train_split1 = int(len(data) * self.trial_train[0])
                        train_split2 = int(len(data) * self.trial_train[1])
                        train_data = data[train_split1:train_split2]
                    else:
                        train_data = data[:split1]
                    val_data = data[split1:split2]
                    test_data = data[split2:]

                    train_data_list.append(train_data)
                    val_data_list.append(val_data)
                    test_data_list.append(test_data)

                return train_data_list, val_data_list, test_data_list

            if split == 2:
                for data in data_list:
                    split = int(len(data) * (1 - self.val_ratio))
                    train_data = data[:split]
                    val_data = data[split:]
                    train_data_list.append(train_data)
                    val_data_list.append(val_data)

                return train_data_list, val_data_list

        def _split_per_trial(segment):
            eegdata = _get_eeg_data(segment)

            train_data_list, val_data_list, test_data_list = _split_data([eegdata], split=3)

            speaker, subject = _get_id(segment)
            train_dataset = SequenceDataset(
                train_data_list[0], segment['label'], speaker, subject, self.patch_size,
                self.train_overlapping_ratio, sub_mean=True, divide_std=self.std_by_window)
            val_dataset = SequenceDataset(
                val_data_list[0], segment['label'], speaker, subject, self.patch_size,
                self.val_overlapping_ratio, sub_mean=True, divide_std=self.std_by_window)
            test_dataset = SequenceDataset(
                test_data_list[0], segment['label'], speaker, subject, self.patch_size,
                self.test_overlapping_ratio, sub_mean=True, divide_std=self.std_by_window)

            return train_dataset, val_dataset, test_dataset

        def _split_train_val(segment):
            eegdata = _get_eeg_data(segment)

            train_data_list, val_data_list = _split_data([eegdata], split=2)

            speaker, subject = _get_id(segment)

            train_dataset = SequenceDataset(
                train_data_list[0], segment['label'], speaker, subject, self.patch_size,
                self.train_overlapping_ratio, sub_mean=True, divide_std=self.std_by_window)
            val_dataset = SequenceDataset(
                val_data_list[0], segment['label'], speaker, subject, self.patch_size,
                self.val_overlapping_ratio, sub_mean=True, divide_std=self.std_by_window)

            return train_dataset, val_dataset

        def _is_leaved(segment):
            if not self.leave_story_completely:
                is_leaved = self.leave_story in segment['stimuli'][segment['label']]
            else:
                is_leaved = self.leave_story in segment['stimuli'][
                    0] or self.leave_story in segment['stimuli'][1]
            return is_leaved

        with open(f"{self.dataset_path}/{self.split_file}.json", 'r') as f:
            info = json.load(f)
        train_datasets = []
        val_datasets = []
        test_datasets = []

        if self.dataset_split_config == "leave_subject":
            for i in range(len(info)):
                for segment in info[i]:
                    if i+1 in self.leave_subject:
                        dataset = _get_test_data(segment, self.test_overlapping_ratio)
                        test_datasets.append(dataset)
                    else:
                        train_dataset, val_dataset = _split_train_val(segment)
                        train_datasets.append(train_dataset)
                        val_datasets.append(val_dataset)
            self.test_datasets = ConcatDataset(test_datasets)

        elif self.dataset_split_config == "all_subject_leave_story":
            for i in range(len(info)):
                test_datasets.append([])
                for segment in info[i]:
                    if _is_leaved(segment):
                        dataset = _get_test_data(segment, self.test_overlapping_ratio)
                        if not self.save_val:
                            test_datasets[i].append(dataset)
                    else:
                        train_dataset, val_dataset = _split_train_val(segment)
                        train_datasets.append(train_dataset)
                        val_datasets.append(val_dataset)
                        if self.save_val:
                            test_datasets[i].append(val_dataset)
            self.test_datasets = [ConcatDataset(
                test_datasets[i]) for i in range(len(test_datasets))]

        elif self.dataset_split_config == "all_subject_per_trial":
            for i in range(len(info)):
                test_datasets.append([])
                for segment in info[i]:
                    train_dataset, val_dataset, test_dataset = _split_per_trial(
                        segment)
                    train_datasets.append(train_dataset)
                    val_datasets.append(val_dataset)
                    test_datasets[i].append(test_dataset)
            self.test_datasets = [ConcatDataset(
                test_datasets[i]) for i in range(len(test_datasets))]

        self.train_datasets = ConcatDataset(train_datasets)
        self.val_datasets = ConcatDataset(val_datasets)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return data.DataLoader(self.train_datasets, batch_size=self.batch_size, shuffle=True,
                               num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.val_datasets, batch_size=self.batch_size, shuffle=False,
                               num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self.test_datasets, list):
            return [data.DataLoader(i, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, pin_memory=self.pin_memory) for i in self.test_datasets]
        else:
            return data.DataLoader(self.test_datasets, batch_size=self.batch_size, shuffle=False,
                                   num_workers=self.num_workers, pin_memory=self.pin_memory)
