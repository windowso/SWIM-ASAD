import json
# import sys
# sys.path.append('/home/zhangzy/SWIM-ASAD/src')

import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from ..batch_data_aug import BatchDataAugment
from .conv_model import ConvNet
from .conv_baseline import ConvNetBaseline
from .mamba_conv import MambaConv
from .transformer_conv import TransformerConv
from .vote_conv import VoteConv


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

        self.loss = nn.CrossEntropyLoss()

        if self.hparams.dataset_split_config == 'all_subject_leave_story' or self.hparams.dataset_split_config == 'all_subject_per_trial':
            self.predictions = [[] for _ in range(self.hparams.subject_num)]
            self.labels = [[] for _ in range(self.hparams.subject_num)]
        elif self.hparams.dataset_split_config == 'leave_subject':
            self.predictions = [[]]
            self.labels = [[]]

    def forward(self, x, label=None):
        if self.hparams.model_name == 'vote':
            return self.model(x, label)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label, subject = batch
        if self.hparams.data_aug_funcs:
            batch_data_aug = BatchDataAugment(self.hparams)
            data = batch_data_aug(data)
        loss, accuracy, _ = self._forward(data, label, subject)
        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/accuracy', accuracy, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        # pay attention that if you use multiple gpus,
        # the loss is only calculated on the gpu 0 if sync_dict = False
        return loss

    def validation_step(self, batch, batch_idx):
        data, label, subject = batch
        loss, accuracy, _ = self._forward(data, label, subject)
        self.log(f'val/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val/accuracy', accuracy, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, label, subject = batch
        loss, accuracy, prediction = self._forward(data, label, subject)
        self.log(f'test/accuracy', accuracy, on_step=False,
                on_epoch=True, prog_bar=True, sync_dist=True)
        self.predictions[dataloader_idx] += prediction.tolist()
        self.labels[dataloader_idx] += label.tolist()
        return loss

    def on_test_epoch_end(self) -> None:
        with open(f'{self.logger.log_dir}/test_results.json', 'w') as f:
            json.dump({"predictions": self.predictions,
                      "labels": self.labels}, f, indent=4)

    def _forward(self, data, label, subject):
        prediction = self(data, label)
        loss = self.loss(prediction[:, :2], label)
        if self.hparams.subject_loss_weight:
            subject_loss = self.loss(prediction[:, -self.hparams.subject_num:], subject.argmax(dim=1))
            loss += self.hparams.subject_loss_weight * subject_loss
        prediction_digit = prediction[:, :2].argmax(dim=1, keepdim=True)
        correct = prediction_digit.eq(label.view_as(prediction_digit)).sum().item()
        accuracy = correct / len(label)
        return loss, accuracy, prediction

    def configure_optimizers(self):
        if self.hparams.model_name == 'mamba':
            optimizer = optim.Adam([
                {'params': self.model.conv.parameters(), 'lr': 1e-5},
                {'params': self.model.fc.parameters(), 'lr': 1e-5},
                {'params': self.model.linear.parameters(), 'lr': 1e-5},
                {'params': self.model.mamba_input.parameters()},
                {'params': self.model.mamba.parameters()},
                {'params': self.model.mamba_head.parameters()},
            ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.model_name == 'transformer':
            optimizer = optim.Adam([
                {'params': self.model.conv.parameters(), 'lr': 1e-5},
                {'params': self.model.fc.parameters(), 'lr': 1e-5},
                {'params': self.model.linear.parameters(), 'lr': 1e-5},
                {'params': self.model.trans_input.parameters()},
                {'params': self.model.trans.parameters()},
                {'params': self.model.trans_head.parameters()},
            ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def load_model(self):
        if self.hparams.model_name == 'cnn':
            self.model = ConvNet(
                self.hparams.out_channels,
                (self.hparams.kernel_size, self.hparams.EEG_channels),
                # self.hparams.patch_size
                self.hparams.dropout_p,
                self.hparams.batch_norm,
                self.hparams.subject_num,
            )
        elif self.hparams.model_name == 'mamba':
            self.model = MambaConv(
                self.hparams.cnn_patch_size,
                self.hparams.cnn_step,
            )
        elif self.hparams.model_name == 'transformer':
            self.model = TransformerConv(
                self.hparams.cnn_patch_size,
                self.hparams.cnn_step,
            )
        elif self.hparams.model_name == 'vote':
            self.model = VoteConv(
                self.hparams.cnn_patch_size,
                self.hparams.cnn_step,
            )
        elif self.hparams.model_name == 'cnn_baseline':
            self.model = ConvNetBaseline(
                self.hparams.EEG_channels,
                self.hparams.out_channels,
                self.hparams.kernel_size
            )
