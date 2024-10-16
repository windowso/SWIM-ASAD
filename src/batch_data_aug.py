import random


class BatchDataAugment():
    def __init__(self, hparams):
        self.hparams = hparams

    def __call__(self, data):
        for func in self.hparams.data_aug_funcs:
            data = getattr(self, func)(data)
        return data
    
    def mask_time(self, data):
        # mask different time for different batch_item in one batch
        mask_time_len_max = int(data.shape[1] * self.hparams.mask_time_ratio)
        for i in range(data.shape[0]):
            mask_time_len = random.randint(0, mask_time_len_max)
            mask_time_start = random.randint(0, data.shape[1] - mask_time_len)
            data[i, mask_time_start:mask_time_start + mask_time_len, :] = 0
        return data
