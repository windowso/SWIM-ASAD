import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, out_channels, kernel_size, dropout_p, batch_norm, subject_num):
        super().__init__()
        self.conv = nn.Sequential(
            # input: batch_size * 1 * patch_size * 64
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size[0] // 2, 0)),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
        )
        self.pool_mean = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.ReLU(),
        )
        self.linear = nn.Linear(4 * out_channels, 2 + subject_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.conv(x)
        y = self.pool_mean(y)
        y = self.fc(y)
        y = self.linear(y)
        return y
