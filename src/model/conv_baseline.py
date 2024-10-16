import torch.nn as nn


class ConvNetBaseline(nn.Module):
    def __init__(self, EEG_channels, out_channels=5, kernel_len=17):
        super().__init__()
        self.conv = nn.Sequential(
            # input: batch_size * 1 * patch_size * 64
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_len, EEG_channels),
                      stride=1, padding=(kernel_len // 2, 0)),
            nn.ReLU(),
        )
        self.pool_mean = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),
            nn.Linear(out_channels, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.conv(x)
        y = self.pool_mean(y)
        y = self.fc(y)
        return y
