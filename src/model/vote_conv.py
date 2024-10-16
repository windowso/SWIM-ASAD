import torch
import torch.nn as nn
import torch.nn.functional as F


class VoteConv(nn.Module):
    def __init__(self, cnn_patch_size, cnn_step):
        super().__init__()
        out_channels = 16
        kernel_size = (5, 64)
        batch_norm = True
        dropout_p = None
        subject_num = 16

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

        self.cnn_patch_size = cnn_patch_size
        self.cnn_step = cnn_step
        self.pad = False

        self.mode = 'vote' # mean, vote or oracle

        for param in self.conv.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x, label=None):
        # x: batch_size * patch_size * 64
        patch_size = x.size(1)
        x = x.unsqueeze(1)
        cnn_states = []
        if self.pad:
            patch_size += self.cnn_patch_size
            x = F.pad(x, (0, 0, self.cnn_patch_size // 2, self.cnn_patch_size // 2))
        for i in range(0, patch_size-self.cnn_patch_size+1, self.cnn_step):
            y = self.conv(x[:, :, i:i+self.cnn_patch_size])
            y = self.pool_mean(y)
            y = self.fc(y)
            y = self.linear(y)[:, :2]
            cnn_states.append(y)
        y = torch.stack(cnn_states, dim=1)
        if self.mode == 'mean':
            y = y.mean(dim=1)
        elif self.mode == 'vote':
            y = y.argmax(dim=-1)
            y = F.one_hot(y, num_classes=2).to(torch.float32).mean(dim=1)
        elif self.mode == 'oracle':
            returned = torch.full((y.size(0),), -1, dtype=torch.long, device=y.device)
            y = y.argmax(dim=-1)
            # return the first correct answer in y
            for batch in range(y.size(0)):
                returned[batch] = y[batch, 0]
                for i in range(y.size(1)):
                    if y[batch, i] == label[batch]:
                        returned[batch] = y[batch, i]
                        break
            y = F.one_hot(returned, num_classes=2).to(torch.float32)
        return y


if __name__ == "__main__":
    model = VoteConv().to('cuda')
    x = torch.randn(1, 128, 64).to('cuda')
    print(model(x).shape)