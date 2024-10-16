import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerConv(nn.Module):
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

        d_model = 64
        nhead = 8
        dim_feedforward = 128
        num_encoder_layers = 3

        input_dim = 64
        output_dim = 2

        # self.mamba_batchnorm = nn.BatchNorm1d(input_dim)
        self.trans_input = nn.Linear(input_dim, d_model)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                activation='relu',
                batch_first=True,
            ),
            num_encoder_layers,
            norm=nn.LayerNorm(d_model),
            # layernorm is normalize on feature dim, in this exp it's on 64 EEG channels
            # norm_first=False so the norm is caculated at last in every layer
        )
        self.trans_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, d_model)),
            nn.Flatten()
        )
        self.trans_head = nn.Linear(d_model, output_dim)


    def forward(self, x):
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
            hid = self.fc(y)
            cnn_states.append(hid)
        y = torch.stack(cnn_states, dim=1)
        mamba_y = y
        mamba_y = self.trans_input(mamba_y)
        mamba_y = self.trans(mamba_y)
        mamba_y = self.trans_pool(mamba_y)
        prediction = self.trans_head(mamba_y)
        return prediction


if __name__ == "__main__":
    model = TransformerConv().to('cuda')
    x = torch.randn(1, 128, 64).to('cuda')
    print(model(x).shape)