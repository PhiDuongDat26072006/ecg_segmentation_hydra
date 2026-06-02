# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout1d(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv1 = ConvBnRelu1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = ConvBnRelu1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)


class StackDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=8, stride=2, padding=3)
        self.conv1 = ConvBnRelu1d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = ConvBnRelu1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2] != x.shape[2]:
            x = F.pad(x, pad=(0, 1))  # pad last dimension of x by (0,1)
        x = torch.cat(tensors=(x, skip), dim=1)  # concatenate along channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ECGUNet(nn.Module):
    def __init__(self, down1, down2, down3, down4, up1, up2, up3, up4, middle, classify):
        super().__init__()

        self.down1 = down1
        self.down2 = down2
        self.down3 = down3
        self.down4 = down4

        self.up4 = up4
        self.up3 = up3
        self.up2 = up2
        self.up1 = up1

        self.middle = middle
        self.classify = classify

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.middle(x)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.classify(x)
