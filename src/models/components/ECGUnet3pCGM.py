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

class StackDecoder3p(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels[0], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels[1], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(in_channels[2], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv1d(in_channels[3], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv1d(in_channels[4], skip_channels, kernel_size=kernel_size, padding=padding)
        self.aggregate = ConvBnRelu1d(skip_channels * 5, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        x = torch.cat(tensors=(x1, x2, x3, x4, x5), dim=1)  # concatenate along channel dimension
        x = self.aggregate(x)  # feature aggregation
        return x


class ECGUNet3pCGM(nn.Module):
    def __init__(self, down1, down2, down3, down4, middle, classify, up1, up2, up3, up4, segment, mask=True):
        super().__init__()
        self.mask = mask

        # filters = [n_channels * (2 ** n) for n in range(5)]  # n_filters for encoder feature maps
        # filters_skip = filters[0]  # n_filters for skip connections # 32
        # filters_decoder = filters_skip * 5  # n_filters for decoder feature maps # 160

        self.down1 = down1
        self.down2 = down2
        self.down3 = down3
        self.down4 = down4
        self.middle = middle

        self.classify = classify

        self.up4 = up4
        self.up3 = up3
        self.up2 = up2
        self.up1 = up1
        self.segment = segment

    def apply_cls_mask(self, seg, cls):
        cls_mask = (cls == 0).float()  # 0 if label is not 0
        seg_masked = torch.stack((
            torch.einsum('bt,b->bt', seg[:, 0, :], cls_mask),  # P
            seg[:, 1, :],  # QRS
            seg[:, 2, :],  # T
            seg[:, 3, :],  # None
        ), dim=1)  # (B,4,len_wave)

        return seg_masked

    def forward(self, x):
        # encoder
        X_enc1, x = self.down1(x)
        X_enc2, x = self.down2(x)
        X_enc3, x = self.down3(x)
        X_enc4, x = self.down4(x)
        X_enc5 = self.middle(x)

        # classification
        aggregate = torch.cat(tensors=[
            F.avg_pool1d(X_enc1, kernel_size=16),
            F.avg_pool1d(X_enc2, kernel_size=8),
            F.avg_pool1d(X_enc3, kernel_size=4),
            F.avg_pool1d(X_enc4, kernel_size=2),
            X_enc5
        ], dim=1)
        X_cls_prob = self.classify(aggregate)
        X_cls = X_cls_prob.argmax(dim=1)  # (B,)

        # decoder
        X_dec5 = X_enc5
        X_dec4 = self.up4(
            F.max_pool1d(X_enc1, kernel_size=8, stride=8),
            F.max_pool1d(X_enc2, kernel_size=4, stride=4),
            F.max_pool1d(X_enc3, kernel_size=2, stride=2),
            X_enc4,
            F.interpolate(X_dec5, size=X_enc4.shape[-1], mode='linear', align_corners=False)
        )
        X_dec3 = self.up3(
            F.max_pool1d(X_enc1, kernel_size=4, stride=4),
            F.max_pool1d(X_enc2, kernel_size=2, stride=2),
            X_enc3,
            F.interpolate(X_dec4, size=X_enc3.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc3.shape[-1], mode='linear', align_corners=False)
        )
        X_dec2 = self.up2(
            F.max_pool1d(X_enc1, kernel_size=2, stride=2),
            X_enc2,
            F.interpolate(X_dec3, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc2.shape[-1], mode='linear', align_corners=False)
        )
        X_dec1 = self.up1(
            X_enc1,
            F.interpolate(X_dec2, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec3, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc1.shape[-1], mode='linear', align_corners=False)
        )

        X_seg = self.segment(X_dec1)

        if self.mask and not self.training:
            X_seg = self.apply_cls_mask(X_seg, X_cls)

        return X_seg, X_cls_prob
