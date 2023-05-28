import torch.nn as nn


class WMSA(nn.Module):
    def __init__(self, channels=64, reduction_ratio=4):
        super(WMSA, self).__init__()
        inter_channels = int(channels // reduction_ratio)

        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        fused_feature = x + residual
        local_feature = self.local_conv(fused_feature)
        global_feature = self.global_conv(fused_feature)
        combined_feature = local_feature + global_feature
        weights = self.sigmoid(combined_feature)

        output = 2 * x * weights + 2 * residual * (1 - weights)
        return output
