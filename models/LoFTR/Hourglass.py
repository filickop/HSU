import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block for Hourglass network"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """Single Hourglass module"""

    def __init__(self, block, num_blocks, in_channels, depth=4):
        super().__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hourglass(block, num_blocks, in_channels, depth)

    def _make_hourglass(self, block, num_blocks, in_channels, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(block(in_channels, in_channels))
            if i == 0:
                res.append(block(in_channels, in_channels))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)

        if n > 1:
            low1 = F.max_pool2d(x, 2, stride=2)
            low1 = self.hg[n - 1][1](low1)
            low2 = self._hourglass_forward(n - 1, low1)
            low3 = self.hg[n - 1][2](low2)
            up2 = F.interpolate(low3, scale_factor=2)
        else:
            up2 = self.hg[n - 1][3](x)

        return up1 + up2

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)


class HeatmapNet(nn.Module):
    """Stacked Hourglass Network for Heatmap Prediction"""

    def __init__(self, in_channels=3, num_keypoints=4, num_stacks=2):
        super().__init__()
        self.num_stacks = num_stacks

        # Initial processing
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 256)

        # Stacked hourglasses
        hgs, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hgs.append(Hourglass(ResidualBlock, num_blocks=4, in_channels=256))
            res.append(self._make_residual(256, 256))
            fc.append(self._make_fc(256, 256))
            score.append(nn.Conv2d(256, num_keypoints, kernel_size=1))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(256, 256, kernel_size=1))
                score_.append(nn.Conv2d(num_keypoints, 256, kernel_size=1))

        self.hgs = nn.ModuleList(hgs)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)

    def _make_fc(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Initial processing
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)

        outputs = []
        for i in range(self.num_stacks):
            y = self.hgs[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            outputs.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return outputs  # List of heatmaps from each stack