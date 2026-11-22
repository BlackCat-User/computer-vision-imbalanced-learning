import torch, torch.nn as nn, torch.nn.functional as F

class PadShortcut(nn.Module):
    def __init__(self, pad):
        super().__init__(); self.pad = pad
    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = PadShortcut(out_ch // 4)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNet32(nn.Module):
    def __init__(self, num_classes=10, n=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make(16, 16, n, 1)
        self.layer2 = self._make(16, 32, n, 2)
        self.layer3 = self._make(32, 64, n, 2)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
    def _make(self, i, o, n, s):
        return nn.Sequential(BasicBlock(i, o, s), *[BasicBlock(o, o, 1) for _ in range(n - 1)])
    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.layer3(self.layer2(self.layer1(x))).mean(dim=(2, 3))
    def forward(self, x):
        return self.fc(self.features(x))
