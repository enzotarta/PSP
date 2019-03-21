'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        #self.shortcut = nn.Sequential()
        self.shortcuta = nn.Sequential()
        self.shortcutb = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcuta = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcutb = nn.BatchNorm2d(self.expansion*planes)
            #self.shortcut = nn.Sequential(
            #nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            #nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        z = []
        if isinstance(x, tuple):
            x , z = x
        out = self.conv1(x)
        if self.training: z.append(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if self.training: z.append(out)
        out = self.bn2(out)
 
        s = self.shortcuta(x)
        #print(isinstance(self.shortcuta, nn.Conv2d))
        if self.training and isinstance(self.shortcuta, nn.Conv2d): z.append(s)
        out += self.shortcutb(s)

        out = F.relu(out)
        return out , z


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        z = []
        if isinstance(x, tuple):
            x , z = x
        out = self.conv1(x)
        if self.training: z.append(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = selv.conv2(out)
        if self.training: z.append(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        if self.training: z.append(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out , z



class ResNetKRMB(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetKRMB, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        z = []
        out = self.conv1(x)
        if self.training: z.append(out)
        out = self.bn1(out)
        out = F.relu(out)

        out,zl = self.layer1(out)
        if self.training: z += zl
        out,zl = self.layer2(out)
        if self.training: z += zl
        out,zl = self.layer3(out)
        if self.training: z += zl
        out,zl = self.layer4(out)
        if self.training: z += zl

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training: z.append(out)
        return out , z


def ResNet18():
    return ResNetKRMB(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNetKRMB(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNetKRMB(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNetKRMB(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNetKRMB(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
