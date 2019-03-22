'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        z = []
        if isinstance(x, tuple):
            x , z = x

        if self.training: 
            y = x.clone().detach_()
            yo = self.bn1(y)
            z.append(yo)
        out = self.bn1(x)

        out = F.relu(out)

        if self.training: 
            y = out.clone().detach_()
            yo = self.conv1(y)
            z.append(yo)
        out = self.conv1(out)

        if self.training: 
            y = out.clone().detach_()
            yo = self.bn2(y)
            z.append(yo)
        out = self.bn2(out)

        out = F.relu(out)

        if self.training: 
            y = out.clone().detach_()
            yo = self.conv2(y)
            z.append(yo)
        out = self.conv2(out)

        out = torch.cat([out,x], 1) 

        return out , z


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        z = []
        if isinstance(x, tuple):
            x , z = x

        if self.training: 
            y = x.clone().detach_()
            yo = self.bn(y)
            z.append(yo)
        out = self.bn(x)

        out = F.relu(out)

        if self.training: 
            y = out.clone().detach_()
            yo = self.conv(y)
            z.append(yo)
        out = self.conv(out)

        out = F.avg_pool2d(out, 2)
        return out , z


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        z = []
        if isinstance(x, tuple):
            x , z = x    

        if self.training: 
            y = x.clone().detach_()
            yo = self.conv1(y)
            z.append(yo)
        out = self.conv1(x)

        out , zl = self.trans1(self.dense1(out))
        if self.training: z += zl
        out , zl = self.trans2(self.dense2(out))
        if self.training:  z += zl
        out , zl = self.trans3(self.dense3(out))
        if self.training:   z += zl
        out , zl = self.dense4(out)
        if self.training:  z += zl

        if self.training: 
            y = out.clone().detach_()
            yo = self.bn(y)
            z.append(yo)
        out = self.bn(out) 

        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)

        if self.training: 
            y = out.clone().detach_()
            yo = self.linear(y)
            z.append(yo)
        out = self.linear(out)

        return out , z

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
