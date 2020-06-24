'''
        Outlier Exposure (OE)

    Code for the models from https://arxiv.org/abs/1812.04606
    almost the same as for the other ResNet Models but since they normalize their CIFAR10
    at dataloading time I include this preprocessing step explicitly in the architecture
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_of_channels=3, dataset='CIFAR10'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_of_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if dataset in ['CIFAR10', 'CIFAR100']:
            self.mean = torch.tensor( [x / 255 for x in [125.3, 123.0, 113.9]])
            self.std = torch.tensor( [x / 255 for x in [63.0, 62.1, 66.7]] )
        elif dataset in ['SVHN']:
            self.mean = torch.tensor( [x for x in [0., 0., 0.]] )
            self.std = torch.tensor( [x for x in [1., 1., 1.]] )
        else:
            self.mean = torch.tensor( [x for x in [ 0.]] )
            self.std = torch.tensor( [x for x in [ 1.]] )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out

    def features(self, x):
        out = (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def gpu(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return super(ResNet, self).cuda()


def ResNet18(num_of_channels=3, num_classes=10, dataset='CIFAR10'):
    return ResNet(BasicBlock, [2,2,2,2],
                  num_of_channels=num_of_channels,
                  num_classes=num_classes, dataset=dataset)


class MNIST_ConvNet(nn.Module):

    def __init__(self):
        super(MNIST_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.features(x)
        # x = F.dropout(x)
        x = self.fc2(x)
        return x

    def features(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        return x
