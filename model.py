import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Siamese(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1):
        super(Siamese, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(73728, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward_one(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.fc2(dis)
        #  return self.sigmoid(out)
        return out


# class Siamese(nn.Module):

#     def __init__(self):
#         super(Siamese, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 64, 10),  # 64@96*96
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 64@48*48
#             nn.Conv2d(64, 128, 7),
#             nn.ReLU(),    # 128@42*42
#             nn.MaxPool2d(2),   # 128@21*21
#             nn.Conv2d(128, 128, 4),
#             nn.ReLU(), # 128@18*18
#             nn.MaxPool2d(2), # 128@9*9
#             nn.Conv2d(128, 256, 4),
#             nn.ReLU(),   # 256@6*6
#         )
#         self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
#         self.out = nn.Linear(4096, 1)

#     def forward_one(self, x):
#         x = self.conv(x)
#         x = x.view(x.size()[0], -1)
#         x = self.liner(x)
#         return x

#     def forward(self, x1, x2):
#         out1 = self.forward_one(x1)
#         out2 = self.forward_one(x2)
#         dis = torch.abs(out1 - out2)
#         out = self.out(dis)
#         #  return self.sigmoid(out)
#         return out


# for test
if __name__ == '__main__':
    # resnet18 = models.resnet18(pretrained=True)
    # pretrained_dict =resnet18.state_dict() 
    # model_dict = model.state_dict() 

    '''
    # net = Siamese()
    # print(net)
    # print(list(net.parameters()))
    '''
    #1、 using Resnet18 to fine tuning models
    # net = torchvision.models.resnet18(pretrained=True)#加载已经训练好的模型
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 1)#将全连接层做出改变类别改为一类
    # print(net)
    # print(list(net.parameters()))

    #2、 fix convolution to train fully connected layer
    # net = torchvision.models.resnet18(pretrained=True)

    # for param in net.parameters():
    #     param.requires_grad = False

    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 1)
    # print(net)
    # print(list(net.parameters()))

    net = Siamese(ResidualBlock)
    print(net)