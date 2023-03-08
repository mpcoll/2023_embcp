import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_l3(nn.Module):

    def __init__(self, block, layers, num_classes, linear_size=4096):
        self.inplanes = 64
        self.num_classes = num_classes
        super(ResNet_l3, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(2)
        # self.fc = nn.Linear(9216 * block.expansion, 512)
        # self.fc = nn.Linear(2048, 512) # Resnet 34
        self.fc = nn.Linear(linear_size, 512) # Resnet 50
        self.fc2 = nn.Linear(512, num_classes)

        # self.fc = nn.Linear(512*block.expansion, num_classes)
        # self.lr = nn.LeakyReLU()
        # self.fc2 = nn.Linear(200, 100)
        # self.fc3 = nn.Linear(100, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.num_classes == 1:
            x1 = self.fc2(x).squeeze()
        else:
            x1 = self.fc2(x)
        # print(x.size())
        # x1 = self.fc(x).squeeze()
        # x1 = self.lr(x1)
        # x1 = self.fc2(x1)
        # x1 = self.lr(x1)
        # x = self.fc3(x1)

        return x1

def resnet50(num_classes, linear_size):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_l3(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                      linear_size=linear_size)
    return model

def resnet34(num_classes, linear_size):
    model = ResNet_l3(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                      linear_size=linear_size)
    return model


def resnet152(num_classes, linear_size):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_l3(Bottleneck, [3, 8, 36, 3], num_classes=num_classes,
                      linear_size=linear_size)

    return model

class AlexNet3D_Dropout(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet3D_Dropout, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU( inplace=True),

            nn.Conv3d(192, 128, kernel_size=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=1),
        )

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(3072, 256),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(256, num_classes),
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        # x = np.swapaxes(x, 1, 2)
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        if self.num_classes == 1: # If regression return squeezed array
            x = self.classifier(x).squeeze()
        else:
            x = self.classifier(x)
        return x




class AlexNet3D_Deeper_Dropout(nn.Module):

    def __init__(self, num_classes = 2, regression=False):
        super(AlexNet3D_Deeper_Dropout, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU( inplace=True),

            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU( inplace=True),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512, 64),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        if self.num_classes == 1: # If regression return squeezed array
            x = self.classifier(x).squeeze()
        else:
            x = self.classifier(x)

        return x

class AlexNet3D_Dropout_Regression(nn.Module):

    def __init__(self, num_classes=1):
        super(AlexNet3D_Dropout_Regression, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU( inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )

        self.regressor = nn.Sequential(nn.Dropout(),
                                        #nn.Linear(1536, 64), ADNI
                                        nn.Linear(256, 64),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes)
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        xp = self.features(x)

        x = xp.view(xp.size(0), -1)
        # print(x.size())

        if self.num_classes == 1: # If regression return squeezed array
            x = self.regressor(x).squeeze()
        else:
            x = self.regressor(x)
        return [x, xp]



class AlexNet2D(nn.Module):

    def __init__(self, num_classes = 2) -> None:
        super(AlexNet2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.size)
        x = self.classifier(x)
        return [x, x]

class MLP2l_fc(nn.Module):

    def __init__(self, num_classes=2, hidden1=512, hidden2=512):
        super(MLP2l_fc, self).__init__()
        self.classifier = nn.Sequential(
           nn.Dropout(0.2),
           nn.Linear(1485, hidden1),
           nn.ReLU(inplace=True),
           nn.Dropout(0.2),
           nn.Linear(hidden1, hidden2),
           nn.ReLU(inplace=True),
           nn.Dropout(0.2),
           nn.Linear(hidden2, num_classes)
           )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return [x, x]




class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, mat_size, bias=False):
        super(E2EBlock, self).__init__()
        self.d = mat_size
        self.cnn1 = torch.nn.Conv2d(in_planes,planes,(1,self.d),bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes,planes,(self.d,1),bias=bias)


    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d,3)+torch.cat([b]*self.d,2)

class BrainNetCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BrainNetCNN, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 1 # Onlyu one channel for now
        self.d = 279 # Hardcoded the brainnetome size, change for different size

        self.e2econv1 = E2EBlock(1,32, self.d, bias=True)
        self.e2econv2 = E2EBlock(32,64, self.d, bias=True)
        self.E2N = torch.nn.Conv2d(64,1,(1,self.d))
        self.N2G = torch.nn.Conv2d(1,256,(self.d,1))
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,30)
        self.dense3 = torch.nn.Linear(30, num_classes)

    def forward(self, x):
        x = x.float()
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),negative_slope=0.33) 
        out = F.leaky_relu(self.E2N(out),negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out),negative_slope=0.33),p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out),negative_slope=0.33),p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out),negative_slope=0.33),p=0.5)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.33)

        if self.num_classes == 1: # If regression return squeezed array
            out = out.squeeze()

        return [out, out]

class BrainNetCNN_deeper(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BrainNetCNN_deeper, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 1 # Onlyu one channel for now
        self.d = 279 # Hardcoded the brainnetome size, change for different size

        self.e2econv1 = E2EBlock(1,32, self.d, bias=True)
        self.e2econv2 = E2EBlock(32,64, self.d, bias=True)
        self.e2econv3 = E2EBlock(64,128, self.d, bias=True)
        self.E2N = torch.nn.Conv2d(128,1,(1,self.d))
        self.N2G = torch.nn.Conv2d(1,256,(self.d,1))
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,30)
        self.dense3 = torch.nn.Linear(30, num_classes)

    def forward(self, x):
        x = x.float()
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv3(out),negative_slope=0.33) 
        out = F.leaky_relu(self.E2N(out),negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out),negative_slope=0.33),p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out),negative_slope=0.33),p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out),negative_slope=0.33),p=0.5)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.33)

        if self.num_classes == 1: # If regression return squeezed array
            out = out.squeeze()

        return [out, out]
# # Test 4d
# import sys
# sys.path.append('/code')
# import torch
# from convNd import convNd
# from convNd import MaxPool4d

# # define basic layer info
# inChans = 1
# outChans = 64
# weight = torch.rand(1)[0]
# bias = torch.rand(1)[0]

# # create input tensor
# x = torch.rand(1, inChans, 5, 5, 5, 5).cuda()
# conv4d = convNd(
#     in_channels=inChans,
#     out_channels=outChans,
#     num_dims=4,
#     kernel_size=3,
#     stride=(2,1,1,1),
#     padding=0,
#     padding_mode='zeros',
#     output_padding=0,
#     is_transposed=False,
#     use_bias=True,
#     groups=1,
#     kernel_initializer=lambda x: torch.nn.init.constant_(x, weight),
#     bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).cuda()

# a = conv4d(x)

# test = MaxPool4d(kernel_size=3, stride=2)



# test(a)

# class AlexNet3D_Dropout(nn.Module):

#     def __init__(self, num_classes = 2):
#         super(AlexNet3D_Dropout, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
#             nn.BatchNorm3d(64),
#             nn.ReLU( inplace=True),
#             nn.MaxPool3d(kernel_size=2, stride=2),

#             nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm3d(128),
#             nn.ReLU( inplace=True),
#             nn.MaxPool3d(kernel_size=3, stride=3),

#             nn.Conv3d(128, 192, kernel_size=3, padding=1),
#             nn.BatchNorm3d(192),
#             nn.ReLU(inplace=True),

#             nn.Conv3d(192, 192, kernel_size=3, padding=1),
#             nn.BatchNorm3d(192),
#             nn.ReLU( inplace=True),

#             nn.Conv3d(192, 128, kernel_size=3, padding=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU( inplace=True),
#             nn.MaxPool3d(kernel_size=3, stride=3),
#         )

#         self.classifier = nn.Sequential(nn.Dropout(),
#                                         nn.Linear(128, 64),
#                                         nn.ReLU( inplace=True),
#                                         nn.Dropout(),
#                                         nn.Linear(64, num_classes),
#                                        )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self,x):
#         x = self.features(x)
#         # print(x.size())
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return [x, x]



###################################################################
# Densenet ()
###################################################################

from collections import OrderedDict


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def generate_densenet(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)

    return model