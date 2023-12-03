"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import torch
import math

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#定义3*3卷积函数。
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True) #inplace = True ,会改变输入数据的值,节
        #省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
        #implace的默认值是False，用False也行
    )

#定义1*1卷积函数
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

#构造类
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2] #断言：

        hidden_dim = round(inp * expand_ratio) #四舍五入
        #在步长为1，并且输入通道和输出通道相等的时候为True，
        #在值为True的时候，使用残差结构。
        self.identity = stride == 1 and inp == oup 

        #根据expand_ratio来判断使用dw+pw还是pw+dw+pw
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x) #定义x+经过卷积的x #是否使用残差结构，
        #根据情况而言。
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [4,  24, 2, 2],
            # [6,  32, 3, 2],
            [4,  64, 4, 2],
            [4,  96, 3, 1],
            # [6, 160, 3, 2],
            [4, 64, 1, 1],
        ]

        # building first layer
        #定义输入的通道数
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)] #初始化网络结构的第一层
        #并将整个网络结构按照列表的形式进行保存
        # building inverted residual blocks
        block = InvertedResidual #block块
        for t, c, n, s in self.cfgs:
            #计算输出的通道数
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            #根据cfg中给定的参数进行网络结构的存储，存储在列表中
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t)) #t值
                #就是expand_ratio
                input_channel = output_channel
        self.features = nn.Sequential(*layers) #将存储好的列表结构进行
        # building last several layers
        # output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        output_channel = 128
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print(output_channel)
        # exit()
        # self.classifier = nn.Linear(output_channel, num_classes)
        self.classifier = nn.Linear(output_channel, 10) #10分类任务

        self._initialize_weights() #?

    def forward(self, x):
        x = self.features(x)
        # print("在forward中输出")
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        # exit()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


if __name__ == "__main__":
    x = torch.randn(4,3,28,28)
    net = MobileNetV2()
    # print(net)
    y = net(x)
    print(y.shape)
    
    # print(y)