from torch import nn
import torch
from torch import Tensor


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def s_pointwise_two_ex(x: Tensor):
    batch_size, num_channels, height, width = x.size()
    if num_channels % 4 == 0:
        x_a, x_b, x_c, x_d = x.chunk(4, dim=1)
        return x_a, x_b, x_c, x_d
    else:
        return x


def s_pointwise_two_reex(x: Tensor):
    batch_size, num_channels, height, width = x.size()
    if num_channels % 4 == 0:
        x_a, x_b, x_c, x_d = x.chunk(4, dim=1)
        x = torch.cat((x_a, x_b, x_b, x_c, x_c, x_d, x_d, x_a), dim=1)
        return x
    else:
        return x


def s_pointwise_eight_groups(x: Tensor):
    x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 = x.chunk(8, dim=1)
    x = torch.cat((x_1, x_2, x_2, x_3, x_3, x_4, x_4, x_5, x_5, x_6, x_6, x_7, x_7, x_8, x_8, x_1), dim=1)
    return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.pointsign = False
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = []

        layers.append(
            nn.Sequential(nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=1, groups=4, bias=False),
                          nn.BatchNorm2d(hidden_channel)))

        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        if self.use_shortcut:
            return x + self.conv(s_pointwise_two_reex(x))
        else:
            return self.conv(s_pointwise_two_reex(x))


class MobileNet_s(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, expansion=6):
        super(MobileNet_s, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # print(self.features[1].conv5[0].weight)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_s_small(num_class=1000):
    return MobileNet_s(num_classes=num_class, expansion=2)


def mobilenet_s_medium(num_class=1000):
    return MobileNet_s(num_classes=num_class, expansion=4)
