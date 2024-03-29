import torch
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

def resnet_picker(arch, dataset):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""
    in_channels = 1 if dataset == 'MNIST' else 3
    num_classes = 10
    if dataset in ['CIFAR10', 'MNIST']:
        num_classes = 10
        initial_conv = [3, 1, 1]
    elif dataset == 'CIFAR100':
        num_classes = 100
        initial_conv = [3, 1, 1]
    elif dataset == 'TinyImageNet':
        num_classes = 200
        initial_conv = [7, 2, 3]
    else:
        raise ValueError(f'Unknown dataset {dataset} for ResNet.')

    if arch == 'ResNet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif 'ResNet20-' in arch and arch[-1].isdigit():
        width_factor = int(arch[-1])
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * width_factor, initial_conv=initial_conv)
    elif arch == 'ResNet28-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet32':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet32-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet44':
        return ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet56':
        return ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet110':
        return ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet18':
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif 'ResNet18-' in arch:  # this breaks the usual notation, but is nicer for now!!
        new_width = int(arch.split('-')[1])
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=new_width, initial_conv=initial_conv)
    elif arch == 'ResNet34':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet50':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet101':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet152':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    else:
        raise ValueError(f'Invalid ResNet [{dataset}] model chosen: {arch}.')


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR-like thingies.

    This is a minor modification of
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
    adding additional options.
    """

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
                 norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
        self._norm_layer = norm_layer

        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=initial_conv[0],
                                     stride=initial_conv[1], padding=initial_conv[2], bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x, penu=False):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if penu:
            return x
        x = self.linear(x)

        return x
