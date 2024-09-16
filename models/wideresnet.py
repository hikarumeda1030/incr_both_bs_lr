import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_layer1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.dropout_rate = dropout_rate
        self.has_shortcut = (input_channels != output_channels)
        self.shortcut_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False) if self.has_shortcut else None

    def forward(self, x):
        shortcut = x
        if self.has_shortcut:
            x = self.relu1(self.batch_norm1(x))
        else:
            shortcut = self.relu1(self.batch_norm1(x))
        out = self.conv_layer1(shortcut if self.has_shortcut else x)
        out = self.relu2(self.batch_norm2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv_layer2(out)
        if self.has_shortcut:
            shortcut = self.shortcut_layer(x)
        return torch.add(shortcut, out)


class ResidualLayer(nn.Module):
    def __init__(self, num_blocks, input_channels, output_channels, block, stride, dropout_rate=0.0):
        super(ResidualLayer, self).__init__()

        self.blocks = nn.Sequential(*[
            block(
                input_channels if i == 0 else output_channels,
                output_channels,
                stride if i == 0 else 1,
                dropout_rate
            ) for i in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class WideResNet(nn.Module):
    def __init__(self, depth, width_multiplier, dropout_rate=0.0, dataset_name='CIFAR100'):
        super(WideResNet, self).__init__()
        layer_channels = [16, 16 * width_multiplier, 32 * width_multiplier, 64 * width_multiplier]
        assert ((depth - 4) % 6 == 0)
        num_blocks_per_layer = (depth - 4) // 6
        block = ResidualBlock

        # Set number of classes based on dataset
        num_classes = 200 if dataset_name == 'TinyImageNet' else 100

        self.initial_conv = nn.Conv2d(3, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_pool = None  # Placeholder if you want to add pooling later

        # Residual layers
        self.layer1 = ResidualLayer(num_blocks_per_layer, layer_channels[0], layer_channels[1], block, 1, dropout_rate)
        self.layer2 = ResidualLayer(num_blocks_per_layer, layer_channels[1], layer_channels[2], block, 2, dropout_rate)
        self.layer3 = ResidualLayer(num_blocks_per_layer, layer_channels[2], layer_channels[3], block, 2, dropout_rate)

        # Global average pooling and fully connected classifier
        self.final_batch_norm = nn.BatchNorm2d(layer_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(layer_channels[3], num_classes)
        self.output_channels = layer_channels[3]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(self, x):
        out = self.initial_conv(x)
        if self.initial_pool is not None:
            out = self.initial_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.final_batch_norm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.output_channels)
        return self.classifier(out)


def build_wide_resnet(model_type='WideResNet40_4', dataset_name='CIFAR100'):
    model_map = {
        'WideResNet40_4': (40, 4),
        'WideResNet16_8': (16, 8),
        'WideResNet28_10': (28, 10),
        'WideResNet28_12': (28, 12),
    }
    depth, width_multiplier = model_map[model_type]
    # Create and return the WideResNet model
    return WideResNet(depth=depth, width_multiplier=width_multiplier, dataset_name=dataset_name)
