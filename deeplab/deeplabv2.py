import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101

"""
DeepLabV2 implementation with ResNet-101 backbone.

- The ResNet-101 backbone is modified to use atrous (dilated) convolutions in conv4_x and conv5_x
  to preserve spatial resolution while increasing the receptive field.
- ASPP (Atrous Spatial Pyramid Pooling) applies parallel convolutions with different dilation rates
  to capture multi-scale context without downsampling.
- A final 1x1 convolution acts as the classifier, followed by bilinear upsampling to restore
  the output to the original image resolution.
- Weights are initialized as in the original DeepLab paper.
"""

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        
        # Add convolutions with different dilation rates
        for rate in dilation_rates:
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=True)
            )
        
        # Weight initialization (same as in `Classifier_Module`)
        for m in self.convs:
            nn.init.normal_(m.weight, mean=0, std=0.01)  # Normal initialization
            nn.init.constant_(m.bias, 0)  # Initialize bias to 0

    def forward(self, x):
        # Sum the outputs of the dilated convolutions
        out = sum(conv(x) for conv in self.convs)
        return out  # No ReLU, return the result directly


# Removes the classifier from ResNet and adjusts convolutions for the dilated backbone
def resnet101_backbone_dilated():
    model = resnet101(weights = "IMAGENET1K_V2")
    layers = list(model.children())

    # Modify dilation for conv4 and conv5
    # conv4_x: dilation 2, stride 1
    for n, m in layers[5].named_modules():
        if 'conv2' in n:
            m.dilation = (2, 2)
            m.padding = (2, 2)
        if 'downsample.0' in n:
            m.stride = (1, 1)
        elif isinstance(m, nn.Conv2d):
            m.stride = (1, 1)

    # conv5_x: dilation 4, stride 1
    for n, m in layers[6].named_modules():
        if 'conv2' in n:
            m.dilation = (4, 4)
            m.padding = (4, 4)
        if 'downsample.0' in n:
            m.stride = (1, 1)
        elif isinstance(m, nn.Conv2d):
            m.stride = (1, 1)

    backbone = nn.Sequential(*layers[:8])  # up to conv5_x
    return backbone


# Full DeepLabV2
class DeepLabV2(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV2, self).__init__()
        self.backbone = resnet101_backbone_dilated()
        self.aspp = ASPP(in_channels=2048, out_channels=512, dilation_rates=[6, 12, 18, 24])
        self.classifier = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = self.backbone(x)

        aspp_out = self.aspp(features)

        out = self.classifier(aspp_out)
        # Note: it's unclear in the paper if the bilinear interpolate is to use both for training (and therefore included in the loss function) or just for test/val
        # In this implementation, bilinear is used every time. The output is then subsampled only for the loss function during training (externally)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out