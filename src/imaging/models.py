"""Model architectures, factored into a real module so checkpoints trained in
train_segmentation_detection.ipynb (where these classes are defined inline) can be
reconstructed and loaded elsewhere -- e.g. imaging_confound_check.ipynb, which needs a
fresh, uncompiled ResNet50UNet to run Grad-CAM against.

ResNet50UNet here is architecturally identical to the one in the training notebook --
kept in sync manually since the notebook version is the trained, verified original and
isn't being refactored to import from here (out of scope for the confound check).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class UpBlock(nn.Module):
    """One U-Net decoder stage: upsample 2x, concat with the matching encoder skip
    connection, then two conv/BN/ReLU layers to mix the concatenated features."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet50UNet(nn.Module):
    """Pretrained ResNet-50 encoder + U-Net decoder (segmentation) + a detection head off
    the encoder bottleneck. 3-channel input (replicated from 1 by the Dataset, not stored
    that way). num_seg_classes=3: background/pancreas/tumour.

    pretrained=False skips downloading/loading ImageNet weights for the encoder -- use this
    when about to immediately overwrite every weight via load_state_dict() from a trained
    checkpoint anyway (e.g. in the confound check), so the pretrained load is pure waste.
    """

    def __init__(self, num_seg_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64ch
        self.maxpool = backbone.maxpool  # /4 total
        self.layer1 = backbone.layer1  # /4, 256ch
        self.layer2 = backbone.layer2  # /8, 512ch
        self.layer3 = backbone.layer3  # /16, 1024ch
        self.layer4 = backbone.layer4  # /32, 2048ch

        self.up4 = UpBlock(2048, 1024, 1024)  # -> /16, matches layer3
        self.up3 = UpBlock(1024, 512, 512)  # -> /8, matches layer2
        self.up2 = UpBlock(512, 256, 256)  # -> /4, matches layer1
        self.up1 = UpBlock(256, 64, 64)  # -> /2, matches stem output
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # -> /1, full res
        self.seg_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)

        self.det_pool = nn.AdaptiveAvgPool2d(1)
        self.det_head = nn.Linear(2048, 1)

    def forward(self, x):
        s0 = self.stem(x)
        p0 = self.maxpool(s0)
        l1 = self.layer1(p0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        d4 = self.up4(l4, l3)
        d3 = self.up3(d4, l2)
        d2 = self.up2(d3, l1)
        d1 = self.up1(d2, s0)
        d0 = self.final_up(d1)
        seg_logits = self.seg_head(d0)

        det_feat = self.det_pool(l4).flatten(1)
        det_logit = self.det_head(det_feat).squeeze(-1)

        return seg_logits, det_logit


class DetectionOnlyWrapper(nn.Module):
    """Grad-CAM (pytorch-grad-cam) expects a model whose forward() returns exactly the
    score(s) to explain. ResNet50UNet.forward() returns (seg_logits, det_logit) -- this
    wrapper exposes just det_logit, since the confound check is specifically about
    whether the DETECTION head learned a scanner/acquisition shortcut, not segmentation."""

    def __init__(self, base_model: ResNet50UNet):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        _, det_logit = self.base_model(x)
        return det_logit
