
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from se import SEBlock
from cbam import CBAMBlock


class AttentionResNet18(nn.Module):
    def __init__(
        self,
        attention_type="none",
        placement=None,
        num_classes=4,
        pretrained=True
    ):
        super().__init__()

        attention_type = attention_type.lower()

        if attention_type not in ("none", "se", "cbam"):
            raise ValueError("attention_type must be none, se, or cbam")

        if attention_type == "none":
            if placement is not None:
                raise ValueError("Baseline must have placement=None")
        elif placement not in ("early", "middle", "late"):
            raise ValueError("Placement must be early, middle, or late")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace ImageNet's 1000-class output layer.
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

        self.attention_type = attention_type
        self.placement = placement

        channels = {
            "early": 64,
            "middle": 128,
            "late": 512
        }

        if attention_type == "none":
            self.attention = nn.Identity()
        else:
            selected_channels = channels[placement]

            if attention_type == "se":
                self.attention = SEBlock(selected_channels)
            else:
                self.attention = CBAMBlock(selected_channels)

    def forward(self, x):
        model = self.backbone

        # Standard ResNet18 stem
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        # Stage 1: Early
        x = model.layer1(x)
        if self.placement == "early":
            x = self.attention(x)

        # Stage 2: Middle
        x = model.layer2(x)
        if self.placement == "middle":
            x = self.attention(x)

        # Stage 3: No attention placement here
        x = model.layer3(x)

        # Stage 4: Late
        x = model.layer4(x)
        if self.placement == "late":
            x = self.attention(x)

        # Standard ResNet18 classification head
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)

        return x
