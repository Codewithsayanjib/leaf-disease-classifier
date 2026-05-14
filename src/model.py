import torch
import torch.nn as nn
import timm
import config

def build_model(num_classes: int, pretrained: bool = True):
    """EfficientNetV2-S pretrained on ImageNet, fine-tuned head."""
    model = timm.create_model(
    "tf_efficientnetv2_s",  # changed from "efficientnetv2_s"
    pretrained=pretrained,
    num_classes=num_classes
)
    return model.to(config.DEVICE)

def load_model(num_classes: int, checkpoint_path: str):
    model = build_model(num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state["model_state_dict"])
    return model