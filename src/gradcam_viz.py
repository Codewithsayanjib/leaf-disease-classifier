# src/gradcam_viz.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import config

def get_gradcam(model, img_path: str, class_names: list):
    """Generate GradCAM heatmap for a single image."""
    model.eval()

    # Target layer (last conv block of EfficientNetV2-S)
    target_layer = [model.conv_head]

    preprocess = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    raw_img = Image.open(img_path).convert("RGB")
    raw_img = raw_img.resize((config.IMG_SIZE, config.IMG_SIZE))
    input_tensor = preprocess(raw_img).unsqueeze(0).to(config.DEVICE)

    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(raw_img) / 255.0
    visualization = show_cam_on_image(rgb_img.astype(np.float32),
                                      grayscale_cam, use_rgb=True)

    # Prediction
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.softmax(out, dim=1)
        top5 = torch.topk(prob, 5)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(visualization); axes[1].set_title("GradCAM"); axes[1].axis("off")

    pred_class = class_names[top5.indices[0][0].item()]
    pred_conf  = top5.values[0][0].item()
    fig.suptitle(f"Predicted: {pred_class} ({pred_conf:.2%})", fontsize=13, fontweight="bold")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = f"{config.PLOTS_DIR}/gradcam_{os.path.basename(img_path)}"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"GradCAM saved → {out_path}")
    return pred_class, pred_conf