# src/gradcam_viz.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
import config

def get_gradcam(model, img_path: str, class_names: list):
    model.eval()
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

    # Overlay heatmap using matplotlib only (no cv2)
    rgb_img = np.array(raw_img).astype(np.float32) / 255.0
    heatmap = cm.jet(grayscale_cam)[:, :, :3]  # RGBA -> RGB
    overlay = 0.5 * rgb_img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # Prediction
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.softmax(out, dim=1)
        top1_idx = prob.argmax(1).item()
        top1_conf = prob[0][top1_idx].item()

    pred_class = class_names[top1_idx]
    pred_conf  = top1_conf

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("GradCAM");  axes[1].axis("off")
    fig.suptitle(f"Predicted: {pred_class} ({pred_conf:.2%})", fontsize=13, fontweight="bold")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = f"{config.PLOTS_DIR}/gradcam_{os.path.basename(img_path)}"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"GradCAM saved → {out_path}")
    return pred_class, pred_conf