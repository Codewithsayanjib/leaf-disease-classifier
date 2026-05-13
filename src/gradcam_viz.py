# src/gradcam_viz.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
import config

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam, class_idx, torch.softmax(output, dim=1)[0, class_idx].item()


def get_gradcam(model, img_path: str, class_names: list):
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    raw_img = Image.open(img_path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    input_tensor = preprocess(raw_img).unsqueeze(0).to(config.DEVICE)

    gradcam = GradCAM(model, model.conv_head)
    cam, class_idx, confidence = gradcam(input_tensor)

    # Resize cam to image size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR
    )
    cam_np = np.array(cam_img) / 255.0

    # Overlay using matplotlib only
    rgb_img = np.array(raw_img).astype(np.float32) / 255.0
    heatmap = cm.jet(cam_np)[:, :, :3]
    overlay = np.clip(0.5 * rgb_img + 0.5 * heatmap, 0, 1)

    pred_class = class_names[class_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("GradCAM");  axes[1].axis("off")
    fig.suptitle(f"Predicted: {pred_class} ({confidence:.2%})", fontsize=13, fontweight="bold")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = f"{config.PLOTS_DIR}/gradcam_{os.path.basename(img_path)}"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return pred_class, confidence