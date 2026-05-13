# src/evaluate.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import config

@torch.no_grad()
def full_evaluate(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(config.DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix → {config.PLOTS_DIR}/confusion_matrix.png")