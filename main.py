# main.py
import torch
from src.dataset import get_dataloaders
from src.model   import build_model, load_model
from src.train   import run_training
from src.evaluate import full_evaluate
import config, os

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    # Save class names for the app
    import json
    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f)
    os.makedirs("models", exist_ok=True)

    # --- TRAIN ---
    model = build_model(num_classes)
    run_training(model, train_loader, val_loader)

    # --- EVALUATE on test set ---
    best_ckpt = f"{config.CHECKPOINT_DIR}/best_model.pth"
    model     = load_model(num_classes, best_ckpt)
    full_evaluate(model, test_loader, class_names)