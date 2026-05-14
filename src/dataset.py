import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import config

def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

def get_dataloaders():
    full_dataset = datasets.ImageFolder(config.DATA_DIR)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    indices = list(range(len(full_dataset)))
    labels  = [full_dataset.targets[i] for i in indices]

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=config.SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=config.SEED
    )

    train_ds = datasets.ImageFolder(config.DATA_DIR, transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(config.DATA_DIR, transform=get_transforms("val"))
    test_ds  = datasets.ImageFolder(config.DATA_DIR, transform=get_transforms("test"))

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(Subset(test_ds, test_idx),
                              batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=False)

    print(f"Classes : {num_classes}")
    print(f"Train   : {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, class_names