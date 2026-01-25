from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from medmnist import ChestMNIST
import torch
import numpy as np

norm_mean = [0.5056, 0.5056, 0.5056]
norm_std = [0.252, 0.252, 0.252]

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def get_train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

def get_val_test_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

def get_dataloaders(batch_size=32, num_workers=4, root='./data'):
    train_transform = get_train_transform()
    val_test_transform = get_val_test_transform()
    
    train_ds = ChestMNIST(root=root, split='train', size=224, transform=train_transform, download=True)
    validation_ds = ChestMNIST(root=root, split='val', size=224, transform=val_test_transform, download=True)
    test_ds = ChestMNIST(root=root, split='test', size=224, transform=val_test_transform, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        validation_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader