from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from medmnist import ChestMNIST
import torch
import numpy as np

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),                 
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_sampler(dataset):
    train_labels = dataset.labels
    class_counts = np.sum(train_labels, axis=0)
    class_weights = 1.0 / class_counts

    sample_weights = []

    for label in train_labels:
        active_indices = np.where(label == 1)[0]

        if len(active_indices) > 0:
            weight = np.max(class_weights[active_indices])
        else:
            weight = np.mean(class_weights)

        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    return sampler

def get_dataloaders(batch_size=32, num_workers=4, root='./data'):
    train_transform = get_train_transform()
    val_test_transform = get_val_test_transform()
    
    train_ds = ChestMNIST(root=root, split='train', size=224, transform=train_transform, download=True)
    validation_ds = ChestMNIST(root=root, split='val', size=224, transform=val_test_transform, download=True)
    test_ds = ChestMNIST(root=root, split='test', size=224, transform=val_test_transform, download=True)

    sampler = get_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
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