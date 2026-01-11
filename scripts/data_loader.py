from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import ChestMNIST

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),                 
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_dataloaders(batch_size=32, num_workers=4, root='./data'):
    transform = get_transform()
    
    train_ds = ChestMNIST(root=root, split='train', size=224, transform=transform, download=True)
    validation_ds = ChestMNIST(root=root, split='val', size=224, transform=transform, download=True)
    test_ds = ChestMNIST(root=root, split='test', size=224, transform=transform, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
