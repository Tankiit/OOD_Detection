import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
import timm
from typing import Optional, Tuple, List, Dict
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import json
import requests
import tarfile
from PIL import Image
import io

class CIFAR10C(Dataset):
    """CIFAR-10-C dataset for corruption-based OOD testing"""
    def __init__(self, root: str, corruption_type: str = 'gaussian_noise', severity: int = 1,
                 transform=None, download: bool = False):
        self.root = root
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # Load corrupted data
        self.data = self._load_corrupted_data()
        
        # Load original labels (corrupted data uses same labels as CIFAR-10)
        cifar10 = CIFAR10(root=root, train=False, download=True)
        self.targets = cifar10.targets
    
    def _load_corrupted_data(self):
        """Load corrupted data for specified corruption type and severity"""
        corruption_path = os.path.join(self.root, 'CIFAR-10-C', f'{self.corruption_type}.npy')
        if not os.path.exists(corruption_path):
            raise ValueError(f"Corruption type {self.corruption_type} not found at {corruption_path}")
        
        print(f"Loading {self.corruption_type} corruption data...")
        # Load all severities
        corrupted_data = np.load(corruption_path)
        # Select specified severity (severity is 1-based)
        return corrupted_data[(self.severity - 1) * 10000:self.severity * 10000]
    
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.data)

def get_cifar10_data(batch_size: int = 64, num_workers: int = 2, model_name: Optional[str] = None):
    """Load CIFAR10 dataset with appropriate transforms"""
    if model_name and 'vit' in model_name:
        # Vision Transformer specific transforms
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard transforms for CNN models
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    trainset = CIFAR10(root='/Users/mukher74/research/data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testset = CIFAR10(root='/Users/mukher74/research/data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def get_ood_data(batch_size: int = 64, num_workers: int = 2, model_name: Optional[str] = None,
                corruption_type: str = 'gaussian_noise', severity: int = 1):
    """Load OOD datasets (SVHN and CIFAR-10-C)"""
    if model_name and 'vit' in model_name:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Load SVHN
    svhn_dataset = SVHN(root='/Users/mukher74/research/data', split='test', download=True, transform=transform)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Load CIFAR-10-C if corruption_type is specified
    ood_loaders = {'svhn': svhn_loader}
    
    if corruption_type is not None:
        try:
            cifar10c_dataset = CIFAR10C(
                root='/Users/mukher74/research/data',
                corruption_type=corruption_type,
                severity=severity,
                transform=transform,
                download=False  # We have the data locally
            )
            cifar10c_loader = DataLoader(cifar10c_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            ood_loaders['cifar10c'] = cifar10c_loader
        except Exception as e:
            print(f"Warning: Could not load CIFAR-10-C data: {e}")
            print("Continuing with SVHN only...")
    
    return ood_loaders

def cifar_demonstration(model_name: str = 'resnet18', save_dir: str = 'checkpoints',
                       corruption_type: str = 'gaussian_noise', severity: int = 1):
    """Complete demonstration with visualization and logging"""
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load datasets with appropriate transforms
    train_loader, test_loader = get_cifar10_data(batch_size=64, num_workers=2, model_name=model_name)
    ood_loaders = get_ood_data(batch_size=64, num_workers=2, model_name=model_name,
                              corruption_type=corruption_type, severity=severity)
    
    # Create and train model
    model = get_model(model_name, num_classes=10).to(device)
    dro_detector, history = detection_focused_training(
        model, train_loader, 
        num_epochs=30,
        epsilon=0.05,
        detection_weight=0.1,
        device=device,
        save_dir=save_dir
    )
    
    # Plot training progress
    plot_training_progress(history, save_path=os.path.join(save_dir, f'training_progress_{model_name}.png'))
    
    # Get validation data
    test_subset = Subset(test_loader.dataset, range(2000))
    test_subset_loader = DataLoader(test_subset, batch_size=2000, shuffle=False)
    test_data = next(iter(test_subset_loader))[0].to(device)
    
    # Evaluate on OOD datasets
    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        print(f"\nEvaluating on {ood_name}...")
        ood_subset = Subset(ood_loader.dataset, range(2000))
        ood_subset_loader = DataLoader(ood_subset, batch_size=2000, shuffle=False)
        ood_data = next(iter(ood_subset_loader))[0].to(device)
        
        # Evaluate and plot detection performance
        ood_results = validate_detection_capability(
            dro_detector, 
            test_data,
            ood_data,
            methods=['margin', 'adaptive_margin', 'entropy', 'energy']
        )
        results[ood_name] = ood_results
        
        # Plot detection performance for this OOD dataset
        plot_detection_performance(
            ood_results,
            save_path=os.path.join(save_dir, f'detection_performance_{model_name}_{ood_name}.png')
        )
    
    return dro_detector, results

if __name__ == "__main__":
    # List of models to try
    models = [
        'resnet18',
        'efficientnet_b0',
        'mobilenetv3_small',
        'vit_tiny_patch16_224'
    ]
    
    # List of corruption types to try
    corruption_types = [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic_transform',
        'pixelate',
        'jpeg_compression'
    ]
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Run demonstration for each model
    for model_name in models:
        print(f"\nTraining with {model_name}...")
        
        # First evaluate on SVHN
        dro_detector, svhn_results = cifar_demonstration(
            model_name,
            save_dir=os.path.join(save_dir, 'svhn'),
            corruption_type=None  # This will only use SVHN
        )
        
        # Then evaluate on each corruption type
        for corruption_type in corruption_types:
            print(f"\nEvaluating on {corruption_type}...")
            try:
                _, cifar10c_results = cifar_demonstration(
                    model_name,
                    save_dir=os.path.join(save_dir, corruption_type),
                    corruption_type=corruption_type,
                    severity=1  # You can try different severities (1-5)
                )
            except Exception as e:
                print(f"Warning: Failed to evaluate on {corruption_type}: {e}")
                continue
        
        # Print summary of results
        print(f"\nResults for {model_name}:")
        print("SVHN Results:")
        print(f"Best AUC: {max(r['auc'] for r in svhn_results['svhn'].values()):.4f}")
        print(f"Best separation: {max(r['separation'] for r in svhn_results['svhn'].values()):.4f}")

def get_model(model_name: str = 'resnet18', num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Factory function to get different models"""
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 
                     'mobilenetv3_small', 'vit_tiny_patch16_224']:
        return TimmFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    else:
        # Fallback to simple CNN with proper max pooling
        return SimpleClassifier(num_classes=num_classes)

def get_transforms(model_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get appropriate transforms for different models"""
    if 'vit' in model_name:
        # Vision Transformer specific transforms
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard transforms for CNN models
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    return transform_train, transform_test

def get_cifar10_data(model_name: str = 'resnet18', batch_size: int = 64, num_workers: int = 2):
    """Load CIFAR10 dataset with appropriate transforms"""
    transform_train, transform_test = get_transforms(model_name)
    
    trainset = CIFAR10(root='/Users/mukher74/research/data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testset = CIFAR10(root='/Users/mukher74/research/data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def get_svhn_data(model_name: str = 'resnet18', batch_size: int = 64, num_workers: int = 2):
    """Load SVHN dataset with appropriate transforms"""
    _, transform = get_transforms(model_name)
    
    oodset = SVHN(root='/Users/mukher74/research/data', split='test', download=True, transform=transform)
    oodloader = DataLoader(oodset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return oodloader

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Separate conv and pool layers for better control
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Store indices for backward pass
        self.pool_indices = []
    
    def forward(self, x):
        # Clear previous indices
        self.pool_indices = []
        
        # First conv block
        x = F.relu(self.conv1(x))
        x, indices1 = self.pool1(x)
        self.pool_indices.append(indices1)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x, indices2 = self.pool2(x)
        self.pool_indices.append(indices2)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x, indices3 = self.pool3(x)
        self.pool_indices.append(indices3)
        
        # Flatten and classify
        features = x.view(x.size(0), -1)
        logits = self.classifier(features)
        return logits, features
    
    def get_features(self, x):
        """Get features without storing indices (for inference)"""
        x = F.relu(self.conv1(x))
        x, _ = self.pool1(x)
        x = F.relu(self.conv2(x))
        x, _ = self.pool2(x)
        x = F.relu(self.conv3(x))
        x, _ = self.pool3(x)
        return x.view(x.size(0), -1)

    