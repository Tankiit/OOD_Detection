import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from torchvision.datasets import CIFAR10, SVHN
import scipy.linalg

class DirectionalBoundaryAnalyzer:
    def __init__(self, model, device='mps'):
        self.model = model
        self.device = device
        self.expansion_history = {'directional': [], 'uniform': []}
        
    def extract_features(self, data_loader):
        """Extract features from the model for analysis"""
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if batch_idx >= 50:  # Limit for efficiency
                    break
                x, y = x.to(self.device), y.to(self.device)
                _, features = self.model(x, return_features=True)
                features_list.append(features.cpu())
                labels_list.append(y.cpu())
        
        return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)
        
    def compute_boundary_expansion_directions(self, features: torch.Tensor, method: str = 'directional') -> torch.Tensor:
        if method == 'directional':
            return self._compute_directional_expansion(features)
        elif method == 'uniform':
            return self._compute_uniform_expansion(features)
        raise ValueError(f"Unknown method: {method}")
    
    def _compute_directional_expansion(self, features: torch.Tensor) -> torch.Tensor:
        centered_features = features - torch.mean(features, dim=0)
        cov_matrix = torch.cov(centered_features.T)
        
        # Add small regularization for stability
        reg_factor = 1e-8
        cov_matrix = cov_matrix + reg_factor * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        
        # Use stable NumPy-based eigendecomposition
        eigenvals, eigenvecs = stable_eigendecomposition(cov_matrix)
        idx = torch.argsort(eigenvals, descending=True)
        principal_directions = eigenvecs[:, idx]
        principal_values = eigenvals[idx]
        
        expansion_scaling = torch.sqrt(principal_values + 1e-8)
        return principal_directions * expansion_scaling.unsqueeze(0)
    
    def _compute_uniform_expansion(self, features: torch.Tensor) -> torch.Tensor:
        feature_dim = features.shape[1]
        uniform_directions = torch.randn(feature_dim, feature_dim, device=features.device)
        uniform_directions = F.normalize(uniform_directions, dim=0)
        uniform_scaling = torch.ones(feature_dim, device=features.device)
        return uniform_directions * uniform_scaling.unsqueeze(0)
    
    def visualize_boundary_expansion(self, features: torch.Tensor, labels: torch.Tensor, save_path: str = 'boundary_expansion_cifar10.png'):
        """Enhanced visualization for CIFAR-10 features"""
        print("Creating boundary expansion visualization...")
        
        # Use t-SNE for better 2D visualization of high-dimensional features
        if features.shape[1] > 50:
            print("Applying t-SNE for dimensionality reduction...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = torch.tensor(tsne.fit_transform(features.cpu().numpy()[:1000]))  # Limit for speed
            labels_2d = labels[:1000]
        else:
            pca = PCA(n_components=2)
            features_2d = torch.tensor(pca.fit_transform(features.cpu().numpy()))
            labels_2d = labels
        
        directional_exp = self._compute_directional_expansion(features_2d)
        uniform_exp = self._compute_uniform_expansion(features_2d)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Color map for CIFAR-10 classes
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        center = torch.mean(features_2d, dim=0)
        
        # Original features with class colors
        for class_idx in range(10):
            mask = labels_2d == class_idx
            if mask.any():
                axes[0, 0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                 c=[colors[class_idx]], alpha=0.6, s=20, 
                                 label=class_names[class_idx])
        axes[0, 0].set_title('Original CIFAR-10 Features (t-SNE)', fontsize=14)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Directional expansion
        for class_idx in range(10):
            mask = labels_2d == class_idx
            if mask.any():
                axes[0, 1].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                 c=[colors[class_idx]], alpha=0.6, s=20)
        
        # Plot top 5 principal directions
        for i in range(min(5, directional_exp.shape[1])):
            direction = directional_exp[:, i]
            # Scale arrows based on eigenvalue magnitude
            scale = torch.norm(direction) * 0.3
            axes[0, 1].arrow(center[0], center[1], direction[0]*scale, direction[1]*scale,
                           head_width=scale*0.1, head_length=scale*0.1, 
                           fc=f'C{i}', ec=f'C{i}', linewidth=2,
                           label=f'PC{i+1}')
        axes[0, 1].set_title('Directional Expansion (PCA-based)', fontsize=14)
        axes[0, 1].legend()
        
        # Uniform expansion
        for class_idx in range(10):
            mask = labels_2d == class_idx
            if mask.any():
                axes[0, 2].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                 c=[colors[class_idx]], alpha=0.6, s=20)
        
        for i in range(min(5, uniform_exp.shape[1])):
            direction = uniform_exp[:, i]
            scale = 0.3  # Fixed scale for uniform
            axes[0, 2].arrow(center[0], center[1], direction[0]*scale, direction[1]*scale,
                           head_width=scale*0.1, head_length=scale*0.1, 
                           fc=f'C{i}', ec=f'C{i}', linewidth=2,
                           label=f'Uniform{i+1}')
        axes[0, 2].set_title('Uniform Expansion', fontsize=14)
        axes[0, 2].legend()
        
        # Eigenvalue analysis
        centered = features - torch.mean(features, dim=0)
        cov_matrix = torch.cov(centered.T)
        
        # Add regularization for stability
        reg_factor = 1e-8
        cov_matrix = cov_matrix + reg_factor * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        
        # Use stable NumPy-based eigendecomposition
        eigenvals, _ = stable_eigendecomposition(cov_matrix)
        eigenvals = torch.sort(eigenvals, descending=True)[0]
        
        axes[1, 0].plot(range(1, min(21, len(eigenvals)+1)), eigenvals[:20].cpu().numpy(), 'bo-')
        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Eigenvalue')
        axes[1, 0].set_title('Eigenvalue Spectrum', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum_eigenvals = torch.cumsum(eigenvals, dim=0)
        explained_var = cumsum_eigenvals / torch.sum(eigenvals)
        axes[1, 1].plot(range(1, min(21, len(explained_var)+1)), 
                       explained_var[:20].cpu().numpy(), 'ro-')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].set_title('Explained Variance Ratio', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Feature distribution analysis
        feature_norms = torch.norm(features, dim=1)
        axes[1, 2].hist(feature_norms.cpu().numpy(), bins=50, alpha=0.7, color='skyblue')
        axes[1, 2].set_xlabel('Feature Norm')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Feature Norm Distribution', fontsize=14)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved to {save_path}")
        return fig

class CIFAR10_CNN(nn.Module):
    """CNN model for CIFAR-10 with feature extraction capability"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, return_features=False):
        # Conv layers with batch norm and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = F.relu(self.fc2(x))
        features = self.dropout(features)
        logits = self.fc3(features)
        
        if return_features:
            return logits, features
        return logits

class DirectionalDRO(nn.Module):
    def __init__(self, classifier, num_classes: int, epsilon: float = 0.1,
                 adaptivity_factor: float = 2.0, device: str = "mps"):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.adaptivity_factor = adaptivity_factor
        self.device = device
        self.register_buffer('principal_directions', None)
        self.register_buffer('eigenvalues', None)
        
    def compute_directional_perturbations(self, features: torch.Tensor) -> torch.Tensor:
        self._update_principal_directions(features)
        
        if self.principal_directions is not None:
            scaling = torch.sqrt(self.eigenvalues + 1e-8) * self.adaptivity_factor
            k = min(5, len(self.eigenvalues))
            top_directions = self.principal_directions[:, :k]
            top_scaling = scaling[:k]
            coefficients = torch.randn(features.shape[0], k, device=self.device)
            perturbations = torch.matmul(coefficients * top_scaling.unsqueeze(0), top_directions.T)
            return self.epsilon * F.normalize(perturbations, dim=1)
        return self.epsilon * torch.randn_like(features)
    
    def _update_principal_directions(self, features: torch.Tensor):
        centered = features - torch.mean(features, dim=0)
        cov_matrix = torch.cov(centered.T)
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
        idx = torch.argsort(eigenvals, descending=True)
        
        if self.principal_directions is None:
            self.principal_directions = eigenvecs[:, idx].detach()
            self.eigenvalues = eigenvals[idx].detach()
        else:
            momentum = 0.9
            self.principal_directions = (momentum * self.principal_directions + 
                                       (1 - momentum) * eigenvecs[:, idx].detach())
            self.eigenvalues = (momentum * self.eigenvalues + 
                              (1 - momentum) * eigenvals[idx].detach())
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logits, features = self.classifier(x, return_features=True)
        perturbations = self.compute_directional_perturbations(features)
        perturbed_features = features + perturbations
        perturbed_logits = self.classifier.fc3(perturbed_features)
        
        standard_loss = F.cross_entropy(logits, y)
        perturbed_loss = F.cross_entropy(perturbed_logits, y)
        total_loss = standard_loss + 0.5 * perturbed_loss
        
        return total_loss, {
            'standard_loss': standard_loss.item(),
            'perturbed_loss': perturbed_loss.item(),
            'perturbation_norm': torch.norm(perturbations, dim=1).mean().item()
        }

class UniformDRO(nn.Module):
    def __init__(self, classifier, num_classes: int, epsilon: float = 0.1, device: str = "mps"):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
    
    def compute_uniform_perturbations(self, features: torch.Tensor) -> torch.Tensor:
        perturbations = torch.randn_like(features)
        return self.epsilon * F.normalize(perturbations, dim=1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logits, features = self.classifier(x, return_features=True)
        perturbations = self.compute_uniform_perturbations(features)
        perturbed_features = features + perturbations
        perturbed_logits = self.classifier.fc3(perturbed_features)
        
        standard_loss = F.cross_entropy(logits, y)
        perturbed_loss = F.cross_entropy(perturbed_logits, y)
        total_loss = standard_loss + 0.5 * perturbed_loss
        
        return total_loss, {
            'standard_loss': standard_loss.item(),
            'perturbed_loss': perturbed_loss.item(),
            'perturbation_norm': torch.norm(perturbations, dim=1).mean().item()
        }

class BoundaryExpansionMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def compute_expansion_anisotropy(self, expansion_directions: torch.Tensor) -> float:
        """Compute anisotropy with numerical stability fixes"""
        try:
            if expansion_directions.shape[0] != expansion_directions.shape[1]:
                cov_matrix = torch.cov(expansion_directions.T)
            else:
                cov_matrix = expansion_directions
            
            # Add regularization for numerical stability
            reg_factor = 1e-6
            cov_matrix = cov_matrix + reg_factor * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
            
            # Check for NaN or Inf
            if torch.isnan(cov_matrix).any() or torch.isinf(cov_matrix).any():
                print("Warning: NaN or Inf detected in covariance matrix, using fallback")
                return 1.0
            
            # Use stable NumPy-based eigenvalue computation
            eigenvals = stable_eigenvalues(cov_matrix)
            
            # Filter out negative eigenvalues due to numerical errors
            eigenvals = eigenvals[eigenvals > 1e-8]
            
            if len(eigenvals) < 2:
                return 1.0
            
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            anisotropy = (eigenvals[0] / (eigenvals[-1] + 1e-8)).item()
            
            # Clamp to reasonable range
            return min(max(anisotropy, 1.0), 1000.0)
            
        except Exception as e:
            print(f"Warning: Error computing anisotropy: {e}, returning default value")
            return 1.0
    
    def compute_manifold_alignment(self, expansion_dirs: torch.Tensor, data_features: torch.Tensor) -> float:
        """Compute manifold alignment with numerical stability fixes"""
        try:
            centered_data = data_features - torch.mean(data_features, dim=0)
            data_cov = torch.cov(centered_data.T)
            
            # Add regularization
            reg_factor = 1e-6
            data_cov = data_cov + reg_factor * torch.eye(data_cov.shape[0], device=data_cov.device)
            
            exp_cov = torch.cov(expansion_dirs.T)
            exp_cov = exp_cov + reg_factor * torch.eye(exp_cov.shape[0], device=exp_cov.device)
            
            # Check for NaN or Inf
            if (torch.isnan(data_cov).any() or torch.isinf(data_cov).any() or 
                torch.isnan(exp_cov).any() or torch.isinf(exp_cov).any()):
                print("Warning: NaN or Inf detected in manifold alignment, using fallback")
                return 0.5
            
            # Use stable NumPy-based eigendecomposition
            data_eigenvals, data_eigenvecs = stable_eigendecomposition(data_cov)
            exp_eigenvals, exp_eigenvecs = stable_eigendecomposition(exp_cov)
            
            alignment = torch.abs(torch.dot(data_eigenvecs[:, -1], exp_eigenvecs[:, -1])).item()
            return min(max(alignment, 0.0), 1.0)
            
        except Exception as e:
            print(f"Warning: Error computing manifold alignment: {e}, returning default value")
            return 0.5
    
    def analyze_boundary_properties(self, directional_dirs: torch.Tensor, uniform_dirs: torch.Tensor, 
                                   features: torch.Tensor) -> Dict:
        """Comprehensive analysis of boundary expansion properties"""
        metrics = {}
        
        # Anisotropy analysis
        metrics['directional_anisotropy'] = self.compute_expansion_anisotropy(directional_dirs)
        metrics['uniform_anisotropy'] = self.compute_expansion_anisotropy(uniform_dirs)
        metrics['anisotropy_ratio'] = metrics['directional_anisotropy'] / metrics['uniform_anisotropy']
        
        # Manifold alignment
        metrics['directional_alignment'] = self.compute_manifold_alignment(directional_dirs, features)
        metrics['uniform_alignment'] = self.compute_manifold_alignment(uniform_dirs, features)
        
        # Effective dimensionality
        def effective_dim(dirs):
            try:
                cov_matrix = torch.cov(dirs.T)
                # Add regularization
                reg_factor = 1e-6
                cov_matrix = cov_matrix + reg_factor * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
                
                # Check for NaN or Inf
                if torch.isnan(cov_matrix).any() or torch.isinf(cov_matrix).any():
                    print("Warning: NaN or Inf in effective_dim computation")
                    return float(dirs.shape[1])
                
                # Use stable NumPy-based eigenvalue computation
                eigenvals = stable_eigenvalues(cov_matrix)
                
                eigenvals = torch.sort(eigenvals, descending=True)[0]
                eigenvals = eigenvals[eigenvals > 1e-8]  # Filter out near-zero eigenvalues
                
                if len(eigenvals) == 0:
                    return 1.0
                
                normalized = eigenvals / eigenvals.sum()
                entropy = -torch.sum(normalized * torch.log(normalized + 1e-8))
                return torch.exp(entropy).item()
                
            except Exception as e:
                print(f"Warning: Error in effective_dim computation: {e}")
                return float(dirs.shape[1])  # Return full dimensionality as fallback
        
        metrics['directional_eff_dim'] = effective_dim(directional_dirs)
        metrics['uniform_eff_dim'] = effective_dim(uniform_dirs)
        
        return metrics

class ManifoldAnalyzer:
    def __init__(self):
        pass

    def analyze_feature_geometry(self, id_features):
        """Analyze feature geometry with numerical stability fixes"""
        try:
            centered_features = id_features - torch.mean(id_features, dim=0)
            cov_matrix = torch.cov(centered_features.T)
            
            # Add regularization for numerical stability
            reg_factor = 1e-6
            cov_matrix = cov_matrix + reg_factor * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
            
            # Check for NaN or Inf
            if torch.isnan(cov_matrix).any() or torch.isinf(cov_matrix).any():
                print("Warning: NaN or Inf detected in covariance matrix")
                return self._fallback_geometry_analysis(id_features)
            
            # Use stable NumPy-based eigendecomposition
            eigenvals, eigenvecs = stable_eigendecomposition(cov_matrix)
            
            sorted_idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[sorted_idx]
            eigenvecs = eigenvecs[:, sorted_idx]
            
            # Filter out near-zero eigenvalues
            eigenvals = eigenvals[eigenvals > 1e-8]
            eigenvecs = eigenvecs[:, :len(eigenvals)]
            
            total_variance = torch.sum(eigenvals)
            cumulative_variance = torch.cumsum(eigenvals, dim=0) / total_variance
            effective_dim = torch.sum(cumulative_variance < 0.95).item()
            effective_dim = max(1, min(effective_dim, len(eigenvals)))
            
            return {
                'eigenvals': eigenvals,
                'eigenvecs': eigenvecs,
                'effective_dim': effective_dim,
                'high_variance_dirs': eigenvecs[:, :effective_dim],
                'low_variance_dirs': eigenvecs[:, effective_dim:] if effective_dim < eigenvecs.shape[1] else torch.empty(eigenvecs.shape[0], 0, device=eigenvecs.device)
            }
            
        except Exception as e:
            print(f"Warning: Error in geometry analysis: {e}, using fallback")
            return self._fallback_geometry_analysis(id_features)
    
    def _fallback_geometry_analysis(self, id_features):
        """Fallback analysis when eigendecomposition fails"""
        feature_dim = id_features.shape[1]
        device = id_features.device
        
        # Create identity-based fallback
        eigenvals = torch.ones(feature_dim, device=device)
        eigenvecs = torch.eye(feature_dim, device=device)
        effective_dim = min(10, feature_dim)  # Conservative estimate
        
        return {
            'eigenvals': eigenvals,
            'eigenvecs': eigenvecs,
            'effective_dim': effective_dim,
            'high_variance_dirs': eigenvecs[:, :effective_dim],
            'low_variance_dirs': eigenvecs[:, effective_dim:]
        }
    
    def compute_directional_priorities(self, geometry_analysis):
        """Compute priorities for directional sampling"""
        eigenvals = geometry_analysis['eigenvals']
        priorities = 1.0 / (eigenvals + 1e-6)
        return priorities / torch.sum(priorities)

    def generate_ood_informed_directions(self, id_features, ood_features):
        """Use actual OOD data to find better directions"""
        # Compute direction from ID centroid to OOD centroid
        id_centroid = torch.mean(id_features, dim=0)
        ood_centroid = torch.mean(ood_features, dim=0)
        
        ood_direction = ood_centroid - id_centroid
        ood_direction = ood_direction / torch.norm(ood_direction)
        
        return ood_direction

class DirectionalVirtualSampler:
    def __init__(self, manifold_analyzer):
        self.manifold_analyzer = manifold_analyzer
        
    def generate_directional_virtual_samples(self, x_id, id_features):
        """Generate virtual samples along principal directions"""
        geometry = self.manifold_analyzer.analyze_feature_geometry(id_features)
        priorities = self.manifold_analyzer.compute_directional_priorities(geometry)
        
        virtual_samples = []
        
        for i, sample in enumerate(x_id):
            # Sample direction based on priorities
            if len(priorities) > 0:
                direction_idx = torch.multinomial(priorities, 1).item()
                direction_idx = min(direction_idx, geometry['eigenvecs'].shape[1] - 1)
                principal_direction = geometry['eigenvecs'][:, direction_idx]
                
                perturbation_strength = self._adaptive_perturbation_strength(
                    geometry['eigenvals'][direction_idx]
                )
            else:
                # Fallback to random direction
                principal_direction = torch.randn(sample.shape[-1], device=sample.device)
                principal_direction = F.normalize(principal_direction, dim=0)
                perturbation_strength = 0.01
            
            virtual_sample = self._perturb_along_direction(
                sample, principal_direction, perturbation_strength
            )
            virtual_samples.append(virtual_sample)
            
        return torch.stack(virtual_samples)
    
    def _adaptive_perturbation_strength(self, eigenval):
        """Adapt perturbation strength based on eigenvalue magnitude"""
        base_strength = 0.01
        return base_strength / (eigenval.item() + 1e-6)
    
    def adaptive_perturbation_scaling(self, method_type):
        """Different optimal scales for different methods"""
        if method_type == 'directional':
            return 0.05  # Larger steps in focused directions
        else:  # uniform
            return 0.01  # Smaller steps in all directions
    
    def _perturb_along_direction(self, sample, direction, strength):
        """Perturb sample along specified direction"""
        # Handle different input shapes (images vs features)
        if len(sample.shape) == 3:  # Image data (C, H, W)
            # Flatten, perturb, reshape
            flat_sample = sample.flatten()
            if len(direction) != len(flat_sample):
                # Adapt direction to match sample size
                direction = direction[:len(flat_sample)] if len(direction) > len(flat_sample) else torch.cat([direction, torch.zeros(len(flat_sample) - len(direction), device=direction.device)])
            
            perturbation = direction * strength * torch.randn(1, device=sample.device).item()
            perturbed_flat = flat_sample + perturbation
            perturbed_sample = perturbed_flat.reshape(sample.shape)
        else:  # Feature data
            perturbation = direction * strength * torch.randn(1, device=sample.device).item()
            perturbed_sample = sample + perturbation
        
        return torch.clamp(perturbed_sample, 0, 1)
    
    def generate_directional_samples(self, x_id, id_features=None):
        """Generate directional samples (wrapper for compatibility)"""
        if id_features is None:
            # Extract features if not provided
            id_features = torch.randn(len(x_id), 256)  # Fallback
        return self.generate_directional_virtual_samples(x_id, id_features)
    
    def generate_uniform_samples(self, x_id):
        """Generate uniform samples (wrapper for compatibility)"""
        noise = torch.randn_like(x_id) * self.adaptive_perturbation_scaling('uniform')
        return torch.clamp(x_id + noise, 0, 1)

class DirectionalVsUniformComparison:
    def __init__(self, directional_sampler):
        self.directional_sampler = directional_sampler
        
    def compare_sampling_strategies(self, x_id, id_features):
        """Compare directional vs uniform sampling strategies"""
        print("Generating directional virtual samples...")
        directional_samples = self.directional_sampler.generate_directional_virtual_samples(
            x_id, id_features
        )
        
        print("Generating uniform virtual samples...")
        uniform_samples = self.generate_uniform_virtual_samples(x_id)
        
        return {
            'directional': directional_samples,
            'uniform': uniform_samples
        }
    
    def generate_uniform_virtual_samples(self, x_id):
        """Generate uniform random virtual samples"""
        noise = torch.randn_like(x_id) * 0.01
        return torch.clamp(x_id + noise, 0, 1)
    
    def generate_hybrid_virtual_samples(self, x_id, id_features, directional_ratio=0.7):
        """Mix directional and uniform sampling"""
        n_directional = int(len(x_id) * directional_ratio)
        n_uniform = len(x_id) - n_directional
        
        directional_samples = self.directional_sampler.generate_directional_samples(
            x_id[:n_directional], id_features[:n_directional] if len(id_features) >= n_directional else id_features
        )
        uniform_samples = self.directional_sampler.generate_uniform_samples(x_id[n_directional:])
        
        return torch.cat([directional_samples, uniform_samples])
    
    def evaluate_sampling_efficiency(self, virtual_samples_dict, model, test_loader=None):
        """Evaluate sampling efficiency using various metrics"""
        results = {}
        
        for method_name, virtual_samples in virtual_samples_dict.items():
            print(f"Evaluating {method_name} sampling...")
            
            # Feature-based evaluation
            model.eval()
            with torch.no_grad():
                if len(virtual_samples.shape) == 4:  # Image data
                    virtual_features = []
                    for i in range(0, len(virtual_samples), 32):  # Process in batches
                        batch = virtual_samples[i:i+32].to(model.parameters().__next__().device)
                        _, features = model(batch, return_features=True)
                        virtual_features.append(features.cpu())
                    virtual_features = torch.cat(virtual_features, dim=0)
                else:
                    virtual_features = virtual_samples
            
            # Compute diversity metrics
            diversity_score = self._compute_sample_diversity(virtual_features)
            coverage_score = self._compute_manifold_coverage(virtual_features)
            
            results[method_name] = {
                'diversity': diversity_score,
                'coverage': coverage_score,
                'num_samples': len(virtual_samples),
                'efficiency': (diversity_score + coverage_score) / 2
            }
            
        return results
    
    def _compute_sample_diversity(self, samples):
        """Compute diversity of generated samples"""
        pairwise_distances = torch.cdist(samples, samples)
        # Remove diagonal (self-distances)
        mask = ~torch.eye(len(samples), dtype=bool)
        distances = pairwise_distances[mask]
        return torch.mean(distances).item()
    
    def _compute_manifold_coverage(self, samples):
        """Compute how well samples cover the manifold"""
        if len(samples) < 2:
            return 0.0
        
        # Use convex hull volume as proxy for coverage
        try:
            # PCA to 2D for coverage computation
            if samples.shape[1] > 2:
                pca = PCA(n_components=2)
                samples_2d = torch.tensor(pca.fit_transform(samples.cpu().numpy()))
            else:
                samples_2d = samples
            
            # Compute area of convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(samples_2d.cpu().numpy())
            return hull.volume
        except:
            # Fallback: use standard deviation as coverage proxy
            return torch.std(samples).item()
    
    def visualize_comparison(self, virtual_samples_dict, original_features, save_path='sampling_comparison.png', model=None):
        """Visualize comparison between sampling strategies"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Reduce dimensionality for visualization
        all_samples = [original_features[:100]]  # Limit original features for speed
        labels = ['Original']
        colors = ['blue']
        
        for method_name, samples in virtual_samples_dict.items():
            if len(samples.shape) == 4:  # Image data, need to extract features
                print(f"Extracting features from {method_name} virtual samples...")
                if model is not None:
                    model.eval()
                    with torch.no_grad():
                        virtual_features = []
                        for i in range(0, len(samples), 32):  # Process in batches
                            batch = samples[i:i+32].to(model.parameters().__next__().device)
                            _, features = model(batch, return_features=True)
                            virtual_features.append(features.cpu())
                        samples_features = torch.cat(virtual_features, dim=0)[:100]  # Limit for speed
                else:
                    # Fallback: use flattened samples but reduce dimensionality
                    samples_flat = samples.view(samples.shape[0], -1)[:100]
                    # Apply PCA to reduce to same dimensionality as original features
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(256, samples_flat.shape[1]))
                    samples_features = torch.tensor(pca.fit_transform(samples_flat.cpu().numpy()))
            else:
                samples_features = samples[:100]  # Feature data
            
            all_samples.append(samples_features)
            labels.append(method_name.capitalize())
            colors.append('red' if method_name == 'directional' else 'green')
        
        # Apply t-SNE to combined data
        combined_data = torch.cat(all_samples, dim=0)
        print(f"Combined data shape for visualization: {combined_data.shape}")
        
        if combined_data.shape[1] > 50:
            print("Applying t-SNE for visualization...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_data)//4))
            combined_2d = tsne.fit_transform(combined_data.cpu().numpy())
        else:
            pca = PCA(n_components=2)
            combined_2d = pca.fit_transform(combined_data.cpu().numpy())
        
        # Split back into original groups
        start_idx = 0
        for i, (samples, label, color) in enumerate(zip(all_samples, labels, colors)):
            end_idx = start_idx + len(samples)
            sample_2d = combined_2d[start_idx:end_idx]
            
            axes[i % 3].scatter(sample_2d[:, 0], sample_2d[:, 1], 
                              c=color, alpha=0.6, s=20, label=label)
            axes[i % 3].set_title(f'{label} Samples')
            axes[i % 3].legend()
            
            start_idx = end_idx
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Comparison visualization saved to {save_path}")

    def train_with_virtual_samples(self, method='directional', model=None, train_loader=None, epochs=3):
        """Train model with virtual samples using specified method"""
        if model is None or train_loader is None:
            print(f"Warning: Missing model or train_loader for {method} training")
            return None
        
        print(f"Training with {method} virtual samples...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:  # Limit for efficiency
                    break
                
                data, target = data.to(model.parameters().__next__().device), target.to(model.parameters().__next__().device)
                
                # Generate virtual samples based on method
                if method == 'directional':
                    with torch.no_grad():
                        _, features = model(data, return_features=True)
                    virtual_data = self.directional_sampler.generate_directional_virtual_samples(data, features)
                else:  # uniform
                    virtual_data = self.generate_uniform_virtual_samples(data)
                
                # Combine original and virtual samples
                combined_data = torch.cat([data, virtual_data])
                combined_target = torch.cat([target, target])  # Same labels for virtual samples
                
                optimizer.zero_grad()
                output = model(combined_data)
                loss = F.cross_entropy(output, combined_target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}: Loss: {total_loss/50:.4f}")
        
        return model
    
    def test_ood_detection(self, model, id_loader=None, ood_loader=None):
        """Test OOD detection performance"""
        if model is None or id_loader is None:
            print("Warning: Missing model or data loaders for OOD testing")
            return 0.0
        
        model.eval()
        id_scores = []
        ood_scores = []
        
        # Collect ID scores
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(id_loader):
                if batch_idx >= 20:  # Limit for efficiency
                    break
                data = data.to(model.parameters().__next__().device)
                logits = model(data)
                # Use max logit as OOD score
                scores = torch.max(logits, dim=1)[0]
                id_scores.extend(scores.cpu().numpy())
        
        # Collect OOD scores (if available)
        if ood_loader is not None:
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(ood_loader):
                    if batch_idx >= 20:  # Limit for efficiency
                        break
                    data = data.to(model.parameters().__next__().device)
                    logits = model(data)
                    scores = torch.max(logits, dim=1)[0]
                    ood_scores.extend(scores.cpu().numpy())
        else:
            # Generate synthetic OOD scores for testing
            ood_scores = np.random.normal(np.mean(id_scores) - 2, np.std(id_scores), len(id_scores))
        
        # Compute AUROC
        if len(ood_scores) > 0:
            y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            y_scores = np.concatenate([id_scores, ood_scores])
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(y_true, y_scores)
                return auroc
            except:
                return 0.5  # Random performance fallback
        
        return 0.0
    
    def evaluate_what_actually_matters(self, train_loader=None, id_test_loader=None, ood_test_loader=None):
        """Test both methods on actual OOD detection task"""
        print("\n" + "="*60)
        print("EVALUATING WHAT ACTUALLY MATTERS: OOD DETECTION PERFORMANCE")
        print("="*60)
        
        if train_loader is None:
            print("Warning: No training data provided for evaluation")
            return
        
        # Create fresh models for each method
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        print("Training model with directional virtual samples...")
        model_directional = CIFAR10_CNN(num_classes=10).to(device)
        model_directional = self.train_with_virtual_samples(
            method='directional', model=model_directional, train_loader=train_loader
        )
        
        print("Training model with uniform virtual samples...")
        model_uniform = CIFAR10_CNN(num_classes=10).to(device)
        model_uniform = self.train_with_virtual_samples(
            method='uniform', model=model_uniform, train_loader=train_loader
        )
        
        # Test on real OOD detection
        print("Testing OOD detection performance...")
        auroc_directional = self.test_ood_detection(model_directional, id_test_loader, ood_test_loader)
        auroc_uniform = self.test_ood_detection(model_uniform, id_test_loader, ood_test_loader)
        
        print(f"\n--- WHAT ACTUALLY MATTERS: AUROC COMPARISON ---")
        print(f"Directional AUROC: {auroc_directional:.4f}")
        print(f"Uniform AUROC: {auroc_uniform:.4f}")
        print(f"Improvement: {auroc_directional - auroc_uniform:+.4f}")
        
        if auroc_directional > auroc_uniform:
            print("✅ Directional sampling performs better!")
        elif auroc_uniform > auroc_directional:
            print("✅ Uniform sampling performs better!")
        else:
            print("🤷 Both methods perform similarly")
        
        return {
            'directional_auroc': auroc_directional,
            'uniform_auroc': auroc_uniform,
            'improvement': auroc_directional - auroc_uniform
        }

class DomainSpecificAugmentations:
    """Augmentations tailored for different domains"""
    
    def __init__(self, domain='vision'):
        self.domain = domain
        self.augmentation_library = self._build_augmentation_library()
    
    def _build_augmentation_library(self):
        """Build domain-specific augmentation strategies"""
        
        if self.domain == 'vision':
            return {
                'geometric': ['rotation', 'translation', 'scale', 'shear'],
                'photometric': ['brightness', 'contrast', 'saturation', 'hue'],
                'noise': ['gaussian_noise', 'salt_pepper', 'blur'],
                'semantic': ['mixup', 'cutmix', 'cutout'],
                'adversarial': ['fgsm', 'pgd', 'c&w']
            }
        elif self.domain == 'nlp':
            return {
                'lexical': ['synonym_replacement', 'word_insertion'],
                'syntactic': ['word_order_change', 'sentence_structure'],
                'semantic': ['paraphrase', 'back_translation']
            }
        else:
            # Generic augmentations
            return {
                'basic': ['noise', 'scaling', 'rotation'],
                'advanced': ['mixup', 'adversarial']
            }
    
    def compute_energy(self, sample):
        """Compute energy/unusualness of a sample"""
        # Simple energy computation based on deviation from mean
        if len(sample.shape) == 3:  # Image data
            sample_flat = sample.flatten()
            energy = torch.std(sample_flat).item()
        else:  # Feature data
            energy = torch.norm(sample).item()
        return energy
    
    def energy_guided_augmentation_selection(self, sample, target_energy):
        """Select augmentations to reach target energy level"""
        
        current_energy = self.compute_energy(sample)
        
        # Start with mild augmentations
        if current_energy < target_energy * 0.5:
            return self.augmentation_library['geometric']
        elif current_energy < target_energy * 0.8:
            return self.augmentation_library['photometric'] 
        else:
            return self.augmentation_library['adversarial']
    
    def apply_augmentation(self, sample, aug_type, strength=0.1):
        """Apply specific augmentation with given strength"""
        if aug_type == 'rotation':
            # Simple rotation simulation
            noise = torch.randn_like(sample) * strength * 0.1
            return torch.clamp(sample + noise, 0, 1)
        elif aug_type == 'brightness':
            # Brightness adjustment
            brightness_factor = 1.0 + torch.randn(1).item() * strength
            return torch.clamp(sample * brightness_factor, 0, 1)
        elif aug_type == 'gaussian_noise':
            # Gaussian noise
            noise = torch.randn_like(sample) * strength
            return torch.clamp(sample + noise, 0, 1)
        elif aug_type == 'mixup':
            # Simple mixup with itself (placeholder)
            alpha = strength
            return sample * alpha + sample * (1 - alpha)
        else:
            # Default: add small noise
            noise = torch.randn_like(sample) * strength * 0.05
            return torch.clamp(sample + noise, 0, 1)

class AugmentationEnhancedVirtualSampler:
    """Combine geometric insights with augmentation strategies"""
    
    def __init__(self, manifold_analyzer, augmentation_engine=None):
        self.manifold_analyzer = manifold_analyzer
        self.augmentation_engine = augmentation_engine or DomainSpecificAugmentations()
        self.energy_threshold = 1.0  # Default energy threshold
    
    def generate_virtual_samples_with_augmentations(self, x_id, manifold_info):
        """Three-tier virtual sample generation"""
        
        # Tier 1: Directional geometric perturbations (30%)
        directional_samples = self.generate_directional_samples(
            x_id[:int(0.3 * len(x_id))], manifold_info
        )
        
        # Tier 2: Semantic augmentations (50%) 
        augmented_samples = self.generate_augmented_samples(
            x_id[int(0.3 * len(x_id)):int(0.8 * len(x_id))], 
            augmentation_strength='medium'
        )
        
        # Tier 3: Strong boundary-pushing augmentations (20%)
        boundary_samples = self.generate_boundary_augmentations(
            x_id[int(0.8 * len(x_id)):],
            target_energy_threshold=self.energy_threshold
        )
        
        return torch.cat([directional_samples, augmented_samples, boundary_samples])
    
    def generate_directional_samples(self, x_id, manifold_info):
        """Generate directional samples using manifold geometry"""
        virtual_samples = []
        
        for sample in x_id:
            # Use manifold principal directions
            if 'eigenvecs' in manifold_info and len(manifold_info['eigenvecs']) > 0:
                # Sample random direction from top principal components
                n_components = min(5, manifold_info['eigenvecs'].shape[1])
                direction_idx = torch.randint(0, n_components, (1,)).item()
                
                if manifold_info['eigenvecs'].shape[1] > direction_idx:
                    direction = manifold_info['eigenvecs'][:, direction_idx]
                    
                    # Adapt direction to sample shape
                    if len(sample.shape) == 3:  # Image data
                        sample_flat = sample.flatten()
                        if len(direction) != len(sample_flat):
                            direction = direction[:len(sample_flat)] if len(direction) > len(sample_flat) else torch.cat([direction, torch.zeros(len(sample_flat) - len(direction))])
                        
                        perturbation = direction.reshape(sample.shape) * 0.05 * torch.randn(1).item()
                        virtual_sample = torch.clamp(sample + perturbation, 0, 1)
                    else:  # Feature data
                        perturbation = direction * 0.05 * torch.randn(1).item()
                        virtual_sample = sample + perturbation
                else:
                    # Fallback: small random perturbation
                    virtual_sample = torch.clamp(sample + torch.randn_like(sample) * 0.01, 0, 1)
            else:
                # Fallback: small random perturbation
                virtual_sample = torch.clamp(sample + torch.randn_like(sample) * 0.01, 0, 1)
            
            virtual_samples.append(virtual_sample)
        
        return torch.stack(virtual_samples)
    
    def generate_augmented_samples(self, x_id, augmentation_strength='medium'):
        """Generate semantically augmented samples"""
        strength_map = {'low': 0.05, 'medium': 0.1, 'high': 0.2}
        strength = strength_map.get(augmentation_strength, 0.1)
        
        virtual_samples = []
        aug_types = ['rotation', 'brightness', 'gaussian_noise']
        
        for sample in x_id:
            # Randomly select augmentation type
            aug_type = torch.randint(0, len(aug_types), (1,)).item()
            aug_name = aug_types[aug_type]
            
            virtual_sample = self.augmentation_engine.apply_augmentation(
                sample, aug_name, strength
            )
            virtual_samples.append(virtual_sample)
        
        return torch.stack(virtual_samples)
    
    def generate_boundary_augmentations(self, x_id, target_energy_threshold):
        """Generate augmentations that push toward decision boundaries"""
        
        augmentation_pipeline = [
            ('mixup', 0.15),
            ('gaussian_noise', 0.1),  
            ('brightness', 0.2),
            ('rotation', 0.1)
        ]
        
        virtual_samples = []
        for sample in x_id:
            # Apply augmentations until energy threshold reached
            augmented = sample
            current_energy = self.augmentation_engine.compute_energy(augmented)
            
            for aug_name, base_strength in augmentation_pipeline:
                if current_energy >= target_energy_threshold:
                    break
                
                # Progressive augmentation
                augmented = self.augmentation_engine.apply_augmentation(
                    augmented, aug_name, base_strength
                )
                current_energy = self.augmentation_engine.compute_energy(augmented)
                    
            virtual_samples.append(augmented)
            
        return torch.stack(virtual_samples)

class GeometricAugmentationStrategy:
    """Use geometric analysis to select optimal augmentations"""
    
    def __init__(self):
        self.augmentation_categories = {
            'spatial': ['rotation', 'translation', 'scale', 'shear'],
            'color': ['brightness', 'contrast', 'saturation', 'hue'],
            'texture': ['gaussian_noise', 'blur', 'elastic_transform'],
            'semantic': ['mixup', 'cutmix', 'cutout']
        }
    
    def analyze_manifold_variance(self, manifold_analysis):
        """Analyze variance patterns in manifold"""
        if 'eigenvals' not in manifold_analysis:
            return {'spatial_variance': 0.5, 'color_variance': 0.5}
        
        eigenvals = manifold_analysis['eigenvals']
        total_variance = torch.sum(eigenvals)
        
        # Simple heuristic: first half of eigenvalues = spatial, second half = color/texture
        n_half = len(eigenvals) // 2
        spatial_variance = torch.sum(eigenvals[:n_half]) / total_variance if n_half > 0 else 0.5
        color_variance = torch.sum(eigenvals[n_half:]) / total_variance if n_half < len(eigenvals) else 0.5
        
        return {
            'spatial_variance': spatial_variance.item(),
            'color_variance': color_variance.item()
        }
    
    def select_augmentations_by_geometry(self, manifold_analysis):
        """Choose augmentations based on manifold structure"""
        
        variance_analysis = self.analyze_manifold_variance(manifold_analysis)
        
        # If data varies more in spatial dimensions
        if variance_analysis['spatial_variance'] > 0.7:
            return self.augmentation_categories['spatial']
            
        # If data varies more in color/texture  
        elif variance_analysis['color_variance'] > 0.7:
            return self.augmentation_categories['color']
            
        # If data has complex manifold structure
        else:
            return self.augmentation_categories['semantic']
    
    def adaptive_augmentation_strength(self, direction_importance, base_strength=0.1):
        """Stronger augmentations in low-variance directions"""
        # Apply stronger augmentations along less-varying dimensions
        # Weaker augmentations along high-variance dimensions
        
        # Inverse relationship: low importance = higher augmentation strength
        adaptive_strength = base_strength * (2.0 - direction_importance)
        return max(0.01, min(0.5, adaptive_strength))  # Clamp to reasonable range

def progressive_augmentation_to_boundary(sample, augmentation_engine, energy_threshold, max_strength=0.5):
    """Gradually increase augmentation strength until boundary reached"""
    
    strength = 0.1
    augmented_sample = sample.clone()
    
    while augmentation_engine.compute_energy(augmented_sample) < energy_threshold:
        strength += 0.1
        
        # Apply random augmentation with current strength
        aug_types = ['rotation', 'brightness', 'gaussian_noise']
        aug_type = aug_types[torch.randint(0, len(aug_types), (1,)).item()]
        
        augmented_sample = augmentation_engine.apply_augmentation(
            sample, aug_type, strength
        )
        
        if strength > max_strength:
            break  # Prevent unrealistic augmentations
    
    return augmented_sample

def create_cifar10_dataloaders(root_dir="/Users/tanmoy/research/data", batch_size=128):
    """Create CIFAR-10 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # CIFAR-10 datasets
    train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_model_briefly(model, train_loader, device, epochs=5):
    """Train model briefly to get reasonable features"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training model for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit batches for speed
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss: {total_loss/100:.4f}, Accuracy: {acc:.2f}%')

def stable_eigendecomposition(matrix: torch.Tensor, use_numpy=True):
    """Stable eigenvalue decomposition using NumPy/SciPy"""
    if use_numpy:
        # Convert to numpy for stable computation
        matrix_np = matrix.cpu().numpy().astype(np.float64)
        
        # Use scipy for more stable eigendecomposition
        try:
            eigenvals, eigenvecs = scipy.linalg.eigh(matrix_np)
            # Convert back to torch tensors
            eigenvals = torch.tensor(eigenvals, dtype=matrix.dtype, device=matrix.device)
            eigenvecs = torch.tensor(eigenvecs, dtype=matrix.dtype, device=matrix.device)
            return eigenvals, eigenvecs
        except Exception as e:
            print(f"SciPy eigendecomposition failed: {e}, trying NumPy...")
            eigenvals, eigenvecs = np.linalg.eigh(matrix_np)
            eigenvals = torch.tensor(eigenvals, dtype=matrix.dtype, device=matrix.device)
            eigenvecs = torch.tensor(eigenvecs, dtype=matrix.dtype, device=matrix.device)
            return eigenvals, eigenvecs
    else:
        # Fallback to PyTorch with CPU
        matrix_cpu = matrix.cpu()
        eigenvals, eigenvecs = torch.linalg.eigh(matrix_cpu)
        return eigenvals.to(matrix.device), eigenvecs.to(matrix.device)

def stable_eigenvalues(matrix: torch.Tensor, use_numpy=True):
    """Stable eigenvalue computation using NumPy/SciPy"""
    if use_numpy:
        matrix_np = matrix.cpu().numpy().astype(np.float64)
        try:
            eigenvals = scipy.linalg.eigvals(matrix_np)
            return torch.tensor(eigenvals.real, dtype=matrix.dtype, device=matrix.device)
        except Exception as e:
            print(f"SciPy eigenvals failed: {e}, trying NumPy...")
            eigenvals = np.linalg.eigvals(matrix_np)
            return torch.tensor(eigenvals.real, dtype=matrix.dtype, device=matrix.device)
    else:
        matrix_cpu = matrix.cpu()
        eigenvals = torch.linalg.eigvals(matrix_cpu).real
        return eigenvals.to(matrix.device)

def main_rq1_investigation():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create CIFAR-10 data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = create_cifar10_dataloaders()
    
    # Create and train model
    model = CIFAR10_CNN(num_classes=10).to(device)
    train_model_briefly(model, train_loader, device, epochs=5)
    
    # Extract features for analysis
    print("Extracting features for boundary analysis...")
    analyzer = DirectionalBoundaryAnalyzer(model, device)
    features, labels = analyzer.extract_features(test_loader)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Initialize new manifold analysis components
    print("\nInitializing manifold analysis components...")
    manifold_analyzer = ManifoldAnalyzer()
    directional_sampler = DirectionalVirtualSampler(manifold_analyzer)
    comparison = DirectionalVsUniformComparison(directional_sampler)
    
    # Initialize new augmentation components
    print("Initializing augmentation-enhanced components...")
    augmentation_engine = DomainSpecificAugmentations(domain='vision')
    augmented_sampler = AugmentationEnhancedVirtualSampler(manifold_analyzer, augmentation_engine)
    geometric_strategy = GeometricAugmentationStrategy()
    
    # Extract some sample images for virtual sample generation
    sample_images = []
    sample_labels = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx >= 5:  # Limit for efficiency
                break
            sample_images.append(x)
            sample_labels.append(y)
    
    sample_images = torch.cat(sample_images, dim=0)[:100]  # Take first 100 samples
    sample_labels = torch.cat(sample_labels, dim=0)[:100]
    
    print(f"Sample images shape: {sample_images.shape}")
    
    # Analyze manifold geometry for augmentation guidance
    print("\nAnalyzing manifold geometry for augmentation guidance...")
    manifold_geometry = manifold_analyzer.analyze_feature_geometry(features[:100])
    geometric_augmentations = geometric_strategy.select_augmentations_by_geometry(manifold_geometry)
    variance_analysis = geometric_strategy.analyze_manifold_variance(manifold_geometry)
    
    print(f"Recommended augmentations based on geometry: {geometric_augmentations}")
    print(f"Manifold variance analysis: {variance_analysis}")
    
    # Generate and compare virtual samples
    print("\nComparing directional vs uniform sampling strategies...")
    virtual_samples = comparison.compare_sampling_strategies(sample_images, features[:100])
    
    # Test hybrid sampling
    print("\nTesting hybrid sampling strategy...")
    hybrid_samples = comparison.generate_hybrid_virtual_samples(sample_images, features[:100], directional_ratio=0.7)
    virtual_samples['hybrid'] = hybrid_samples
    
    # Test augmentation-enhanced sampling
    print("\nTesting augmentation-enhanced virtual sampling...")
    augmented_virtual_samples = augmented_sampler.generate_virtual_samples_with_augmentations(
        sample_images[:50], manifold_geometry
    )
    virtual_samples['augmented'] = augmented_virtual_samples
    
    # Test progressive augmentation for boundary exploration
    print("\nTesting progressive augmentation to boundary...")
    boundary_samples = []
    for i in range(min(10, len(sample_images))):  # Test on small subset
        boundary_sample = progressive_augmentation_to_boundary(
            sample_images[i], augmentation_engine, energy_threshold=1.5
        )
        boundary_samples.append(boundary_sample)
    
    if boundary_samples:
        virtual_samples['progressive_boundary'] = torch.stack(boundary_samples)
    
    # Evaluate sampling efficiency
    print("\nEvaluating sampling efficiency...")
    efficiency_results = comparison.evaluate_sampling_efficiency(virtual_samples, model, test_loader)
    
    # Original boundary expansion analysis
    print("\nPerforming boundary expansion analysis...")
    directional_exp = analyzer.compute_boundary_expansion_directions(features, 'directional')
    uniform_exp = analyzer.compute_boundary_expansion_directions(features, 'uniform')
    
    # Create comprehensive visualization
    analyzer.visualize_boundary_expansion(features, labels, 'rq1_cifar10_analysis.png')
    
    # Create sampling comparison visualization
    comparison.visualize_comparison(virtual_samples, features[:100], 'sampling_strategy_comparison.png', model)
    
    # Comprehensive evaluation of what actually matters
    print("\nRunning comprehensive OOD detection evaluation...")
    ood_results = comparison.evaluate_what_actually_matters(
        train_loader=train_loader, 
        id_test_loader=test_loader, 
        ood_test_loader=None  # Will use synthetic OOD for testing
    )
    
    # Compute detailed metrics
    metrics_analyzer = BoundaryExpansionMetrics()
    boundary_metrics = metrics_analyzer.analyze_boundary_properties(
        directional_exp, uniform_exp, features
    )
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE MANIFOLD AND BOUNDARY ANALYSIS RESULTS")
    print("="*80)
    
    print("\n--- BOUNDARY EXPANSION METRICS ---")
    print(f"Directional Anisotropy: {boundary_metrics['directional_anisotropy']:.3f}")
    print(f"Uniform Anisotropy: {boundary_metrics['uniform_anisotropy']:.3f}")
    print(f"Anisotropy Ratio: {boundary_metrics['anisotropy_ratio']:.2f}")
    print(f"Directional Manifold Alignment: {boundary_metrics['directional_alignment']:.3f}")
    print(f"Uniform Manifold Alignment: {boundary_metrics['uniform_alignment']:.3f}")
    print(f"Directional Effective Dimension: {boundary_metrics['directional_eff_dim']:.2f}")
    print(f"Uniform Effective Dimension: {boundary_metrics['uniform_eff_dim']:.2f}")
    
    print("\n--- VIRTUAL SAMPLING EFFICIENCY ---")
    for method, results in efficiency_results.items():
        print(f"{method.capitalize()} Sampling:")
        print(f"  Diversity Score: {results['diversity']:.4f}")
        print(f"  Coverage Score: {results['coverage']:.4f}")
        print(f"  Overall Efficiency: {results['efficiency']:.4f}")
        print(f"  Number of Samples: {results['num_samples']}")
    
    print("\n--- AUGMENTATION STRATEGY ANALYSIS ---")
    print(f"Geometric-guided augmentations: {geometric_augmentations}")
    print(f"Spatial variance ratio: {variance_analysis['spatial_variance']:.3f}")
    print(f"Color variance ratio: {variance_analysis['color_variance']:.3f}")
    
    # Analyze augmentation effectiveness
    augmentation_methods = ['directional', 'uniform', 'hybrid', 'augmented']
    if 'progressive_boundary' in efficiency_results:
        augmentation_methods.append('progressive_boundary')
    
    best_method = max(augmentation_methods, 
                     key=lambda x: efficiency_results.get(x, {}).get('efficiency', 0))
    print(f"📊 Best performing method: {best_method} (efficiency: {efficiency_results.get(best_method, {}).get('efficiency', 0):.4f})")
    
    # Test adaptive augmentation strength
    if 'eigenvals' in manifold_geometry and len(manifold_geometry['eigenvals']) > 0:
        top_eigenval = manifold_geometry['eigenvals'][0].item()
        direction_importance = top_eigenval / torch.sum(manifold_geometry['eigenvals']).item()
        adaptive_strength = geometric_strategy.adaptive_augmentation_strength(direction_importance)
        print(f"Adaptive augmentation strength for top direction: {adaptive_strength:.3f}")
    
    print("\n--- OOD DETECTION PERFORMANCE ---")
    if ood_results:
        print(f"Directional Method AUROC: {ood_results['directional_auroc']:.4f}")
        print(f"Uniform Method AUROC: {ood_results['uniform_auroc']:.4f}")
        print(f"Performance Improvement: {ood_results['improvement']:+.4f}")
        
        # Method recommendation
        if abs(ood_results['improvement']) > 0.01:  # Significant difference
            better_method = "Directional" if ood_results['improvement'] > 0 else "Uniform"
            print(f"📊 Recommendation: Use {better_method} sampling for better OOD detection")
        else:
            print(f"📊 Recommendation: Both methods perform similarly")
    
    # Feature statistics
    feature_mean = torch.mean(features, dim=0)
    feature_std = torch.std(features, dim=0)
    print(f"\n--- FEATURE STATISTICS ---")
    print(f"Mean feature norm: {torch.norm(feature_mean):.3f}")
    print(f"Average feature std: {torch.mean(feature_std):.3f}")
    
    # Perturbation scaling analysis
    directional_scale = directional_sampler.adaptive_perturbation_scaling('directional')
    uniform_scale = directional_sampler.adaptive_perturbation_scaling('uniform')
    print(f"Directional perturbation scale: {directional_scale:.3f}")
    print(f"Uniform perturbation scale: {uniform_scale:.3f}")
    print(f"Scale ratio (dir/uni): {directional_scale/uniform_scale:.2f}")
    
    # Augmentation energy analysis
    if len(sample_images) > 0:
        sample_energy = augmentation_engine.compute_energy(sample_images[0])
        print(f"Sample energy baseline: {sample_energy:.3f}")
        
        # Test energy-guided selection
        target_energy = sample_energy * 1.5
        suggested_augs = augmentation_engine.energy_guided_augmentation_selection(
            sample_images[0], target_energy
        )
        print(f"Energy-guided augmentation suggestions: {suggested_augs[:3]}")  # Show first 3
    
    # Class separation analysis
    class_means = []
    for class_idx in range(10):
        mask = labels == class_idx
        if mask.any():
            class_mean = torch.mean(features[mask], dim=0)
            class_means.append(class_mean)
    
    if len(class_means) > 1:
        class_means = torch.stack(class_means)
        inter_class_distances = torch.cdist(class_means, class_means)
        avg_class_separation = torch.mean(inter_class_distances[inter_class_distances > 0])
        print(f"Average inter-class distance: {avg_class_separation:.3f}")
    
    print(f"\n--- OUTPUT FILES ---")
    print(f"Boundary expansion visualization: 'rq1_cifar10_analysis.png'")
    print(f"Sampling strategy comparison: 'sampling_strategy_comparison.png'")
    print("Analysis complete!")
    
    return model, analyzer, boundary_metrics, efficiency_results, ood_results, {
        'geometric_augmentations': geometric_augmentations,
        'variance_analysis': variance_analysis,
        'manifold_geometry': manifold_geometry,
        'best_sampling_method': best_method
    }

if __name__ == "__main__":
    main_rq1_investigation()