import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class DirectionalBoundaryAnalyzer:
    def __init__(self, model, device='mps'):
        self.model = model
        self.device = device
        self.expansion_history = {'directional': [], 'uniform': []}
        
    def compute_boundary_expansion_directions(self, features: torch.Tensor, method: str = 'directional') -> torch.Tensor:
        if method == 'directional':
            return self._compute_directional_expansion(features)
        elif method == 'uniform':
            return self._compute_uniform_expansion(features)
        raise ValueError(f"Unknown method: {method}")
    
    def _compute_directional_expansion(self, features: torch.Tensor) -> torch.Tensor:
        centered_features = features - torch.mean(features, dim=0)
        cov_matrix = torch.cov(centered_features.T)
        
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
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
    
    def visualize_boundary_expansion(self, features: torch.Tensor, save_path: str = 'boundary_expansion.png'):
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = torch.tensor(pca.fit_transform(features.cpu().numpy()))
        else:
            features_2d = features
        
        directional_exp = self._compute_directional_expansion(features_2d)
        uniform_exp = self._compute_uniform_expansion(features_2d)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        center = torch.mean(features_2d, dim=0)
        
        axes[0].scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20)
        axes[0].set_title('Original Features')
        axes[0].set_aspect('equal')
        
        axes[1].scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20)
        for i in range(min(3, directional_exp.shape[1])):
            direction = directional_exp[:, i]
            axes[1].arrow(center[0], center[1], direction[0]*2, direction[1]*2,
                         head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}')
        axes[1].set_title('Directional Expansion')
        
        axes[2].scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20)
        for i in range(min(3, uniform_exp.shape[1])):
            direction = uniform_exp[:, i]
            axes[2].arrow(center[0], center[1], direction[0]*2, direction[1]*2,
                         head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}')
        axes[2].set_title('Uniform Expansion')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return fig

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
        perturbed_logits = self.classifier.fc(features + perturbations)
        
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
        perturbed_logits = self.classifier.fc(features + perturbations)
        
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
        if expansion_directions.shape[0] != expansion_directions.shape[1]:
            cov_matrix = torch.cov(expansion_directions.T)
        else:
            cov_matrix = expansion_directions
        
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.sort(eigenvals, descending=True)[0]
        return (eigenvals[0] / (eigenvals[-1] + 1e-8)).item()
    
    def compute_manifold_alignment(self, expansion_dirs: torch.Tensor, data_features: torch.Tensor) -> float:
        centered_data = data_features - torch.mean(data_features, dim=0)
        data_cov = torch.cov(centered_data.T)
        data_eigenvals, data_eigenvecs = torch.linalg.eigh(data_cov)
        
        exp_cov = torch.cov(expansion_dirs.T)
        exp_eigenvals, exp_eigenvecs = torch.linalg.eigh(exp_cov)
        
        return torch.abs(torch.dot(data_eigenvecs[:, -1], exp_eigenvecs[:, -1])).item()

def main_rq1_investigation():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x, return_features=False):
            feat = self.features(x)
            logits = self.fc(feat)
            if return_features:
                return logits, feat
            return logits
    
    model = SimpleModel().to(device)
    dummy_features = torch.randn(200, 128, device=device)
    
    analyzer = DirectionalBoundaryAnalyzer(model, device)
    directional_exp = analyzer.compute_boundary_expansion_directions(dummy_features, 'directional')
    uniform_exp = analyzer.compute_boundary_expansion_directions(dummy_features, 'uniform')
    
    analyzer.visualize_boundary_expansion(dummy_features, 'rq1_demo.png')
    
    metrics = BoundaryExpansionMetrics()
    dir_anisotropy = metrics.compute_expansion_anisotropy(directional_exp)
    uni_anisotropy = metrics.compute_expansion_anisotropy(uniform_exp)
    
    print(f"Directional Anisotropy: {dir_anisotropy:.3f}")
    print(f"Uniform Anisotropy: {uni_anisotropy:.3f}")
    print(f"Ratio: {dir_anisotropy/uni_anisotropy:.2f}")

if __name__ == "__main__":
    main_rq1_investigation()