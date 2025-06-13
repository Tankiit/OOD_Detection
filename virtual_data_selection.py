import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pytorch_ood.benchmark import ImageNet_OpenOOD
from pytorch_ood.detector import MaxSoftmax
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights






class PromisingIDSelector:
    """
    Identifies most promising ID samples for virtual OOD generation
    Based on boundary-critical analysis and manifold geometry
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.influence_cache = {}
        self.boundary_cache = {}
        
    # =================== I. BOUNDARY-CRITICAL SAMPLE IDENTIFICATION ===================
    
    def compute_influence_scores(self, train_loader, test_samples, method='tracin'):
        """
        A. Influence-Based Sample Ranking
        Compute influence of each training sample on decision boundaries
        """
        if method == 'tracin':
            return self._compute_tracin_scores(train_loader, test_samples)
        elif method == 'influence_function':
            return self._compute_influence_functions(train_loader, test_samples)
        else:
            raise ValueError(f"Unknown influence method: {method}")
    
    def _compute_tracin_scores(self, train_loader, test_samples):
        """TracIn: 100x faster than traditional influence functions"""
        print("Computing TracIn influence scores...")
        
        # Step 1: Compute test gradients
        test_gradients = []
        for test_sample in test_samples:
            test_grad = self._compute_sample_gradient(test_sample)
            test_gradients.append(test_grad)
        
        # Step 2: Compute training sample influences
        influence_scores = []
        
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            train_x, train_y = train_x.to(self.device), train_y.to(self.device)
            
            batch_influences = []
            for i in range(len(train_x)):
                # Compute gradient for this training sample
                train_grad = self._compute_sample_gradient((train_x[i:i+1], train_y[i:i+1]))
                
                # Compute TracIn score: sum of dot products with test gradients
                tracin_score = 0.0
                for test_grad in test_gradients:
                    tracin_score += torch.dot(train_grad.flatten(), test_grad.flatten()).item()
                
                batch_influences.append(tracin_score)
            
            influence_scores.extend(batch_influences)
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx * len(train_x)} training samples...")
        
        return np.array(influence_scores)
    
    def _compute_sample_gradient(self, sample_batch):
        """Compute gradient of loss w.r.t. model parameters for a sample"""
        x, y = sample_batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        grad_params = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_params.append(param.grad.flatten())
        
        return torch.cat(grad_params)
    
    def compute_boundary_proximity(self, data_loader):
        """
        B. Decision Boundary Proximity Analysis
        Identify samples near decision boundaries
        """
        print("Computing boundary proximity scores...")
        
        boundary_distances = []
        gradient_magnitudes = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # 1. Margin-based boundary distance
                batch_distances = self._compute_margin_distances(x, y)
                boundary_distances.extend(batch_distances)
                
                # 2. Gradient magnitude analysis
                batch_gradients = self._compute_gradient_magnitudes(x)
                gradient_magnitudes.extend(batch_gradients)
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx * len(x)} samples for boundary analysis...")
        
        return np.array(boundary_distances), np.array(gradient_magnitudes)
    
    def _compute_margin_distances(self, x, y):
        """Compute distance to decision boundary using margin analysis"""
        # Extract features
        if hasattr(self.model, 'features'):
            features = self.model.features(x)
        else:
            features = self._extract_features(x)
        
        # Get class centroids (simplified approach)
        logits = self.model(x)
        probabilities = F.softmax(logits, dim=1)
        
        # Compute distances to decision boundary
        # Using confidence as proxy for boundary distance
        max_probs, predicted = torch.max(probabilities, dim=1)
        
        # Higher confidence = further from boundary
        boundary_distances = (1.0 - max_probs).cpu().numpy()
        
        return boundary_distances.tolist()
    
    def _compute_gradient_magnitudes(self, x):
        """Compute gradient magnitude w.r.t. input"""
        x_grad = x.clone().requires_grad_(True)
        
        outputs = self.model(x_grad)
        # Use max logit as score
        max_logits, _ = torch.max(outputs, dim=1)
        loss = torch.sum(max_logits)
        
        loss.backward()
        
        # Compute gradient magnitudes
        grad_magnitudes = torch.norm(x_grad.grad.view(len(x), -1), dim=1)
        
        return grad_magnitudes.detach().cpu().numpy().tolist()
    
    # =================== II. MANIFOLD GEOMETRY-BASED SELECTION ===================
    
    def analyze_manifold_geometry(self, data_loader, n_components=50):
        """
        A. High-Variance Direction Sampling
        Analyze data manifold structure for virtual generation guidance
        """
        print("Analyzing manifold geometry...")
        
        # Step 1: Extract all features
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                features = self._extract_features(x)
                all_features.append(features.cpu())
                all_labels.append(y)
                
                if batch_idx % 10 == 0:
                    print(f"  Extracted features from {batch_idx * len(x)} samples...")
        
        features_matrix = torch.cat(all_features, dim=0).numpy()
        labels_vector = torch.cat(all_labels, dim=0).numpy()
        
        # Step 2: Principal Component Analysis
        pca_results = self._compute_pca_analysis(features_matrix, n_components)
        
        # Step 3: Variance direction analysis
        variance_analysis = self._analyze_variance_directions(features_matrix, pca_results)
        
        return {
            'features': features_matrix,
            'labels': labels_vector,
            'pca_results': pca_results,
            'variance_analysis': variance_analysis
        }
    
    def _compute_pca_analysis(self, features, n_components):
        """Compute PCA and analyze principal directions"""
        print("  Computing PCA analysis...")
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        
        # Analyze explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find effective dimensionality (95% variance)
        effective_dim = np.argmax(cumulative_variance >= 0.95) + 1
        
        return {
            'pca': pca,
            'transformed_features': features_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'effective_dimensionality': effective_dim,
            'principal_components': pca.components_
        }
    
    def _analyze_variance_directions(self, features, pca_results):
        """Analyze high and low variance directions for virtual generation"""
        print("  Analyzing variance directions...")
        
        eigenvalues = pca_results['explained_variance_ratio']
        principal_components = pca_results['principal_components']
        
        # Identify high-variance directions (top 20%)
        n_high_var = max(1, int(0.2 * len(eigenvalues)))
        high_variance_directions = principal_components[:n_high_var]
        high_variance_values = eigenvalues[:n_high_var]
        
        # Identify low-variance directions (bottom 20%)
        n_low_var = max(1, int(0.2 * len(eigenvalues)))
        low_variance_directions = principal_components[-n_low_var:]
        low_variance_values = eigenvalues[-n_low_var:]
        
        return {
            'high_variance_directions': high_variance_directions,
            'high_variance_values': high_variance_values,
            'low_variance_directions': low_variance_directions,
            'low_variance_values': low_variance_values,
            'spatial_variance_ratio': np.sum(high_variance_values) / np.sum(eigenvalues)
        }
    
    def detect_manifold_boundaries(self, features, k_neighbors=10):
        """
        B. Manifold Boundary Detection
        Identify samples near manifold boundaries using density analysis
        """
        print("Detecting manifold boundaries...")
        
        # Step 1: Local density analysis
        density_scores = self._compute_local_density(features, k_neighbors)
        
        # Step 2: Manifold curvature assessment
        curvature_scores = self._estimate_manifold_curvature(features, k_neighbors)
        
        return {
            'density_scores': density_scores,
            'curvature_scores': curvature_scores,
            'boundary_candidates': self._identify_boundary_candidates(density_scores, curvature_scores)
        }
    
    def _compute_local_density(self, features, k_neighbors):
        """Compute local density using k-NN"""
        print("  Computing local density scores...")
        
        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 to exclude self
        nbrs.fit(features)
        
        # Compute distances to k-th nearest neighbor
        distances, indices = nbrs.kneighbors(features)
        
        # Density = 1 / (volume of k-NN ball)
        # Approximated as 1 / (distance to k-th neighbor)
        kth_distances = distances[:, -1]  # Distance to k-th neighbor
        density_scores = 1.0 / (kth_distances + 1e-8)  # Add small epsilon for stability
        
        return density_scores
    
    def _estimate_manifold_curvature(self, features, k_neighbors):
        """Estimate local manifold curvature"""
        print("  Estimating manifold curvature...")
        
        curvature_scores = []
        
        # Fit k-NN for neighborhood analysis
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nbrs.fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        for i in range(len(features)):
            # Get neighborhood
            neighbor_indices = indices[i, 1:]  # Exclude self
            neighborhood = features[neighbor_indices]
            
            # Compute local PCA
            centered_neighborhood = neighborhood - np.mean(neighborhood, axis=0)
            
            if len(neighborhood) > 2:  # Need at least 3 points for meaningful PCA
                try:
                    # Compute covariance and eigenvalues
                    cov_matrix = np.cov(centered_neighborhood.T)
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                    
                    # Curvature approximation: ratio of smallest to largest eigenvalue
                    if eigenvals[0] > 1e-8:
                        curvature = eigenvals[-1] / eigenvals[0]
                    else:
                        curvature = 0.0
                except:
                    curvature = 0.0
            else:
                curvature = 0.0
            
            curvature_scores.append(curvature)
        
        return np.array(curvature_scores)
    
    def _identify_boundary_candidates(self, density_scores, curvature_scores):
        """Identify samples that are good candidates for manifold boundary"""
        # Normalize scores
        density_norm = (density_scores - np.min(density_scores)) / (np.max(density_scores) - np.min(density_scores) + 1e-8)
        curvature_norm = (curvature_scores - np.min(curvature_scores)) / (np.max(curvature_scores) - np.min(curvature_scores) + 1e-8)
        
        # Boundary candidates: low density OR high curvature
        # Low density = near manifold edge
        # High curvature = complex local geometry
        boundary_score = (1 - density_norm) + curvature_norm
        
        # Select top 20% as boundary candidates
        threshold = np.percentile(boundary_score, 80)
        boundary_candidates = boundary_score > threshold
        
        return boundary_candidates
    
    # =================== HELPER METHODS ===================
    
    def _extract_features(self, x):
        """Extract features from model (adapt based on your model structure)"""
        if hasattr(self.model, 'features'):
            return self.model.features(x)
        elif hasattr(self.model, 'feature_extractor'):
            return self.model.feature_extractor(x)
        else:
            # For models without explicit feature extraction
            # Extract from penultimate layer
            modules = list(self.model.children())
            for layer in modules[:-1]:  # All layers except the last
                x = layer(x)
            return x.view(x.size(0), -1)  # Flatten
    
    # =================== MAIN SELECTION INTERFACE ===================
    
    def select_promising_samples(self, train_loader, n_select=100, method='composite'):
        """
        Main interface: Select most promising ID samples for virtual generation
        """
        print(f"Selecting {n_select} most promising samples using {method} method...")
        
        if method == 'composite':
            return self._composite_selection(train_loader, n_select)
        elif method == 'influence_only':
            return self._influence_only_selection(train_loader, n_select)
        elif method == 'boundary_only':
            return self._boundary_only_selection(train_loader, n_select)
        elif method == 'manifold_only':
            return self._manifold_only_selection(train_loader, n_select)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _composite_selection(self, train_loader, n_select):
        """Combine influence, boundary, and manifold analysis"""
        # This would implement the composite scoring from the full framework
        # For now, placeholder
        print("Composite selection not fully implemented in skeleton")
        return self._influence_only_selection(train_loader, n_select)
    
    def _influence_only_selection(self, train_loader, n_select):
        """Select based on influence scores only"""
        # Get a few test samples for influence computation
        test_samples = []
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 3:  # Use first 3 batches as test samples
                break
            test_samples.extend([(x[i:i+1], y[i:i+1]) for i in range(min(10, len(x)))])
        
        # Compute influence scores
        influence_scores = self.compute_influence_scores(train_loader, test_samples[:10])
        
        # Select top samples
        top_indices = np.argsort(influence_scores)[-n_select:]
        
        return {
            'selected_indices': top_indices,
            'influence_scores': influence_scores,
            'selection_method': 'influence_only'
        }


# =================== USAGE EXAMPLE ===================

def example_usage():
    device = "mps"
    from torchvision.datasets import CIFAR10,CIFAR100
    from pytorch_ood.dataset.img import (
        LSUNCrop,
        LSUNResize,
        Textures,
        TinyImageNetCrop,
        TinyImageNetResize,
        Places365
    )
    from pytorch_ood.detector import (
        ODIN,
        EnergyBased,
        Entropy,
        KLMatching,
        Mahalanobis,
        MaxLogit,
        MaxSoftmax,
        ViM,
        RMD,
        DICE,
        SHE,
        Gram,
        MultiMahalanobis,
    )
    from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed
    from pytorch_ood.model import resnet50,WideResNet
    from torch.utils.data import DataLoader
    model = resnet50(ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
    selector = PromisingIDSelector(model, device=device)
    
    # Load data
    dataset = CIFAR10(root="/Users/mukher74/research/data", train=True, download=True)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    
    
    

if __name__ == "__main__":
    example_usage()