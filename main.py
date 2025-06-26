"""
Research Framework: OOD Detection with Random Labels
Investigating whether OOD detectors measure semantic understanding or statistical artifacts

Research Questions:
RQ1: Semantic vs. Statistical Detection Decomposition
RQ2: Label Dependency Test Across Methods  
RQ3: True Semantic OOD Detection Design
RQ4: Causal Analysis of OOD Detection Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import copy
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE FRAMEWORK: Model Training with Real vs Random Labels
# ============================================================================

class RandomLabelTrainer:
    """Core trainer that can train with real or random labels"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'real': [], 'random': []}
        
    def create_random_labels(self, targets, num_classes):
        """Create random labels maintaining class balance"""
        random_labels = torch.randint(0, num_classes, targets.shape)
        return random_labels
        
    def train_with_label_type(self, train_loader, label_type='real', epochs=100, 
                             lr=0.001, save_checkpoints=True):
        """Train model with either real or random labels"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Create random labels if needed
        if label_type == 'random':
            # Pre-generate random labels for consistency
            random_label_map = {}
            for batch_idx, (data, targets) in enumerate(train_loader):
                random_labels = self.create_random_labels(targets, 10)  # Assuming 10 classes
                random_label_map[batch_idx] = random_labels
        
        history = []
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Use random labels if specified
                if label_type == 'random':
                    targets = random_label_map[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                epoch_acc += (pred == targets).float().mean().item()
            
            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_acc / len(train_loader)
            history.append({'loss': avg_loss, 'accuracy': avg_acc})
            
            if epoch % 20 == 0:
                print(f'{label_type.title()} Labels - Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}')
        
        self.training_history[label_type] = history
        return self.model

# ============================================================================
# RQ1: SEMANTIC VS STATISTICAL DETECTION DECOMPOSITION
# ============================================================================

class SemanticStatisticalDecomposer:
    """Decompose OOD detection performance into semantic vs statistical components"""
    
    def __init__(self):
        self.results = {}
        
    def create_shift_datasets(self, base_dataset, shift_types=['semantic', 'covariate']):
        """Create different types of distribution shifts"""
        shift_datasets = {}
        
        # Semantic shift: completely different classes (e.g., CIFAR-10 vs SVHN)
        if 'semantic' in shift_types:
            semantic_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
            semantic_data = datasets.SVHN(root='./data', split='test', download=True, 
                                        transform=semantic_transform)
            shift_datasets['semantic'] = semantic_data
            
        # Covariate shift: same classes, different styles (corruptions, etc.)
        if 'covariate' in shift_types:
            covariate_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=3, sigma=1.0),  # Blur corruption
                transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Color shift
            ])
            # Apply to same dataset but with corruptions
            covariate_data = datasets.CIFAR10(root='./data', train=False, download=True, 
                                            transform=covariate_transform)
            shift_datasets['covariate'] = covariate_data
            
        return shift_datasets
    
    def analyze_representation_structure(self, model_real, model_random, test_loader):
        """Analyze how representations differ between real vs random label training"""
        def extract_features(model, loader):
            model.eval()
            features = []
            labels = []
            with torch.no_grad():
                for data, targets in loader:
                    data = data.cuda() if torch.cuda.is_available() else data
                    # Extract penultimate layer features
                    feat = model.penultimate(data) if hasattr(model, 'penultimate') else model(data)
                    features.append(feat.cpu())
                    labels.append(targets)
            return torch.cat(features), torch.cat(labels)
        
        # Extract features from both models
        feat_real, labels = extract_features(model_real, test_loader)
        feat_random, _ = extract_features(model_random, test_loader)
        
        # PCA analysis
        pca_real = PCA(n_components=50)
        pca_random = PCA(n_components=50)
        
        feat_real_pca = pca_real.fit_transform(feat_real.numpy())
        feat_random_pca = pca_random.fit_transform(feat_random.numpy())
        
        # Calculate explained variance ratios
        variance_real = pca_real.explained_variance_ratio_
        variance_random = pca_random.explained_variance_ratio_
        
        # Principal component alignment analysis
        alignment_score = np.mean([
            np.abs(np.dot(pca_real.components_[i], pca_random.components_[i])) 
            for i in range(min(10, len(pca_real.components_)))
        ])
        
        return {
            'variance_real': variance_real,
            'variance_random': variance_random,
            'pc_alignment': alignment_score,
            'features_real': feat_real_pca,
            'features_random': feat_random_pca,
            'labels': labels.numpy()
        }
    
    def decompose_ood_performance(self, ood_scores_real, ood_scores_random, 
                                 shift_type, true_labels):
        """Decompose OOD detection into semantic and statistical components"""
        # AUROC for real labels (semantic + statistical)
        auroc_real = roc_auc_score(true_labels, ood_scores_real)
        
        # AUROC for random labels (statistical only)
        auroc_random = roc_auc_score(true_labels, ood_scores_random)
        
        # Semantic component (difference)
        semantic_component = auroc_real - auroc_random
        statistical_component = auroc_random - 0.5  # Above random chance
        
        return {
            f'{shift_type}_auroc_real': auroc_real,
            f'{shift_type}_auroc_random': auroc_random,
            f'{shift_type}_semantic_component': semantic_component,
            f'{shift_type}_statistical_component': statistical_component,
            f'{shift_type}_semantic_ratio': semantic_component / (auroc_real - 0.5) if auroc_real > 0.5 else 0
        }

# ============================================================================
# RQ2: LABEL DEPENDENCY TEST ACROSS METHODS
# ============================================================================

class OODMethodTester:
    """Test various OOD detection methods on real vs random label models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.methods = {
            'msp': self.maximum_softmax_probability,
            'energy': self.energy_score,
            'mahalanobis': self.mahalanobis_distance,
            'odin': self.odin_score,
            'gradient_norm': self.gradient_norm_score
        }
        
    def maximum_softmax_probability(self, model, data):
        """MSP: Maximum Softmax Probability"""
        model.eval()
        with torch.no_grad():
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            scores = torch.max(probs, dim=1)[0]
        return scores.cpu().numpy()
    
    def energy_score(self, model, data, temperature=1.0):
        """Energy-based OOD detection"""
        model.eval()
        with torch.no_grad():
            logits = model(data)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy.cpu().numpy()
    
    def mahalanobis_distance(self, model, data, train_features=None, train_labels=None):
        """Mahalanobis distance in feature space"""
        model.eval()
        with torch.no_grad():
            features = model.penultimate(data) if hasattr(model, 'penultimate') else model(data)
            
            if train_features is None:
                # Simplified version - just use feature magnitude as proxy
                scores = -torch.norm(features, dim=1)
            else:
                # TODO: Implement proper Mahalanobis distance with class-conditional covariance
                scores = -torch.norm(features, dim=1)
                
        return scores.cpu().numpy()
    
    def odin_score(self, model, data, temperature=1000, epsilon=0.0014):
        """ODIN: Out-of-DIstribution detector for Neural networks"""
        model.eval()
        data.requires_grad_(True)
        
        # Forward pass
        logits = model(data)
        
        # Temperature scaling
        logits = logits / temperature
        
        # Input preprocessing (gradient-based)
        if epsilon > 0:
            loss = torch.mean(torch.logsumexp(logits, dim=1))
            loss.backward()
            
            # Add perturbation
            data_grad = data.grad.data
            sign_data_grad = data_grad.sign()
            perturbed_data = data - epsilon * sign_data_grad
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            
            # Re-forward with perturbed data
            logits = model(perturbed_data)
            logits = logits / temperature
        
        # Calculate confidence scores
        probs = F.softmax(logits, dim=1)
        scores = torch.max(probs, dim=1)[0]
        
        return scores.detach().cpu().numpy()
    
    def gradient_norm_score(self, model, data):
        """Gradient norm based OOD detection"""
        model.eval()
        data.requires_grad_(True)
        
        logits = model(data)
        # Use max logit for gradient computation
        max_logits = torch.max(logits, dim=1)[0]
        
        scores = []
        for i in range(len(max_logits)):
            if data.grad is not None:
                data.grad.zero_()
            max_logits[i].backward(retain_graph=True)
            grad_norm = torch.norm(data.grad[i]).item()
            scores.append(1.0 / (grad_norm + 1e-8))  # Inverse gradient norm
            
        return np.array(scores)
    
    def test_all_methods(self, model_real, model_random, id_loader, ood_loader):
        """Test all OOD methods on both real and random label models"""
        results = {}
        
        for method_name, method_func in self.methods.items():
            print(f"Testing {method_name}...")
            
            # Get ID scores
            id_scores_real = []
            id_scores_random = []
            
            for data, _ in id_loader:
                data = data.to(self.device)
                id_scores_real.extend(method_func(model_real, data))
                id_scores_random.extend(method_func(model_random, data))
            
            # Get OOD scores  
            ood_scores_real = []
            ood_scores_random = []
            
            for data, _ in ood_loader:
                data = data.to(self.device)
                ood_scores_real.extend(method_func(model_real, data))
                ood_scores_random.extend(method_func(model_random, data))
            
            # Calculate AUROC for both models
            # Create labels (0 for ID, 1 for OOD)
            labels = np.concatenate([
                np.zeros(len(id_scores_real)), 
                np.ones(len(ood_scores_real))
            ])
            
            # Combine scores (higher should indicate OOD)
            combined_scores_real = np.concatenate([-np.array(id_scores_real), -np.array(ood_scores_real)])
            combined_scores_random = np.concatenate([-np.array(id_scores_random), -np.array(ood_scores_random)])
            
            try:
                auroc_real = roc_auc_score(labels, combined_scores_real)
                auroc_random = roc_auc_score(labels, combined_scores_random)
                
                results[method_name] = {
                    'auroc_real': auroc_real,
                    'auroc_random': auroc_random,
                    'degradation': auroc_real - auroc_random,
                    'degradation_ratio': (auroc_real - auroc_random) / auroc_real if auroc_real > 0 else 0
                }
            except Exception as e:
                print(f"Error calculating AUROC for {method_name}: {e}")
                results[method_name] = {
                    'auroc_real': 0.5,
                    'auroc_random': 0.5,
                    'degradation': 0.0,
                    'degradation_ratio': 0.0
                }
        
        return results

# ============================================================================
# RQ3: TRUE SEMANTIC OOD DETECTION DESIGN
# ============================================================================

class SemanticOODDetector:
    """Design OOD detectors that specifically require semantic understanding"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def semantic_consistency_score(self, data, augmentations=None):
        """Score based on semantic consistency under transformations"""
        if augmentations is None:
            augmentations = [
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotation
                lambda x: torch.flip(x, dims=[3]),           # Horizontal flip
                lambda x: F.interpolate(x, scale_factor=0.8, mode='bilinear', align_corners=False)  # Scale
            ]
        
        self.model.eval()
        with torch.no_grad():
            # Original prediction
            orig_logits = self.model(data)
            orig_probs = F.softmax(orig_logits, dim=1)
            
            consistency_scores = []
            for aug_func in augmentations:
                aug_data = aug_func(data)
                # Resize back if needed
                if aug_data.shape != data.shape:
                    aug_data = F.interpolate(aug_data, size=data.shape[2:], mode='bilinear', align_corners=False)
                    
                aug_logits = self.model(aug_data)
                aug_probs = F.softmax(aug_logits, dim=1)
                
                # Calculate KL divergence between original and augmented predictions
                kl_div = F.kl_div(torch.log(aug_probs + 1e-8), orig_probs, reduction='none').sum(dim=1)
                consistency_scores.append(kl_div)
            
            # Lower KL divergence = higher semantic consistency = more likely ID
            avg_consistency = torch.stack(consistency_scores).mean(dim=0)
            
        return -avg_consistency.cpu().numpy()  # Negative because lower KL = higher ID likelihood
    
    def causal_intervention_score(self, data):
        """Score based on causal intervention in feature space"""
        self.model.eval()
        
        # Extract features
        if hasattr(self.model, 'penultimate'):
            features = self.model.penultimate(data)
        else:
            # Assume last layer before classifier
            features = self.model(data)
            
        # Perform random interventions on features
        intervention_scores = []
        num_interventions = 5
        
        for _ in range(num_interventions):
            # Randomly mask some features
            mask = torch.rand_like(features) > 0.1  # Keep 90% of features
            intervened_features = features * mask
            
            # Pass through final layers to get predictions
            if hasattr(self.model, 'classifier'):
                intervened_logits = self.model.classifier(intervened_features)
            else:
                # Assume features are already logits
                intervened_logits = intervened_features
                
            # Calculate prediction stability
            orig_pred = torch.argmax(self.model(data), dim=1)
            intervened_pred = torch.argmax(intervened_logits, dim=1)
            
            stability = (orig_pred == intervened_pred).float()
            intervention_scores.append(stability)
        
        # Higher stability = more robust semantic representation = more likely ID
        avg_stability = torch.stack(intervention_scores).mean(dim=0)
        return avg_stability.cpu().numpy()
    
    def multi_scale_semantic_score(self, data):
        """Combine multiple semantic-aware scoring functions"""
        consistency_score = self.semantic_consistency_score(data)
        causal_score = self.causal_intervention_score(data)
        
        # Combine scores (you can experiment with different combinations)
        combined_score = 0.7 * consistency_score + 0.3 * causal_score
        return combined_score

# ============================================================================
# RQ4: CAUSAL ANALYSIS OF OOD DETECTION MECHANISMS  
# ============================================================================

class CausalOODAnalyzer:
    """Analyze causal mechanisms behind OOD detection success/failure"""
    
    def __init__(self):
        self.causal_graph = {}
        
    def analyze_feature_importance(self, model_real, model_random, data_loader):
        """Analyze which features drive OOD detection in real vs random models"""
        
        def get_feature_attributions(model, data):
            """Get feature importance using integrated gradients approximation"""
            model.eval()
            data.requires_grad_(True)
            
            output = model(data)
            # Use max class for attribution
            max_class = torch.argmax(output, dim=1)
            
            attributions = []
            for i, class_idx in enumerate(max_class):
                if data.grad is not None:
                    data.grad.zero_()
                output[i, class_idx].backward(retain_graph=True)
                
                # Get input attributions
                attr = data.grad[i].abs().mean(dim=0)  # Average over channels
                attributions.append(attr.cpu().numpy())
                
            return np.array(attributions)
        
        real_attributions = []
        random_attributions = []
        
        for data, _ in data_loader:
            data = data.cuda() if torch.cuda.is_available() else data
            
            real_attr = get_feature_attributions(model_real, data)
            random_attr = get_feature_attributions(model_random, data)
            
            real_attributions.append(real_attr)
            random_attributions.append(random_attr)
            
            if len(real_attributions) > 10:  # Limit for efficiency
                break
        
        real_attributions = np.concatenate(real_attributions)
        random_attributions = np.concatenate(random_attributions)
        
        return {
            'real_mean_attribution': real_attributions.mean(axis=0),
            'random_mean_attribution': random_attributions.mean(axis=0),
            'attribution_difference': real_attributions.mean(axis=0) - random_attributions.mean(axis=0),
            'attribution_correlation': np.corrcoef(
                real_attributions.flatten(), 
                random_attributions.flatten()
            )[0, 1]
        }
    
    def mediator_analysis(self, model, id_data, ood_data):
        """Analyze what mediates the relationship between input and OOD score"""
        # Extract intermediate representations
        def hook_features(name, features_dict):
            def hook(model, input, output):
                features_dict[name] = output.detach()
            return hook
        
        features_dict = {}
        hooks = []
        
        # Register hooks for intermediate layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and 'classifier' not in name:
                hook = module.register_forward_hook(hook_features(name, features_dict))
                hooks.append(hook)
        
        # Get features for ID and OOD data
        model.eval()
        with torch.no_grad():
            _ = model(id_data[:10])  # Small batch for efficiency
            id_features = copy.deepcopy(features_dict)
            
            _ = model(ood_data[:10])
            ood_features = copy.deepcopy(features_dict)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze which layers are most discriminative
        layer_discriminability = {}
        for layer_name in id_features.keys():
            id_feat = id_features[layer_name].view(id_features[layer_name].size(0), -1)
            ood_feat = ood_features[layer_name].view(ood_features[layer_name].size(0), -1)
            
            # Simple discriminability measure: mean distance between ID and OOD
            id_center = id_feat.mean(dim=0)
            ood_center = ood_feat.mean(dim=0)
            discriminability = torch.norm(id_center - ood_center).item()
            
            layer_discriminability[layer_name] = discriminability
        
        return layer_discriminability
    
    def counterfactual_analysis(self, model_real, model_random, test_data):
        """Counterfactual: What would happen if we swap model components?"""
        
        # Create hybrid model: early layers from one, late layers from another
        def create_hybrid_model(early_model, late_model, split_layer='layer3'):
            hybrid = copy.deepcopy(early_model)
            
            # Replace later layers
            for name, module in late_model.named_children():
                if split_layer in name or 'fc' in name or 'classifier' in name:
                    setattr(hybrid, name, module)
            
            return hybrid
        
        # Create hybrid models
        hybrid_real_early = create_hybrid_model(model_real, model_random)
        hybrid_random_early = create_hybrid_model(model_random, model_real)
        
        models = {
            'real': model_real,
            'random': model_random,
            'hybrid_real_early': hybrid_real_early,
            'hybrid_random_early': hybrid_random_early
        }
        
        # Test OOD detection performance of hybrid models
        results = {}
        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                outputs = model(test_data)
                energy_scores = -torch.logsumexp(outputs, dim=1)
                results[model_name] = energy_scores.cpu().numpy()
        
        return results

# ============================================================================
# MAIN EXPERIMENTAL RUNNER
# ============================================================================

class ExperimentRunner:
    """Main class to run all experiments"""
    
    def __init__(self, model_class, dataset_name='CIFAR10'):
        self.model_class = model_class
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
    def setup_data(self, batch_size=128):
        """Setup data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if self.dataset_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def run_all_experiments(self, epochs=50):
        """Run all research questions"""
        print("Setting up data...")
        train_loader, test_loader = self.setup_data()
        
        print("Training models with real and random labels...")
        
        # Train model with real labels
        model_real = self.model_class()
        trainer_real = RandomLabelTrainer(model_real, self.device)
        model_real = trainer_real.train_with_label_type(train_loader, 'real', epochs)
        
        # Train model with random labels  
        model_random = self.model_class()
        trainer_random = RandomLabelTrainer(model_random, self.device)
        model_random = trainer_random.train_with_label_type(train_loader, 'random', epochs)
        
        print("\n" + "="*80)
        print("RQ1: SEMANTIC VS STATISTICAL DECOMPOSITION")
        print("="*80)
        
        decomposer = SemanticStatisticalDecomposer()
        
        # Analyze representation structure
        repr_analysis = decomposer.analyze_representation_structure(
            model_real, model_random, test_loader
        )
        self.results['rq1_representation'] = repr_analysis
        print(f"Principal component alignment: {repr_analysis['pc_alignment']:.4f}")
        
        # Create shift datasets and analyze
        shift_datasets = decomposer.create_shift_datasets(test_loader.dataset)
        
        for shift_type, shift_data in shift_datasets.items():
            shift_loader = DataLoader(shift_data, batch_size=128, shuffle=False)
            
            # Get OOD scores using energy method
            energy_real = []
            energy_random = []
            
            for data, _ in test_loader:
                data = data.to(self.device)
                # ID samples
                with torch.no_grad():
                    logits_real = model_real(data)
                    logits_random = model_random(data)
                    energy_real.extend((-torch.logsumexp(logits_real, dim=1)).cpu().numpy())
                    energy_random.extend((-torch.logsumexp(logits_random, dim=1)).cpu().numpy())
            
            energy_ood_real = []
            energy_ood_random = []
            
            for data, _ in shift_loader:
                data = data.to(self.device)
                # OOD samples
                with torch.no_grad():
                    logits_real = model_real(data)
                    logits_random = model_random(data)
                    energy_ood_real.extend((-torch.logsumexp(logits_real, dim=1)).cpu().numpy())
                    energy_ood_random.extend((-torch.logsumexp(logits_random, dim=1)).cpu().numpy())
                    
                if len(energy_ood_real) > len(energy_real):  # Match sizes
                    break
            
            # Create labels and scores
            labels = np.concatenate([np.zeros(len(energy_real)), np.ones(len(energy_ood_real))])
            scores_real = np.concatenate([energy_real, energy_ood_real])
            scores_random = np.concatenate([energy_random, energy_ood_random])
            
            decomp_results = decomposer.decompose_ood_performance(
                scores_real, scores_random, shift_type, labels
            )
            self.results[f'rq1_{shift_type}'] = decomp_results
            
            print(f"\n{shift_type.title()} Shift Results:")
            print(f"  Real labels AUROC: {decomp_results[f'{shift_type}_auroc_real']:.4f}")
            print(f"  Random labels AUROC: {decomp_results[f'{shift_type}_auroc_random']:.4f}")
            print(f"  Semantic component: {decomp_results[f'{shift_type}_semantic_component']:.4f}")
            print(f"  Semantic ratio: {decomp_results[f'{shift_type}_semantic_ratio']:.4f}")
        
        print("\n" + "="*80)
        print("RQ2: LABEL DEPENDENCY TEST")
        print("="*80)
        
        method_tester = OODMethodTester(self.device)
        
        # Use semantic shift data for testing
        if 'semantic' in shift_datasets:
            ood_loader = DataLoader(shift_datasets['semantic'], batch_size=128, shuffle=False)
            method_results = method_tester.test_all_methods(
                model_real, model_random, test_loader, ood_loader
            )
            self.results['rq2_methods'] = method_results
            
            print("\nMethod Comparison (Real vs Random Labels):")
            print("Method           | Real AUROC | Random AUROC | Degradation | Deg. Ratio")
            print("-" * 70)
            for method, results in method_results.items():
                print(f"{method:15s} | {results['auroc_real']:8.4f} | {results['auroc_random']:10.4f} | "
                      f"{results['degradation']:9.4f} | {results['degradation_ratio']:8.4f}")
        
        print("\n" + "="*80)
        print("RQ3: SEMANTIC OOD DETECTOR DESIGN")
        print("="*80)
        
        semantic_detector = SemanticOODDetector(model_real, self.device)
        
        # Test semantic detector on both models
        semantic_scores_real = []
        semantic_scores_random = []
        
        for data, _ in test_loader:
            data = data.to(self.device)
            # Test on real model
            semantic_detector.model = model_real
            scores_real = semantic_detector.multi_scale_semantic_score(data)
            semantic_scores_real.extend(scores_real)
            
            # Test on random model
            semantic_detector.model = model_random
            scores_random = semantic_detector.multi_scale_semantic_score(data)
            semantic_scores_random.extend(scores_random)
            
            if len(semantic_scores_real) > 500:  # Limit for efficiency
                break
        
        print(f"Semantic detector mean score (real model): {np.mean(semantic_scores_real):.4f}")
        print(f"Semantic detector mean score (random model): {np.mean(semantic_scores_random):.4f}")
        print(f"Score difference: {np.mean(semantic_scores_real) - np.mean(semantic_scores_random):.4f}")
        
        self.results['rq3_semantic'] = {
            'scores_real': semantic_scores_real,
            'scores_random': semantic_scores_random
        }
        
        print("\n" + "="*80)
        print("RQ4: CAUSAL ANALYSIS")
        print("="*80)
        
        causal_analyzer = CausalOODAnalyzer()
        
        # Feature importance analysis
        attribution_results = causal_analyzer.analyze_feature_importance(
            model_real, model_random, test_loader
        )
        self.results['rq4_attributions'] = attribution_results
        print(f"Feature attribution correlation: {attribution_results['attribution_correlation']:.4f}")
        
        # Mediator analysis
        id_batch = next(iter(test_loader))[0].to(self.device)
        if 'semantic' in shift_datasets:
            ood_batch = next(iter(DataLoader(shift_datasets['semantic'], batch_size=128)))[0].to(self.device)
            mediator_results = causal_analyzer.mediator_analysis(model_real, id_batch, ood_batch)
            self.results['rq4_mediators'] = mediator_results
            
            print("\nLayer discriminability:")
            for layer, score in sorted(mediator_results.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {layer}: {score:.4f}")
        
        # Counterfactual analysis
        counterfactual_results = causal_analyzer.counterfactual_analysis(
            model_real, model_random, id_batch
        )
        self.results['rq4_counterfactual'] = counterfactual_results
        
        print("\nCounterfactual analysis complete.")
        
        return self.results
    
    def plot_results(self):
        """Create visualizations of key results"""
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Semantic vs Statistical decomposition
        if 'rq1_semantic' in self.results and 'rq1_covariate' in self.results:
            semantic_results = self.results['rq1_semantic']
            covariate_results = self.results['rq1_covariate']
            
            shifts = ['Semantic', 'Covariate'] 
            real_aurocs = [semantic_results['semantic_auroc_real'], covariate_results['covariate_auroc_real']]
            random_aurocs = [semantic_results['semantic_auroc_random'], covariate_results['covariate_auroc_random']]
            
            x = np.arange(len(shifts))
            width = 0.35
            
            axes[0,0].bar(x - width/2, real_aurocs, width, label='Real Labels', alpha=0.8)
            axes[0,0].bar(x + width/2, random_aurocs, width, label='Random Labels', alpha=0.8)
            axes[0,0].set_ylabel('AUROC')
            axes[0,0].set_title('RQ1: AUROC by Shift Type and Label Type')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(shifts)
            axes[0,0].legend()
            axes[0,0].grid(alpha=0.3)
        
        # Plot 2: Method degradation
        if 'rq2_methods' in self.results:
            methods = list(self.results['rq2_methods'].keys())
            degradations = [self.results['rq2_methods'][m]['degradation'] for m in methods]
            
            axes[0,1].barh(methods, degradations, alpha=0.8, color='orange')
            axes[0,1].set_xlabel('AUROC Degradation (Real - Random)')
            axes[0,1].set_title('RQ2: Method Degradation with Random Labels')
            axes[0,1].grid(alpha=0.3)
        
        # Plot 3: Representation analysis
        if 'rq1_representation' in self.results:
            repr_results = self.results['rq1_representation']
            
            # Plot explained variance
            variance_real = repr_results['variance_real'][:10]
            variance_random = repr_results['variance_random'][:10]
            
            x = np.arange(len(variance_real))
            axes[1,0].plot(x, variance_real, 'o-', label='Real Labels', alpha=0.8)
            axes[1,0].plot(x, variance_random, 's-', label='Random Labels', alpha=0.8)
            axes[1,0].set_xlabel('Principal Component')
            axes[1,0].set_ylabel('Explained Variance Ratio')
            axes[1,0].set_title('RQ1: PCA Explained Variance')
            axes[1,0].legend()
            axes[1,0].grid(alpha=0.3)
        
        # Plot 4: Semantic detector comparison
        if 'rq3_semantic' in self.results:
            scores_real = self.results['rq3_semantic']['scores_real']
            scores_random = self.results['rq3_semantic']['scores_random']
            
            axes[1,1].hist(scores_real, bins=30, alpha=0.7, label='Real Labels', density=True)
            axes[1,1].hist(scores_random, bins=30, alpha=0.7, label='Random Labels', density=True)
            axes[1,1].set_xlabel('Semantic OOD Score')
            axes[1,1].set_ylabel('Density')
            axes[1,1].set_title('RQ3: Semantic Detector Score Distributions')
            axes[1,1].legend()
            axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ood_random_labels_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Define a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, num_classes)
            
        def penultimate(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            return x.view(x.size(0), -1)
            
        def forward(self, x):
            features = self.penultimate(x)
            return self.fc(features)
    
    # Run experiments
    runner = ExperimentRunner(SimpleModel, 'CIFAR10')
    results = runner.run_all_experiments(epochs=10)  # Reduced for demo
    
    # Plot results
    runner.plot_results()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("Key insights:")
    print("1. Check how much OOD detection relies on semantic vs statistical patterns")
    print("2. Identify which methods are most dependent on meaningful labels")
    print("3. Test novel semantic-aware OOD detection approaches")
    print("4. Understand causal mechanisms behind OOD detection")