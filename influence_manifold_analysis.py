import torch

class InfluenceTracker:
    def __init__(self, model, device='mps'):
        self.model = model
        self.device = device
        self.influence_cache = {}
        self.boundary_influence_scores = {}
        self.feature_value_scores = {}
        self.discriminative_power = {}
    
    def compute_boundary_influence_scores(self, features, labels, boundary_analyzer):
        """Identify samples that most influence decision boundaries"""
        # Step 1: Extract boundary-critical directions from your existing analyzer
        directional_expansion = boundary_analyzer._compute_directional_expansion(features)
        
        # Step 2: Compute sample influence on these directions
        influence_scores = self._compute_directional_influence(features, directional_expansion)
        
        # Step 3: Weight by boundary proximity
        boundary_distances = self._compute_boundary_distances(features, labels)
        
        # Step 4: Combined influence score
        combined_scores = influence_scores * (1.0 / (boundary_distances + 1e-6))
        
        return combined_scores
    
    def _compute_directional_influence(self, features, principal_directions):
        """How much each sample influences the principal directions"""
        # Project features onto principal directions
        projections = torch.matmul(features, principal_directions)
        
        # Compute influence as variance contribution along each direction
        influence_per_direction = torch.var(projections, dim=0)
        
        # Aggregate influence across top-k directions
        return torch.mean(influence_per_direction[:5])  # Top 5 directions

    def compute_feature_discrimination_power(self, id_features, labels):
        """Extend your existing geometry analysis with discrimination power"""
        geometry = self.analyze_feature_geometry(id_features)
        
        # Step 1: Inter-class vs intra-class variance ratio
        discrimination_scores = self._compute_fisher_discriminant_ratio(id_features, labels)
        
        # Step 2: Boundary-relevant feature ranking
        boundary_relevance = self._rank_features_by_boundary_relevance(
            geometry['eigenvecs'], geometry['eigenvals']
        )
        
        # Step 3: Combined feature value score
        feature_values = discrimination_scores * boundary_relevance
        
        return {
            **geometry,
            'feature_values': feature_values,
            'discrimination_scores': discrimination_scores,
            'boundary_relevance': boundary_relevance
        }
    
    def _compute_fisher_discriminant_ratio(self, features, labels):
        """Compute between-class vs within-class variance ratio"""
        # Between-class variance
        class_means = {}
        overall_mean = torch.mean(features, dim=0)
        
        for class_id in torch.unique(labels):
            mask = (labels == class_id)
            class_means[class_id.item()] = torch.mean(features[mask], dim=0)
        
        between_class_var = torch.zeros(features.shape[1], device=features.device)
        within_class_var = torch.zeros(features.shape[1], device=features.device)
        
        for class_id, class_mean in class_means.items():
            mask = (labels == class_id)
            n_class = mask.sum()
            
            # Between-class variance
            between_class_var += n_class * (class_mean - overall_mean) ** 2
            
            # Within-class variance
            class_features = features[mask]
            within_class_var += torch.sum((class_features - class_mean) ** 2, dim=0)
        
        # Fisher discriminant ratio
        fisher_ratio = between_class_var / (within_class_var + 1e-8)
        return fisher_ratio

    # Placeholder for methods that may be required from ManifoldAnalyzer
    def analyze_feature_geometry(self, features):
        # Dummy implementation, replace with actual geometry analysis
        # Should return a dict with keys 'eigenvecs' and 'eigenvals'
        return {'eigenvecs': torch.eye(features.shape[1]), 'eigenvals': torch.ones(features.shape[1])}

    def _rank_features_by_boundary_relevance(self, eigenvecs, eigenvals):
        # Dummy implementation, replace with actual ranking logic
        return eigenvals

    def _compute_boundary_distances(self, features, labels):
        # Dummy implementation, replace with actual boundary distance computation
        return torch.ones(features.shape[0], device=features.device) 