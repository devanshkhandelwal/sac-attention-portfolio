"""
Attention-based Actor Network with Market Regime Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for market feature analysis."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + output)
        
        return output, attention_weights


class MarketRegimeClassifier(nn.Module):
    """Classifies market regimes: Bull, Bear, Volatile, Sideways."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.regime_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4)  # 4 regimes
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.regime_net(x)
        regime_probs = F.softmax(logits, dim=-1)
        regime_pred = torch.argmax(regime_probs, dim=-1)
        
        return {
            'regime_logits': logits,
            'regime_probs': regime_probs,
            'regime_pred': regime_pred
        }


class AttentionActor(nn.Module):
    """
    Advanced Actor Network with Multi-Head Attention and Regime Detection.
    
    Architecture:
    1. Feature embedding layer
    2. Multi-head self-attention for market feature analysis
    3. Regime classification head
    4. Regime-aware policy head
    5. Portfolio weight generation via softmax
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_assets: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Regime classification
        self.regime_classifier = MarketRegimeClassifier(hidden_dim)
        
        # Regime-aware policy heads
        self.regime_policies = nn.ModuleDict({
            'bull': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_assets)
            ),
            'bear': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_assets)
            ),
            'volatile': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_assets)
            ),
            'sideways': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_assets)
            )
        })
        
        # Global policy head (fallback)
        self.global_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_assets)
        )
        
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through attention-based actor.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            
        Returns:
            Dictionary containing:
            - weights: Portfolio weights [batch_size, n_assets]
            - logits: Raw logits before softmax
            - regime_info: Regime classification results
            - attention_weights: Attention weights for interpretability
        """
        batch_size = obs.size(0)
        
        # Feature embedding
        embedded = self.feature_embedding(obs)  # [batch_size, hidden_dim]
        
        # Reshape for attention (treat features as sequence)
        # Create a sequence by treating each feature group as a token
        feature_groups = self._create_feature_groups(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Apply multi-head attention
        attention_output = feature_groups
        attention_weights_list = []
        
        for attention_layer in self.attention_layers:
            attention_output, attn_weights = attention_layer(attention_output)
            attention_weights_list.append(attn_weights)
        
        # Global average pooling to get final representation
        final_representation = attention_output.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Regime classification
        regime_info = self.regime_classifier(final_representation)
        
        # Regime-aware policy generation
        regime_pred = regime_info['regime_pred']
        regime_probs = regime_info['regime_probs']
        
        # Generate logits for each regime
        regime_logits = {}
        for i, regime in enumerate(['bull', 'bear', 'volatile', 'sideways']):
            regime_logits[regime] = self.regime_policies[regime](final_representation)
        
        # Weighted combination of regime-specific policies
        combined_logits = torch.zeros_like(regime_logits['bull'])
        for i, regime in enumerate(['bull', 'bear', 'volatile', 'sideways']):
            regime_weight = regime_probs[:, i:i+1]  # [batch_size, 1]
            combined_logits += regime_weight * regime_logits[regime]
        
        # Add global policy as regularization
        global_logits = self.global_policy(final_representation)
        combined_logits = 0.8 * combined_logits + 0.2 * global_logits
        
        # Apply temperature scaling
        scaled_logits = combined_logits / self.temperature
        
        # Generate portfolio weights via softmax
        weights = F.softmax(scaled_logits, dim=-1)
        
        return {
            'weights': weights,
            'logits': scaled_logits,
            'regime_info': regime_info,
            'regime_logits': regime_logits,
            'attention_weights': attention_weights_list,
            'final_representation': final_representation
        }
    
    def _create_feature_groups(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Create feature groups for attention mechanism.
        Groups related features together (e.g., price features, technical indicators, etc.)
        """
        batch_size, hidden_dim = embedded.size()
        
        # Create 4 feature groups by splitting the hidden representation
        # Ensure each group has the same size as the hidden_dim for attention
        group_size = hidden_dim // 4
        remainder = hidden_dim % 4
        
        # Split into 4 groups, padding the last group if necessary
        groups = []
        for i in range(4):
            start_idx = i * group_size
            if i == 3:  # Last group gets any remainder
                end_idx = start_idx + group_size + remainder
            else:
                end_idx = start_idx + group_size
            
            group = embedded[:, start_idx:end_idx]
            
            # Pad to match hidden_dim
            if group.size(1) < hidden_dim:
                padding = torch.zeros(batch_size, hidden_dim - group.size(1), device=embedded.device)
                group = torch.cat([group, padding], dim=1)
            
            groups.append(group)
        
        # Stack groups to create sequence dimension
        feature_groups = torch.stack(groups, dim=1)  # [batch_size, 4, hidden_dim]
        
        return feature_groups
    
    def get_regime_interpretation(self, regime_pred: torch.Tensor) -> str:
        """Convert regime prediction to human-readable format."""
        regime_names = ['Bull Market', 'Bear Market', 'Volatile Market', 'Sideways Market']
        return regime_names[regime_pred.item()]
    
    def get_attention_interpretation(self, attention_weights: list, feature_names: list = None) -> Dict:
        """Interpret attention weights for explainability."""
        if feature_names is None:
            feature_names = [f'Feature_Group_{i}' for i in range(4)]
        
        # Average attention weights across heads and layers
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1))  # [batch_size, seq_len, seq_len]
        
        # Get attention to each feature group
        attention_scores = avg_attention.mean(dim=-1)  # [batch_size, seq_len]
        
        interpretation = {
            'feature_attention': {
                name: score.item() for name, score in zip(feature_names, attention_scores[0])
            },
            'most_attended_feature': feature_names[attention_scores[0].argmax().item()],
            'attention_entropy': -torch.sum(attention_scores[0] * torch.log(attention_scores[0] + 1e-8)).item()
        }
        
        return interpretation


def test_attention_actor():
    """Test the attention actor implementation."""
    obs_dim = 50
    n_assets = 5
    batch_size = 32
    
    actor = AttentionActor(obs_dim, n_assets)
    obs = torch.randn(batch_size, obs_dim)
    
    output = actor(obs)
    
    print("Attention Actor Test Results:")
    print(f"Input shape: {obs.shape}")
    print(f"Weights shape: {output['weights'].shape}")
    print(f"Regime prediction: {actor.get_regime_interpretation(output['regime_info']['regime_pred'][0])}")
    print(f"Attention interpretation: {actor.get_attention_interpretation(output['attention_weights'])}")
    
    # Verify weights sum to 1
    weight_sums = output['weights'].sum(dim=-1)
    print(f"Weight sums (should be ~1.0): {weight_sums.mean().item():.4f} ± {weight_sums.std().item():.4f}")


if __name__ == "__main__":
    test_attention_actor()
