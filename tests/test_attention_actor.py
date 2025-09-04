"""
Tests for the Attention Actor Network
"""

import pytest
import torch
import numpy as np
from src.nets.attention_actor import AttentionActor, MultiHeadAttention, MarketRegimeClassifier


class TestMultiHeadAttention:
    """Test the multi-head attention mechanism."""
    
    def test_attention_forward(self):
        """Test forward pass of multi-head attention."""
        d_model = 64
        n_heads = 8
        seq_len = 4
        batch_size = 2
        
        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = attention(x)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        d_model = 64
        n_heads = 8
        seq_len = 4
        batch_size = 2
        
        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -1] = 0  # Mask last position
        
        output, attn_weights = attention(x, mask)
        
        assert output.shape == x.shape
        # Check that masked positions have zero attention
        assert torch.allclose(attn_weights[:, :, :, -1], torch.zeros_like(attn_weights[:, :, :, -1]))


class TestMarketRegimeClassifier:
    """Test the market regime classifier."""
    
    def test_regime_classifier_forward(self):
        """Test forward pass of regime classifier."""
        input_dim = 128
        hidden_dim = 64
        batch_size = 4
        
        classifier = MarketRegimeClassifier(input_dim, hidden_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = classifier(x)
        
        assert 'regime_logits' in output
        assert 'regime_probs' in output
        assert 'regime_pred' in output
        
        assert output['regime_logits'].shape == (batch_size, 4)
        assert output['regime_probs'].shape == (batch_size, 4)
        assert output['regime_pred'].shape == (batch_size,)
        
        # Check probabilities sum to 1
        assert torch.allclose(output['regime_probs'].sum(dim=-1), torch.ones(batch_size))
        
        # Check predictions are valid
        assert torch.all(output['regime_pred'] >= 0)
        assert torch.all(output['regime_pred'] < 4)


class TestAttentionActor:
    """Test the attention-based actor network."""
    
    def test_actor_forward(self):
        """Test forward pass of attention actor."""
        obs_dim = 50
        n_assets = 5
        batch_size = 3
        
        actor = AttentionActor(obs_dim, n_assets)
        obs = torch.randn(batch_size, obs_dim)
        
        output = actor(obs)
        
        # Check output structure
        required_keys = ['weights', 'logits', 'regime_info', 'regime_logits', 'attention_weights', 'final_representation']
        for key in required_keys:
            assert key in output, f"Missing key: {key}"
        
        # Check shapes
        assert output['weights'].shape == (batch_size, n_assets)
        assert output['logits'].shape == (batch_size, n_assets)
        assert output['final_representation'].shape == (batch_size, 256)  # hidden_dim
        
        # Check weights sum to 1
        assert torch.allclose(output['weights'].sum(dim=-1), torch.ones(batch_size), atol=1e-6)
        
        # Check weights are non-negative
        assert torch.all(output['weights'] >= 0)
        
        # Check regime info
        regime_info = output['regime_info']
        assert regime_info['regime_logits'].shape == (batch_size, 4)
        assert regime_info['regime_probs'].shape == (batch_size, 4)
        assert regime_info['regime_pred'].shape == (batch_size,)
    
    def test_actor_different_batch_sizes(self):
        """Test actor with different batch sizes."""
        obs_dim = 50
        n_assets = 5
        actor = AttentionActor(obs_dim, n_assets)
        
        for batch_size in [1, 2, 8, 16]:
            obs = torch.randn(batch_size, obs_dim)
            output = actor(obs)
            
            assert output['weights'].shape == (batch_size, n_assets)
            assert torch.allclose(output['weights'].sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    
    def test_actor_temperature_scaling(self):
        """Test temperature scaling in actor."""
        obs_dim = 50
        n_assets = 5
        batch_size = 2
        
        # Test with different temperatures
        for temperature in [0.5, 1.0, 2.0]:
            actor = AttentionActor(obs_dim, n_assets, temperature=temperature)
            obs = torch.randn(batch_size, obs_dim)
            
            output = actor(obs)
            
            # Higher temperature should lead to more uniform weights
            if temperature > 1.0:
                # Check that weights are more uniform (higher entropy)
                weights = output['weights']
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
                # This is a heuristic check - higher temperature should generally lead to higher entropy
                assert torch.all(entropy > 0.5)  # Reasonable entropy threshold
    
    def test_actor_regime_interpretation(self):
        """Test regime interpretation functionality."""
        obs_dim = 50
        n_assets = 5
        actor = AttentionActor(obs_dim, n_assets)
        
        # Test regime interpretation
        regime_pred = torch.tensor([0, 1, 2, 3])
        interpretations = [actor.get_regime_interpretation(pred) for pred in regime_pred]
        
        expected_regimes = ['Bull Market', 'Bear Market', 'Volatile Market', 'Sideways Market']
        assert interpretations == expected_regimes
    
    def test_actor_attention_interpretation(self):
        """Test attention interpretation functionality."""
        obs_dim = 50
        n_assets = 5
        actor = AttentionActor(obs_dim, n_assets)
        
        obs = torch.randn(1, obs_dim)
        output = actor(obs)
        
        interpretation = actor.get_attention_interpretation(output['attention_weights'])
        
        assert 'feature_attention' in interpretation
        assert 'most_attended_feature' in interpretation
        assert 'attention_entropy' in interpretation
        
        # Check feature attention sums to 1
        feature_attention = interpretation['feature_attention']
        total_attention = sum(feature_attention.values())
        assert abs(total_attention - 1.0) < 1e-6
    
    def test_actor_gradient_flow(self):
        """Test that gradients flow properly through the actor."""
        obs_dim = 50
        n_assets = 5
        actor = AttentionActor(obs_dim, n_assets)
        
        obs = torch.randn(2, obs_dim, requires_grad=True)
        output = actor(obs)
        
        # Compute a dummy loss
        loss = output['weights'].sum()
        loss.backward()
        
        # Check that gradients exist
        assert obs.grad is not None
        assert not torch.allclose(obs.grad, torch.zeros_like(obs.grad))
        
        # Check that all parameters have gradients
        for param in actor.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_actor_deterministic_behavior(self):
        """Test that actor produces deterministic outputs with same input."""
        obs_dim = 50
        n_assets = 5
        actor = AttentionActor(obs_dim, n_assets)
        actor.eval()  # Set to evaluation mode
        
        obs = torch.randn(1, obs_dim)
        
        # Forward pass twice
        output1 = actor(obs)
        output2 = actor(obs)
        
        # Should be identical in eval mode
        assert torch.allclose(output1['weights'], output2['weights'])
        assert torch.allclose(output1['logits'], output2['logits'])
    
    def test_actor_feature_groups_creation(self):
        """Test the feature groups creation method."""
        obs_dim = 50
        n_assets = 5
        actor = AttentionActor(obs_dim, n_assets)
        
        batch_size = 2
        embedded = torch.randn(batch_size, 256)  # hidden_dim
        
        feature_groups = actor._create_feature_groups(embedded)
        
        # Should create 4 feature groups
        assert feature_groups.shape == (batch_size, 4, 256)
        
        # Each group should have the same size
        for i in range(4):
            assert feature_groups[:, i, :].shape == (batch_size, 256)


if __name__ == "__main__":
    pytest.main([__file__])
