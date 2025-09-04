"""
Enhanced SAC Agent with Attention-Based Actor and Regime Awareness
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.sac import SACAgent, SACConfig, ReplayBuffer
from src.nets.attention_actor import AttentionActor
from src.nets.actor_critic import TwinCritic


class AttentionSACAgent(SACAgent):
    """
    Enhanced SAC Agent with Attention-Based Actor and Regime Awareness.
    
    Features:
    - Multi-head attention for market feature analysis
    - Regime detection and regime-aware policies
    - Enhanced exploration with regime-specific noise
    - Interpretable attention weights
    - Regime performance tracking
    """
    
    def __init__(self, config: SACConfig):
        super().__init__(config)
        
        # Replace standard actor with attention-based actor
        self.actor = AttentionActor(
            obs_dim=config.obs_dim,
            n_assets=config.n_assets,
            hidden_dim=config.actor_hidden[0],
            n_heads=8,
            n_layers=2,
            dropout=0.1,
            temperature=1.0
        )
        
        # Initialize attention actor
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=config.lr
        )
        
        # Regime tracking
        self.regime_history = []
        self.regime_performance = {}
        self.attention_history = []
        
        # Enhanced exploration
        self.regime_exploration_noise = {
            'bull': 0.15,      # Higher exploration in bull markets
            'bear': 0.25,      # Even higher in bear markets
            'volatile': 0.3,   # Highest in volatile markets
            'sideways': 0.1    # Lower in sideways markets
        }
    
    def select_action(
        self, 
        obs: np.ndarray, 
        greedy: bool = False,
        return_attention: bool = False
    ) -> np.ndarray:
        """
        Select action using attention-based actor.
        
        Args:
            obs: Observation array
            greedy: Whether to use greedy action selection
            return_attention: Whether to return attention weights
            
        Returns:
            Action array (portfolio weights)
        """
        self.actor.eval()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            actor_output = self.actor(obs_tensor)
            
            weights = actor_output['weights'].squeeze(0).numpy()
            regime_info = actor_output['regime_info']
            attention_weights = actor_output['attention_weights']
            
            # Store regime and attention information
            regime_pred = regime_info['regime_pred'].item()
            regime_names = ['bull', 'bear', 'volatile', 'sideways']
            current_regime = regime_names[regime_pred]
            
            self.regime_history.append(current_regime)
            if return_attention:
                self.attention_history.append(attention_weights)
            
            # Add regime-specific exploration noise
            if not greedy and self.cfg.exploration_noise > 0:
                regime_noise = self.regime_exploration_noise.get(current_regime, 0.1)
                noise_scale = self.cfg.exploration_noise * regime_noise
                
                # Add noise to logits before softmax
                logits = actor_output['logits'].squeeze(0).numpy()
                noise = np.random.normal(0, noise_scale, logits.shape)
                noisy_logits = logits + noise
                
                # Convert back to weights
                weights = self._logits_to_weights(noisy_logits)
            
            return weights
    
    def _logits_to_weights(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to valid portfolio weights."""
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / np.sum(exp_logits)
        
        # Ensure non-negative and sum to 1
        weights = np.maximum(weights, 0)
        weights = weights / (np.sum(weights) + 1e-12)
        
        return weights
    
    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update agent with enhanced attention-based learning.
        
        Returns:
            Dictionary of loss values and metrics
        """
        if len(self.replay) < self.cfg.batch_size:
            return {}
        
        batch_size = batch_size or self.cfg.batch_size
        batch = self.replay.sample(batch_size)
        
        obs, actions, rewards, next_obs, dones = batch
        
        # Convert to tensors
        obs = torch.FloatTensor(obs)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # Update critics
        critic_loss = self._update_critics(obs, actions, rewards, next_obs, dones)
        
        # Update actor with attention
        actor_loss, attention_metrics = self._update_attention_actor(obs)
        
        # Update alpha
        alpha_loss = self._update_alpha(obs)
        
        # Soft update target networks
        self._soft_update_targets()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item(),
            **attention_metrics
        }
    
    def _update_critics(self, obs, actions, rewards, next_obs, dones):
        """Update critic networks."""
        # Critic target
        with torch.no_grad():
            a2 = self.actor(next_obs)['weights']
            q1_t, q2_t = self.critic_tgt(next_obs, a2)
            q_min = torch.min(q1_t, q2_t)
            v = q_min + self.alpha * (-torch.sum(a2 * torch.log(a2 + 1e-12), dim=-1))
            y = rewards + (1.0 - dones.float()) * self.cfg.gamma * v

        # Critic loss
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        
        return critic_loss.item()
    
    def _update_attention_actor(self, obs: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Update attention-based actor."""
        self.actor.train()
        
        # Get actor output with attention
        actor_output = self.actor(obs)
        weights = actor_output['weights']
        logits = actor_output['logits']
        regime_info = actor_output['regime_info']
        attention_weights = actor_output['attention_weights']
        
        # Compute Q-values
        q1, q2 = self.critic(obs, weights)
        q_min = torch.min(q1, q2)
        
        # Compute entropy
        log_probs = torch.log(weights + 1e-8)
        entropy = -(weights * log_probs).sum(dim=-1, keepdim=True)
        
        # Actor loss (SAC objective)
        actor_loss = -(q_min - self.alpha * entropy).mean()
        
        # Add attention regularization
        attention_reg = self._compute_attention_regularization(attention_weights)
        actor_loss += 0.01 * attention_reg
        
        # Add regime consistency loss
        regime_consistency_loss = self._compute_regime_consistency_loss(regime_info)
        actor_loss += 0.005 * regime_consistency_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Compute attention metrics
        attention_metrics = self._compute_attention_metrics(attention_weights, regime_info)
        
        return actor_loss.item(), attention_metrics
    
    def _update_alpha(self, obs):
        """Update entropy temperature."""
        with torch.no_grad():
            actor_output = self.actor(obs)
            weights = actor_output['weights']
            ent = -torch.sum(weights * torch.log(weights + 1e-12), dim=-1)
        
        alpha_loss = (-(self.log_alpha * (ent.detach() - self.target_entropy))).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        
        return alpha_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        with torch.no_grad():
            for p_tgt, p in zip(self.critic_tgt.parameters(), self.critic.parameters()):
                p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
    
    def _compute_attention_regularization(self, attention_weights: list) -> torch.Tensor:
        """Compute attention regularization to encourage diverse attention."""
        total_reg = 0.0
        
        for attn_weights in attention_weights:
            # Encourage attention diversity (lower entropy = more focused)
            # We want some focus but not too much
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
            target_entropy = 0.5  # Moderate focus
            reg = torch.mean((attn_entropy - target_entropy) ** 2)
            total_reg += reg
        
        return total_reg / len(attention_weights)
    
    def _compute_regime_consistency_loss(self, regime_info: Dict) -> torch.Tensor:
        """Compute loss to encourage regime consistency."""
        regime_probs = regime_info['regime_probs']
        
        # Encourage confident regime predictions (but not too confident)
        max_prob = torch.max(regime_probs, dim=-1)[0]
        target_confidence = 0.7  # 70% confidence target
        consistency_loss = torch.mean((max_prob - target_confidence) ** 2)
        
        return consistency_loss
    
    def _compute_attention_metrics(self, attention_weights: list, regime_info: Dict) -> Dict[str, float]:
        """Compute attention interpretability metrics."""
        metrics = {}
        
        # Average attention entropy across layers and heads
        total_entropy = 0.0
        for attn_weights in attention_weights:
            # attn_weights shape: [batch_size, n_heads, seq_len, seq_len]
            batch_size, n_heads, seq_len, _ = attn_weights.shape
            
            # Compute entropy for each head
            for head in range(n_heads):
                head_weights = attn_weights[:, head, :, :]
                entropy = -torch.sum(head_weights * torch.log(head_weights + 1e-8), dim=(-2, -1))
                total_entropy += torch.mean(entropy)
        
        metrics['attention_entropy'] = (total_entropy / (len(attention_weights) * attention_weights[0].shape[1])).item()
        
        # Regime prediction confidence
        regime_probs = regime_info['regime_probs']
        max_prob = torch.max(regime_probs, dim=-1)[0]
        metrics['regime_confidence'] = torch.mean(max_prob).item()
        
        # Regime distribution
        regime_preds = regime_info['regime_pred']
        for i, regime_name in enumerate(['bull', 'bear', 'volatile', 'sideways']):
            regime_count = torch.sum(regime_preds == i).item()
            metrics[f'regime_{regime_name}_ratio'] = regime_count / len(regime_preds)
        
        return metrics
    
    def get_attention_interpretation(self, obs: np.ndarray) -> Dict[str, Any]:
        """Get detailed attention interpretation for a single observation."""
        self.actor.eval()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            actor_output = self.actor(obs_tensor)
            
            regime_info = actor_output['regime_info']
            attention_weights = actor_output['attention_weights']
            
            # Get regime prediction
            regime_pred = regime_info['regime_pred'].item()
            regime_names = ['Bull Market', 'Bear Market', 'Volatile Market', 'Sideways Market']
            regime_name = regime_names[regime_pred]
            regime_confidence = regime_info['regime_probs'][0, regime_pred].item()
            
            # Analyze attention patterns
            attention_analysis = self._analyze_attention_patterns(attention_weights)
            
            return {
                'regime': regime_name,
                'regime_confidence': regime_confidence,
                'regime_probabilities': regime_info['regime_probs'][0].numpy().tolist(),
                'attention_analysis': attention_analysis,
                'portfolio_weights': actor_output['weights'][0].numpy().tolist()
            }
    
    def _analyze_attention_patterns(self, attention_weights: list) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability."""
        analysis = {}
        
        # Average attention across layers
        avg_attention = torch.stack(attention_weights).mean(dim=0)  # [batch_size, n_heads, seq_len, seq_len]
        
        # Get attention to each feature group
        feature_group_attention = avg_attention.mean(dim=(1, 3))  # [batch_size, seq_len]
        
        feature_groups = ['Price Features', 'Volume Features', 'Technical Features', 'Momentum Features']
        
        analysis['feature_attention'] = {
            group: attention.item() 
            for group, attention in zip(feature_groups, feature_group_attention[0])
        }
        
        # Most attended feature group
        most_attended_idx = torch.argmax(feature_group_attention[0]).item()
        analysis['most_attended_feature'] = feature_groups[most_attended_idx]
        
        # Attention diversity
        attention_entropy = -torch.sum(feature_group_attention[0] * torch.log(feature_group_attention[0] + 1e-8))
        analysis['attention_diversity'] = attention_entropy.item()
        
        return analysis
    
    def get_regime_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance across different regimes."""
        if not self.regime_history:
            return {}
        
        # Count regime occurrences
        regime_counts = {}
        for regime in ['bull', 'bear', 'volatile', 'sideways']:
            regime_counts[regime] = self.regime_history.count(regime)
        
        total_periods = len(self.regime_history)
        regime_percentages = {
            regime: count / total_periods * 100 
            for regime, count in regime_counts.items()
        }
        
        return {
            'regime_distribution': regime_percentages,
            'total_periods': total_periods,
            'regime_transitions': self._analyze_regime_transitions()
        }
    
    def _analyze_regime_transitions(self) -> Dict[str, int]:
        """Analyze regime transition patterns."""
        if len(self.regime_history) < 2:
            return {}
        
        transitions = {}
        for i in range(1, len(self.regime_history)):
            from_regime = self.regime_history[i-1]
            to_regime = self.regime_history[i]
            transition = f"{from_regime} -> {to_regime}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        return transitions


def test_attention_sac():
    """Test the attention-based SAC agent."""
    config = SACConfig(
        obs_dim=54,  # 50 features + 4 regime features
        n_assets=5,
        actor_hidden=(256, 256),
        critic_hidden=(256, 256),
        lr=3e-4,
        exploration_noise=0.2
    )
    
    agent = AttentionSACAgent(config)
    
    # Test action selection
    obs = np.random.randn(54)
    action = agent.select_action(obs, return_attention=True)
    
    print("Attention SAC Agent Test:")
    print(f"Action shape: {action.shape}")
    print(f"Action sum: {np.sum(action):.4f}")
    print(f"Regime history: {agent.regime_history}")
    
    # Test attention interpretation
    interpretation = agent.get_attention_interpretation(obs)
    print(f"\nAttention Interpretation:")
    print(f"Regime: {interpretation['regime']}")
    print(f"Confidence: {interpretation['regime_confidence']:.3f}")
    print(f"Most attended feature: {interpretation['attention_analysis']['most_attended_feature']}")
    
    # Test update
    for _ in range(10):
        agent.replay.push(obs, action, 0.01, obs, False)
    
    losses = agent.update()
    print(f"\nUpdate losses: {losses}")


if __name__ == "__main__":
    test_attention_sac()
