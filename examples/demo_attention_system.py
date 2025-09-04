#!/usr/bin/env python3
"""
Demo script for the Attention-Based SAC Portfolio Allocator

This script demonstrates the key features of the attention-based SAC system:
1. Multi-head attention for market feature analysis
2. Market regime detection and classification
3. Regime-aware reward functions
4. Interactive attention visualization
5. Portfolio allocation with interpretability

Usage:
    python examples/demo_attention_system.py
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nets.attention_actor import AttentionActor, test_attention_actor
from src.envs.regime_rewards import RegimeDetector, RegimeAwareReward, test_regime_rewards
from src.agents.attention_sac import AttentionSACAgent, SACConfig
from src.utils.visualization import AttentionVisualizer, create_sample_visualizations


def demo_attention_actor():
    """Demonstrate the attention-based actor network."""
    print("🎯 Demo 1: Attention-Based Actor Network")
    print("=" * 50)
    
    # Create attention actor
    obs_dim = 50
    n_assets = 5
    actor = AttentionActor(obs_dim, n_assets, hidden_dim=256, n_heads=8, n_layers=2)
    
    # Generate sample observation
    obs = torch.randn(1, obs_dim)
    
    # Forward pass
    output = actor(obs)
    
    print(f"Input observation shape: {obs.shape}")
    print(f"Portfolio weights shape: {output['weights'].shape}")
    print(f"Portfolio weights: {output['weights'][0].detach().numpy()}")
    print(f"Weights sum: {output['weights'][0].sum().item():.4f}")
    
    # Regime detection
    regime_pred = output['regime_info']['regime_pred'][0].item()
    regime_names = ['Bull Market', 'Bear Market', 'Volatile Market', 'Sideways Market']
    regime_probs = output['regime_info']['regime_probs'][0].detach().numpy()
    
    print(f"\nMarket Regime Detection:")
    print(f"Predicted regime: {regime_names[regime_pred]}")
    print(f"Regime probabilities: {regime_probs}")
    
    # Attention analysis
    attention_interpretation = actor.get_attention_interpretation(output['attention_weights'])
    print(f"\nAttention Analysis:")
    print(f"Feature attention: {attention_interpretation['feature_attention']}")
    print(f"Most attended feature: {attention_interpretation['most_attended_feature']}")
    print(f"Attention entropy: {attention_interpretation['attention_entropy']:.4f}")
    
    return actor, output


def demo_regime_detection():
    """Demonstrate market regime detection."""
    print("\n🔍 Demo 2: Market Regime Detection")
    print("=" * 50)
    
    # Create regime detector
    detector = RegimeDetector(lookback_window=20)
    
    # Generate sample market data
    np.random.seed(42)
    n_periods = 100
    n_assets = 5
    
    # Simulate different market regimes
    bull_returns = np.random.normal(0.001, 0.01, (30, n_assets))  # Bull market
    bear_returns = np.random.normal(-0.0005, 0.015, (30, n_assets))  # Bear market
    volatile_returns = np.random.normal(0.0002, 0.025, (40, n_assets))  # Volatile market
    
    all_returns = np.vstack([bull_returns, bear_returns, volatile_returns])
    all_prices = np.cumprod(1 + all_returns.mean(axis=1))
    
    # Detect regimes over time
    detected_regimes = []
    for i in range(20, len(all_returns)):
        regime = detector.detect_regime(all_returns[:i+1], all_prices[:i+1])
        detected_regimes.append(regime.name)
    
    print(f"Detected regimes over time:")
    regime_counts = {}
    for regime in detected_regimes:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} periods ({count/len(detected_regimes)*100:.1f}%)")
    
    return detector, detected_regimes


def demo_regime_aware_rewards():
    """Demonstrate regime-aware reward functions."""
    print("\n💰 Demo 3: Regime-Aware Reward Functions")
    print("=" * 50)
    
    # Create regime-aware reward function
    reward_fn = RegimeAwareReward(
        momentum_weight=0.001,
        volatility_penalty=0.0005,
        diversification_weight=0.01,
        action_change_weight=0.005
    )
    
    # Generate sample data
    n_assets = 5
    weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])  # Current portfolio
    prev_weights = np.array([0.25, 0.25, 0.25, 0.15, 0.1])  # Previous portfolio
    returns = np.array([0.01, -0.005, 0.02, 0.008, -0.002])  # Asset returns
    prices = np.random.randn(20, n_assets)  # Price history
    base_reward = 0.005  # Base portfolio return
    
    # Compute regime-aware reward
    total_reward, components = reward_fn.compute_reward(
        weights, returns, prev_weights, prices, base_reward
    )
    
    print(f"Portfolio weights: {weights}")
    print(f"Asset returns: {returns}")
    print(f"Base reward: {base_reward:.4f}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"\nReward components:")
    for key, value in components.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    return reward_fn, components


def demo_attention_sac_agent():
    """Demonstrate the complete attention-based SAC agent."""
    print("\n🤖 Demo 4: Attention-Based SAC Agent")
    print("=" * 50)
    
    # Create agent configuration
    config = SACConfig(
        obs_dim=50,
        n_assets=5,
        actor_hidden=(256, 256),
        critic_hidden=(256, 256),
        lr=3e-4,
        exploration_noise=0.2
    )
    
    # Create attention-based SAC agent
    agent = AttentionSACAgent(config)
    
    # Simulate training episodes
    print("Simulating training episodes...")
    for episode in range(5):
        obs = np.random.randn(50)
        action = agent.select_action(obs, return_attention=True)
        reward = np.random.normal(0.001, 0.01)
        next_obs = np.random.randn(50)
        
        # Store experience
        agent.replay.push(obs, action, reward, next_obs, False)
        
        print(f"Episode {episode + 1}: Action sum = {np.sum(action):.4f}, Regime = {agent.regime_history[-1]}")
    
    # Update agent
    if len(agent.replay) >= config.batch_size:
        losses = agent.update()
        print(f"\nTraining update losses:")
        for key, value in losses.items():
            print(f"  {key}: {value:.6f}")
    
    # Get attention interpretation
    obs = np.random.randn(50)
    interpretation = agent.get_attention_interpretation(obs)
    
    print(f"\nAttention Interpretation:")
    print(f"Regime: {interpretation['regime']}")
    print(f"Confidence: {interpretation['regime_confidence']:.3f}")
    print(f"Portfolio weights: {interpretation['portfolio_weights']}")
    print(f"Most attended feature: {interpretation['attention_analysis']['most_attended_feature']}")
    
    # Regime performance summary
    regime_summary = agent.get_regime_performance_summary()
    print(f"\nRegime Performance Summary:")
    print(f"Total periods: {regime_summary['total_periods']}")
    print(f"Regime distribution: {regime_summary['regime_distribution']}")
    
    return agent, interpretation


def demo_visualization():
    """Demonstrate attention visualization capabilities."""
    print("\n📊 Demo 5: Attention Visualization")
    print("=" * 50)
    
    # Create sample data for visualization
    np.random.seed(42)
    n_timesteps = 50
    n_assets = 5
    n_features = 4
    
    # Generate sample attention weights
    attention_history = []
    for _ in range(n_timesteps):
        layer_attn = []
        for _ in range(2):  # 2 layers
            attn = torch.randn(1, 8, n_features, n_features)  # [batch, heads, seq, seq]
            attn = torch.softmax(attn, dim=-1)
            layer_attn.append(attn)
        attention_history.append(layer_attn)
    
    # Generate sample portfolio weights
    weights_history = []
    for _ in range(n_timesteps):
        weights = np.random.dirichlet(np.ones(n_assets))
        weights_history.append(weights)
    
    # Generate sample regime history
    regimes = ['bull', 'bear', 'volatile', 'sideways']
    regime_history = np.random.choice(regimes, n_timesteps).tolist()
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    print("Creating attention visualizations...")
    
    # Create attention heatmap
    fig1 = visualizer.plot_attention_heatmap(attention_history[0])
    print("✓ Attention heatmap created")
    
    # Create attention evolution plot
    fig2 = visualizer.plot_attention_evolution(attention_history)
    print("✓ Attention evolution plot created")
    
    # Create regime transitions plot
    fig3 = visualizer.plot_regime_transitions(regime_history)
    print("✓ Regime transitions plot created")
    
    # Create comprehensive portfolio analysis
    fig4 = visualizer.plot_portfolio_weights_with_attention(
        weights_history, attention_history, regime_history
    )
    print("✓ Portfolio analysis plot created")
    
    # Save visualizations
    output_dir = Path("assets/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig1.write_html(str(output_dir / "attention_heatmap.html"))
    fig2.write_html(str(output_dir / "attention_evolution.html"))
    fig3.write_html(str(output_dir / "regime_transitions.html"))
    fig4.write_html(str(output_dir / "portfolio_analysis.html"))
    
    print(f"📁 Visualizations saved to {output_dir}")
    
    return visualizer


def main():
    """Run all demos."""
    print("🚀 Attention-Based SAC Portfolio Allocator Demo")
    print("=" * 60)
    print("This demo showcases the advanced features of our attention-based")
    print("portfolio allocation system with market regime detection.")
    print("=" * 60)
    
    try:
        # Run all demos
        actor, actor_output = demo_attention_actor()
        detector, regimes = demo_regime_detection()
        reward_fn, reward_components = demo_regime_aware_rewards()
        agent, interpretation = demo_attention_sac_agent()
        visualizer = demo_visualization()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Multi-head attention for market feature analysis")
        print("✓ Market regime detection (Bull/Bear/Volatile/Sideways)")
        print("✓ Regime-aware reward functions")
        print("✓ Attention-based SAC agent with interpretability")
        print("✓ Interactive attention visualizations")
        print("✓ Portfolio allocation with regime awareness")
        
        print(f"\n📊 Generated visualizations saved to assets/plots/")
        print(f"🔍 Check the HTML files for interactive attention analysis!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
