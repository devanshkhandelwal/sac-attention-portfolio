#!/usr/bin/env python3
"""
System Test for Attention-Based SAC Portfolio Allocator

This script tests the complete system to ensure all components work together.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.nets.attention_actor import AttentionActor, MultiHeadAttention, MarketRegimeClassifier
        from src.envs.regime_rewards import RegimeDetector, RegimeAwareReward, MarketRegime
        from src.agents.attention_sac import AttentionSACAgent, SACConfig
        from src.utils.visualization import AttentionVisualizer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_attention_actor():
    """Test the attention actor."""
    print("\n🧠 Testing Attention Actor...")
    
    try:
        from src.nets.attention_actor import AttentionActor
        
        # Create actor
        actor = AttentionActor(obs_dim=50, n_assets=5)
        
        # Test forward pass
        obs = torch.randn(1, 50)
        output = actor(obs)
        
        # Check output structure
        assert 'weights' in output
        assert 'regime_info' in output
        assert 'attention_weights' in output
        
        # Check shapes
        assert output['weights'].shape == (1, 5)
        assert torch.allclose(output['weights'].sum(), torch.tensor(1.0), atol=1e-6)
        
        print("✅ Attention Actor test passed")
        return True
    except Exception as e:
        print(f"❌ Attention Actor test failed: {e}")
        return False

def test_regime_detection():
    """Test regime detection."""
    print("\n🔍 Testing Regime Detection...")
    
    try:
        from src.envs.regime_rewards import RegimeDetector, MarketRegime
        
        # Create detector
        detector = RegimeDetector(lookback_window=20)
        
        # Test with different market conditions
        # Bull market
        bull_returns = np.random.normal(0.001, 0.01, (25, 5))
        bull_prices = np.cumprod(1 + bull_returns.mean(axis=1))
        bull_regime = detector.detect_regime(bull_returns, bull_prices)
        
        # Volatile market
        volatile_returns = np.random.normal(0.0, 0.03, (25, 5))
        volatile_prices = np.cumprod(1 + volatile_returns.mean(axis=1))
        volatile_regime = detector.detect_regime(volatile_returns, volatile_prices)
        
        # Check that regimes are detected
        assert bull_regime in [MarketRegime.BULL, MarketRegime.SIDEWAYS]
        assert volatile_regime == MarketRegime.VOLATILE
        
        print("✅ Regime Detection test passed")
        return True
    except Exception as e:
        print(f"❌ Regime Detection test failed: {e}")
        return False

def test_regime_rewards():
    """Test regime-aware rewards."""
    print("\n💰 Testing Regime-Aware Rewards...")
    
    try:
        from src.envs.regime_rewards import RegimeAwareReward, MarketRegime
        
        # Create reward function
        reward_fn = RegimeAwareReward()
        
        # Test reward computation
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.005])
        prev_weights = np.array([0.25, 0.25, 0.25, 0.15, 0.1])
        prices = np.random.randn(20, 5)
        base_reward = 0.01
        
        total_reward, components = reward_fn.compute_reward(
            weights, returns, prev_weights, prices, base_reward, MarketRegime.BULL
        )
        
        # Check reward components
        assert 'base_reward' in components
        assert 'regime_bonus' in components
        assert 'momentum_bonus' in components
        assert 'volatility_penalty' in components
        assert 'diversification_bonus' in components
        assert 'action_change_bonus' in components
        
        print("✅ Regime-Aware Rewards test passed")
        return True
    except Exception as e:
        print(f"❌ Regime-Aware Rewards test failed: {e}")
        return False

def test_attention_sac_agent():
    """Test the attention-based SAC agent."""
    print("\n🤖 Testing Attention SAC Agent...")
    
    try:
        from src.agents.attention_sac import AttentionSACAgent, SACConfig
        
        # Create configuration
        config = SACConfig(
            obs_dim=50,
            n_assets=5,
            actor_hidden=(256, 256),
            critic_hidden=(256, 256),
            lr=3e-4,
            exploration_noise=0.2
        )
        
        # Create agent
        agent = AttentionSACAgent(config)
        
        # Test action selection
        obs = np.random.randn(50)
        action = agent.select_action(obs, return_attention=True)
        
        # Check action
        assert action.shape == (5,)
        assert np.allclose(np.sum(action), 1.0, atol=1e-6)
        assert np.all(action >= 0)
        
        # Test attention interpretation
        interpretation = agent.get_attention_interpretation(obs)
        assert 'regime' in interpretation
        assert 'attention_analysis' in interpretation
        assert 'portfolio_weights' in interpretation
        
        print("✅ Attention SAC Agent test passed")
        return True
    except Exception as e:
        print(f"❌ Attention SAC Agent test failed: {e}")
        return False

def test_visualization():
    """Test visualization components."""
    print("\n📊 Testing Visualization...")
    
    try:
        from src.utils.visualization import AttentionVisualizer
        
        # Create visualizer
        visualizer = AttentionVisualizer()
        
        # Create sample data
        np.random.seed(42)
        n_timesteps = 10
        n_assets = 5
        n_features = 4
        
        # Generate sample attention weights
        attention_history = []
        for _ in range(n_timesteps):
            layer_attn = []
            for _ in range(2):  # 2 layers
                attn = torch.randn(1, 8, n_features, n_features)
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
        
        # Test visualization creation
        fig1 = visualizer.plot_attention_heatmap(attention_history[0])
        fig2 = visualizer.plot_attention_evolution(attention_history)
        fig3 = visualizer.plot_regime_transitions(regime_history)
        fig4 = visualizer.plot_portfolio_weights_with_attention(
            weights_history, attention_history, regime_history
        )
        
        # Check that figures are created
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
        assert fig4 is not None
        
        print("✅ Visualization test passed")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end system."""
    print("\n🚀 Testing End-to-End System...")
    
    try:
        from src.agents.attention_sac import AttentionSACAgent, SACConfig
        
        # Create configuration
        config = SACConfig(
            obs_dim=50,
            n_assets=5,
            actor_hidden=(256, 256),
            critic_hidden=(256, 256),
            lr=3e-4,
            exploration_noise=0.2
        )
        
        # Create agent
        agent = AttentionSACAgent(config)
        
        # Simulate training episodes
        for episode in range(3):
            obs = np.random.randn(50)
            action = agent.select_action(obs, return_attention=True)
            reward = np.random.normal(0.001, 0.01)
            next_obs = np.random.randn(50)
            
            # Store experience
            agent.replay.push(obs, action, reward, next_obs, False)
        
        # Update agent
        if len(agent.replay) >= config.batch_size:
            losses = agent.update()
            assert 'critic_loss' in losses
            assert 'actor_loss' in losses
            assert 'alpha' in losses
        
        # Test regime performance summary
        regime_summary = agent.get_regime_performance_summary()
        assert 'regime_distribution' in regime_summary
        assert 'total_periods' in regime_summary
        
        print("✅ End-to-End System test passed")
        return True
    except Exception as e:
        print(f"❌ End-to-End System test failed: {e}")
        return False

def main():
    """Run all system tests."""
    print("🧪 Attention-Based SAC Portfolio Allocator - System Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_attention_actor,
        test_regime_detection,
        test_regime_rewards,
        test_attention_sac_agent,
        test_visualization,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
