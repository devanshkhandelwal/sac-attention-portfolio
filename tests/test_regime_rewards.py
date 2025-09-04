"""
Tests for the Regime-Aware Reward Functions
"""

import pytest
import numpy as np
from src.envs.regime_rewards import (
    RegimeDetector, 
    RegimeAwareReward, 
    RegimePerformanceAnalyzer,
    MarketRegime
)


class TestRegimeDetector:
    """Test the market regime detector."""
    
    def test_regime_detection_bull_market(self):
        """Test detection of bull market regime."""
        detector = RegimeDetector(lookback_window=20)
        
        # Create bull market data (positive trend, low volatility)
        returns = np.random.normal(0.001, 0.01, (25, 5))  # Positive returns, low vol
        prices = np.cumprod(1 + returns.mean(axis=1))
        
        regime = detector.detect_regime(returns, prices)
        
        # Should detect bull market
        assert regime in [MarketRegime.BULL, MarketRegime.SIDEWAYS]  # Could be sideways if trend not strong enough
    
    def test_regime_detection_bear_market(self):
        """Test detection of bear market regime."""
        detector = RegimeDetector(lookback_window=20)
        
        # Create bear market data (negative trend, low volatility)
        returns = np.random.normal(-0.001, 0.01, (25, 5))  # Negative returns, low vol
        prices = np.cumprod(1 + returns.mean(axis=1))
        
        regime = detector.detect_regime(returns, prices)
        
        # Should detect bear market
        assert regime in [MarketRegime.BEAR, MarketRegime.SIDEWAYS]  # Could be sideways if trend not strong enough
    
    def test_regime_detection_volatile_market(self):
        """Test detection of volatile market regime."""
        detector = RegimeDetector(lookback_window=20)
        
        # Create volatile market data (high volatility)
        returns = np.random.normal(0.0, 0.03, (25, 5))  # High volatility
        prices = np.cumprod(1 + returns.mean(axis=1))
        
        regime = detector.detect_regime(returns, prices)
        
        # Should detect volatile market
        assert regime == MarketRegime.VOLATILE
    
    def test_regime_detection_insufficient_data(self):
        """Test regime detection with insufficient data."""
        detector = RegimeDetector(lookback_window=20)
        
        # Create data with less than lookback window
        returns = np.random.randn(10, 5)
        prices = np.random.randn(10)
        
        regime = detector.detect_regime(returns, prices)
        
        # Should return sideways for insufficient data
        assert regime == MarketRegime.SIDEWAYS
    
    def test_regime_detection_multi_dimensional(self):
        """Test regime detection with multi-dimensional data."""
        detector = RegimeDetector(lookback_window=20)
        
        # Create multi-dimensional data
        returns = np.random.randn(25, 5, 3)  # 3D array
        prices = np.random.randn(25, 3)  # 2D array
        
        regime = detector.detect_regime(returns, prices)
        
        # Should handle multi-dimensional data
        assert regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.VOLATILE, MarketRegime.SIDEWAYS]
    
    def test_trend_strength_calculation(self):
        """Test trend strength calculation."""
        detector = RegimeDetector()
        
        # Create data with known trend
        prices = np.array([100, 101, 102, 103, 104])  # Upward trend
        
        trend_strength = detector._calculate_trend_strength(prices)
        
        # Should be positive for upward trend
        assert trend_strength > 0
    
    def test_momentum_calculation(self):
        """Test momentum calculation."""
        detector = RegimeDetector()
        
        # Create data with known momentum
        returns = np.array([0.01, 0.02, 0.015, 0.018, 0.02])  # Generally positive
        
        momentum = detector._calculate_momentum(returns)
        
        # Should be positive for positive returns
        assert momentum > 0


class TestRegimeAwareReward:
    """Test the regime-aware reward function."""
    
    def test_reward_computation_bull_market(self):
        """Test reward computation in bull market."""
        reward_fn = RegimeAwareReward()
        
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.005])  # Positive returns
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
        assert 'regime' in components
        assert 'regime_name' in components
        
        # Bull market should have positive regime bonus for positive momentum
        assert components['regime'] == MarketRegime.BULL.value
        assert components['regime_name'] == 'BULL'
    
    def test_reward_computation_bear_market(self):
        """Test reward computation in bear market."""
        reward_fn = RegimeAwareReward()
        
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        returns = np.array([-0.01, -0.02, -0.015, -0.008, -0.005])  # Negative returns
        prev_weights = np.array([0.25, 0.25, 0.25, 0.15, 0.1])
        prices = np.random.randn(20, 5)
        base_reward = -0.005
        
        total_reward, components = reward_fn.compute_reward(
            weights, returns, prev_weights, prices, base_reward, MarketRegime.BEAR
        )
        
        # Bear market should have different regime bonus
        assert components['regime'] == MarketRegime.BEAR.value
        assert components['regime_name'] == 'BEAR'
    
    def test_reward_computation_volatile_market(self):
        """Test reward computation in volatile market."""
        reward_fn = RegimeAwareReward()
        
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        returns = np.array([0.05, -0.03, 0.04, -0.02, 0.01])  # High volatility
        prev_weights = np.array([0.25, 0.25, 0.25, 0.15, 0.1])
        prices = np.random.randn(20, 5)
        base_reward = 0.002
        
        total_reward, components = reward_fn.compute_reward(
            weights, returns, prev_weights, prices, base_reward, MarketRegime.VOLATILE
        )
        
        # Volatile market should have higher volatility penalty
        assert components['regime'] == MarketRegime.VOLATILE.value
        assert components['regime_name'] == 'VOLATILE'
        assert components['volatility_penalty'] > 0
    
    def test_reward_computation_sideways_market(self):
        """Test reward computation in sideways market."""
        reward_fn = RegimeAwareReward()
        
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        returns = np.array([0.001, -0.001, 0.002, -0.001, 0.001])  # Low volatility
        prev_weights = np.array([0.25, 0.25, 0.25, 0.15, 0.1])
        prices = np.random.randn(20, 5)
        base_reward = 0.0005
        
        total_reward, components = reward_fn.compute_reward(
            weights, returns, prev_weights, prices, base_reward, MarketRegime.SIDEWAYS
        )
        
        # Sideways market should have neutral regime bonus
        assert components['regime'] == MarketRegime.SIDEWAYS.value
        assert components['regime_name'] == 'SIDEWAYS'
        assert components['regime_bonus'] == 0.0
    
    def test_diversification_bonus(self):
        """Test diversification bonus calculation."""
        reward_fn = RegimeAwareReward()
        
        # Equal weights (high diversification)
        equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Concentrated weights (low diversification)
        concentrated_weights = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        
        returns = np.random.randn(5)
        prev_weights = np.random.randn(5)
        prices = np.random.randn(20, 5)
        base_reward = 0.01
        
        # Test with equal weights
        _, components_equal = reward_fn.compute_reward(
            equal_weights, returns, prev_weights, prices, base_reward, MarketRegime.SIDEWAYS
        )
        
        # Test with concentrated weights
        _, components_concentrated = reward_fn.compute_reward(
            concentrated_weights, returns, prev_weights, prices, base_reward, MarketRegime.SIDEWAYS
        )
        
        # Equal weights should have higher diversification bonus
        assert components_equal['diversification_bonus'] > components_concentrated['diversification_bonus']
    
    def test_action_change_bonus(self):
        """Test action change bonus calculation."""
        reward_fn = RegimeAwareReward()
        
        weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        returns = np.random.randn(5)
        prices = np.random.randn(20, 5)
        base_reward = 0.01
        
        # Small change
        small_change = np.array([0.28, 0.22, 0.2, 0.2, 0.1])
        _, components_small = reward_fn.compute_reward(
            weights, returns, small_change, prices, base_reward, MarketRegime.SIDEWAYS
        )
        
        # Large change
        large_change = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
        _, components_large = reward_fn.compute_reward(
            weights, returns, large_change, prices, base_reward, MarketRegime.SIDEWAYS
        )
        
        # Large change should have higher action change bonus
        assert components_large['action_change_bonus'] > components_small['action_change_bonus']


class TestRegimePerformanceAnalyzer:
    """Test the regime performance analyzer."""
    
    def test_regime_performance_analysis(self):
        """Test regime performance analysis."""
        analyzer = RegimePerformanceAnalyzer()
        
        # Create sample data
        n_periods = 100
        n_assets = 5
        
        returns = np.random.randn(n_periods, n_assets) * 0.02
        prices = np.cumprod(1 + returns.mean(axis=1))
        portfolio_returns = np.random.randn(n_periods) * 0.01
        
        regime_performance = analyzer.analyze_regime_performance(returns, prices, portfolio_returns)
        
        # Should have performance metrics for each regime
        assert isinstance(regime_performance, dict)
        
        for regime_name, metrics in regime_performance.items():
            assert isinstance(metrics, dict)
            assert 'mean_return' in metrics
            assert 'volatility' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'win_rate' in metrics
            assert 'total_return' in metrics
    
    def test_regime_mask_generation(self):
        """Test regime mask generation."""
        analyzer = RegimePerformanceAnalyzer()
        
        # Create sample data
        n_periods = 50
        returns = np.random.randn(n_periods, 5)
        prices = np.random.randn(n_periods)
        
        # Test mask for each regime
        for regime in MarketRegime:
            mask = analyzer._get_regime_mask(returns, prices, regime)
            
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == bool
            assert len(mask) == n_periods
    
    def test_regime_metrics_computation(self):
        """Test regime metrics computation."""
        analyzer = RegimePerformanceAnalyzer()
        
        # Create sample returns
        returns = np.array([0.01, -0.005, 0.02, 0.008, -0.002, 0.015, -0.01, 0.005])
        
        metrics = analyzer._compute_regime_metrics(returns)
        
        assert 'mean_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'total_return' in metrics
        
        # Check that metrics are reasonable
        assert metrics['win_rate'] >= 0 and metrics['win_rate'] <= 1
        assert metrics['volatility'] >= 0
        assert metrics['max_drawdown'] <= 0
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        analyzer = RegimePerformanceAnalyzer()
        
        # Create returns with known drawdown
        returns = np.array([0.1, 0.05, -0.1, -0.05, 0.02, 0.03])  # Should have drawdown
        
        max_dd = analyzer._compute_max_drawdown(returns)
        
        # Should be negative (drawdown)
        assert max_dd <= 0
        
        # Test with all positive returns
        positive_returns = np.array([0.01, 0.02, 0.015, 0.008, 0.005])
        max_dd_positive = analyzer._compute_max_drawdown(positive_returns)
        
        # Should be 0 or very close to 0
        assert max_dd_positive <= 0.001


if __name__ == "__main__":
    pytest.main([__file__])
