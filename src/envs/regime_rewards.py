"""
Regime-Aware Reward Functions for Portfolio Optimization
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = 0
    BEAR = 1
    VOLATILE = 2
    SIDEWAYS = 3


class RegimeDetector:
    """Detects market regimes based on price action and volatility."""
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.regime_thresholds = {
            'volatility_threshold': 0.02,  # 2% daily volatility
            'trend_threshold': 0.05,       # 5% trend strength
            'momentum_threshold': 0.1      # 10% momentum
        }
    
    def detect_regime(self, returns: np.ndarray, prices: np.ndarray) -> MarketRegime:
        """
        Detect current market regime based on recent price action.
        
        Args:
            returns: Recent returns array (can be 1D or 2D)
            prices: Recent prices array (can be 1D or 2D)
            
        Returns:
            Detected market regime
        """
        if len(returns) < self.lookback_window:
            return MarketRegime.SIDEWAYS
        
        recent_returns = returns[-self.lookback_window:]
        recent_prices = prices[-self.lookback_window:]
        
        # Flatten if multi-dimensional
        if recent_returns.ndim > 1:
            recent_returns = recent_returns.flatten()
        if recent_prices.ndim > 1:
            recent_prices = recent_prices.flatten()
        
        # Calculate regime indicators
        volatility = np.std(recent_returns)
        trend_strength = self._calculate_trend_strength(recent_prices)
        momentum = self._calculate_momentum(recent_returns)
        
        # Regime classification logic
        if volatility > self.regime_thresholds['volatility_threshold']:
            return MarketRegime.VOLATILE
        elif trend_strength > self.regime_thresholds['trend_threshold']:
            if momentum > 0:
                return MarketRegime.BULL
            else:
                return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression slope."""
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return abs(slope) / prices[0]  # Normalized slope
    
    def _calculate_momentum(self, returns: np.ndarray) -> float:
        """Calculate momentum as weighted average of recent returns."""
        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weights = weights / weights.sum()
        return np.sum(returns * weights)


class RegimeAwareReward:
    """
    Regime-aware reward function that adapts to different market conditions.
    """
    
    def __init__(
        self,
        base_reward_weight: float = 1.0,
        regime_weights: Optional[Dict[MarketRegime, float]] = None,
        momentum_weight: float = 0.001,
        volatility_penalty: float = 0.0005,
        diversification_weight: float = 0.01,
        action_change_weight: float = 0.005
    ):
        self.base_reward_weight = base_reward_weight
        self.regime_weights = regime_weights or {
            MarketRegime.BULL: 1.2,      # Favor momentum in bull markets
            MarketRegime.BEAR: 0.8,      # Reduce risk in bear markets
            MarketRegime.VOLATILE: 0.6,  # Strongly favor diversification
            MarketRegime.SIDEWAYS: 1.0   # Neutral weighting
        }
        
        self.momentum_weight = momentum_weight
        self.volatility_penalty = volatility_penalty
        self.diversification_weight = diversification_weight
        self.action_change_weight = action_change_weight
        
        self.regime_detector = RegimeDetector()
    
    def compute_reward(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        prev_weights: np.ndarray,
        prices: np.ndarray,
        base_reward: float,
        regime: Optional[MarketRegime] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute regime-aware reward.
        
        Args:
            weights: Current portfolio weights
            returns: Asset returns
            prev_weights: Previous portfolio weights
            prices: Recent price history
            base_reward: Base portfolio return
            regime: Detected market regime (if None, will detect)
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        if regime is None:
            regime = self.regime_detector.detect_regime(returns, prices)
        
        # Base reward with regime weighting
        regime_weight = self.regime_weights.get(regime, 1.0)
        weighted_base_reward = base_reward * regime_weight
        
        # Regime-specific reward components
        regime_bonus = self._compute_regime_bonus(weights, returns, regime)
        momentum_bonus = self._compute_momentum_bonus(weights, returns, regime)
        volatility_penalty = self._compute_volatility_penalty(weights, returns, regime)
        diversification_bonus = self._compute_diversification_bonus(weights, regime)
        action_change_bonus = self._compute_action_change_bonus(weights, prev_weights, regime)
        
        # Combine all components
        total_reward = (
            weighted_base_reward +
            regime_bonus +
            momentum_bonus -
            volatility_penalty +
            diversification_bonus +
            action_change_bonus
        )
        
        reward_components = {
            'base_reward': weighted_base_reward,
            'regime_bonus': regime_bonus,
            'momentum_bonus': momentum_bonus,
            'volatility_penalty': volatility_penalty,
            'diversification_bonus': diversification_bonus,
            'action_change_bonus': action_change_bonus,
            'regime': regime.value,
            'regime_name': regime.name
        }
        
        return total_reward, reward_components
    
    def _compute_regime_bonus(self, weights: np.ndarray, returns: np.ndarray, regime: MarketRegime) -> float:
        """Compute regime-specific bonus."""
        if regime == MarketRegime.BULL:
            # Reward momentum and growth
            momentum_signal = np.dot(weights, returns)
            return self.momentum_weight * np.tanh(momentum_signal * 5)
        
        elif regime == MarketRegime.BEAR:
            # Reward defensive positioning
            defensive_signal = 1 - np.dot(weights, returns)
            return self.momentum_weight * 0.5 * np.tanh(defensive_signal * 3)
        
        elif regime == MarketRegime.VOLATILE:
            # Reward low volatility and diversification
            portfolio_vol = np.abs(np.dot(weights, returns))
            return -self.volatility_penalty * portfolio_vol * 2
        
        else:  # SIDEWAYS
            # Neutral regime, minimal bonus
            return 0.0
    
    def _compute_momentum_bonus(self, weights: np.ndarray, returns: np.ndarray, regime: MarketRegime) -> float:
        """Compute momentum-based bonus."""
        if regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            # Strong momentum signals in trending markets
            momentum_signal = np.dot(weights, returns)
            return self.momentum_weight * np.tanh(momentum_signal * 3)
        else:
            # Reduced momentum in volatile/sideways markets
            momentum_signal = np.dot(weights, returns)
            return self.momentum_weight * 0.3 * np.tanh(momentum_signal * 2)
    
    def _compute_volatility_penalty(self, weights: np.ndarray, returns: np.ndarray, regime: MarketRegime) -> float:
        """Compute volatility penalty."""
        portfolio_vol = np.abs(np.dot(weights, returns))
        
        if regime == MarketRegime.VOLATILE:
            # Higher penalty in volatile markets
            return self.volatility_penalty * portfolio_vol * 3
        elif regime == MarketRegime.BEAR:
            # Moderate penalty in bear markets
            return self.volatility_penalty * portfolio_vol * 1.5
        else:
            # Standard penalty
            return self.volatility_penalty * portfolio_vol
    
    def _compute_diversification_bonus(self, weights: np.ndarray, regime: MarketRegime) -> float:
        """Compute diversification bonus."""
        herfindahl = np.sum(weights ** 2)
        diversity_bonus = (1.0 - herfindahl) * self.diversification_weight
        
        if regime == MarketRegime.VOLATILE:
            # Strong diversification bonus in volatile markets
            return diversity_bonus * 2
        elif regime == MarketRegime.BEAR:
            # Moderate diversification bonus in bear markets
            return diversity_bonus * 1.5
        else:
            # Standard diversification bonus
            return diversity_bonus
    
    def _compute_action_change_bonus(self, weights: np.ndarray, prev_weights: np.ndarray, regime: MarketRegime) -> float:
        """Compute action change bonus."""
        change_magnitude = np.sqrt(np.sum((weights - prev_weights) ** 2))
        
        if regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            # Reward more active rebalancing in trending markets
            return np.tanh(change_magnitude * 10) * self.action_change_weight * 1.5
        elif regime == MarketRegime.VOLATILE:
            # Moderate rebalancing in volatile markets
            return np.tanh(change_magnitude * 8) * self.action_change_weight * 1.2
        else:
            # Standard rebalancing bonus
            return np.tanh(change_magnitude * 10) * self.action_change_weight


class RegimePerformanceAnalyzer:
    """Analyzes performance across different market regimes."""
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
    
    def analyze_regime_performance(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        portfolio_returns: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze portfolio performance across different market regimes.
        
        Args:
            returns: Asset returns
            prices: Price history
            portfolio_returns: Portfolio returns
            
        Returns:
            Dictionary with performance metrics by regime
        """
        regime_performance = {}
        
        for regime in MarketRegime:
            regime_mask = self._get_regime_mask(returns, prices, regime)
            if np.sum(regime_mask) > 0:
                regime_portfolio_returns = portfolio_returns[regime_mask]
                regime_performance[regime.name] = self._compute_regime_metrics(regime_portfolio_returns)
        
        return regime_performance
    
    def _get_regime_mask(self, returns: np.ndarray, prices: np.ndarray, regime: MarketRegime) -> np.ndarray:
        """Get boolean mask for specific regime periods."""
        mask = np.zeros(len(returns), dtype=bool)
        
        for i in range(len(returns)):
            if i >= self.regime_detector.lookback_window:
                start_idx = i - self.regime_detector.lookback_window
                detected_regime = self.regime_detector.detect_regime(
                    returns[start_idx:i+1], 
                    prices[start_idx:i+1]
                )
                mask[i] = (detected_regime == regime)
        
        return mask
    
    def _compute_regime_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Compute performance metrics for a regime."""
        if len(returns) == 0:
            return {}
        
        return {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
            'max_drawdown': self._compute_max_drawdown(returns),
            'win_rate': np.mean(returns > 0),
            'total_return': np.sum(returns)
        }
    
    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)


def test_regime_rewards():
    """Test the regime-aware reward system."""
    n_assets = 5
    n_periods = 100
    
    # Generate synthetic data
    np.random.seed(42)
    returns = np.random.randn(n_periods, n_assets) * 0.02
    prices = np.cumprod(1 + returns.mean(axis=1))
    weights = np.random.dirichlet(np.ones(n_assets))
    prev_weights = np.random.dirichlet(np.ones(n_assets))
    base_reward = 0.01
    
    # Test regime detection
    detector = RegimeDetector()
    regime = detector.detect_regime(returns[-20:], prices[-20:])
    print(f"Detected regime: {regime.name}")
    
    # Test regime-aware rewards
    reward_fn = RegimeAwareReward()
    total_reward, components = reward_fn.compute_reward(
        weights, returns[-1], prev_weights, prices[-20:], base_reward, regime
    )
    
    print(f"Total reward: {total_reward:.6f}")
    print("Reward components:")
    for key, value in components.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Test performance analysis
    analyzer = RegimePerformanceAnalyzer()
    portfolio_returns = np.random.randn(n_periods) * 0.01
    regime_performance = analyzer.analyze_regime_performance(returns, prices, portfolio_returns)
    
    print("\nRegime Performance Analysis:")
    for regime_name, metrics in regime_performance.items():
        print(f"\n{regime_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    test_regime_rewards()
