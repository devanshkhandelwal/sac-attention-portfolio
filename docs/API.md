# API Reference

This document provides detailed API documentation for the Attention-Based SAC Portfolio Allocator.

## Core Components

### AttentionActor

The main attention-based actor network for portfolio allocation.

```python
class AttentionActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_assets: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0
    )
```

#### Parameters
- `obs_dim` (int): Observation dimension (number of input features)
- `n_assets` (int): Number of assets in the portfolio
- `hidden_dim` (int): Hidden dimension for the network (default: 256)
- `n_heads` (int): Number of attention heads (default: 8)
- `n_layers` (int): Number of attention layers (default: 2)
- `dropout` (float): Dropout rate (default: 0.1)
- `temperature` (float): Temperature for softmax scaling (default: 1.0)

#### Methods

##### `forward(obs: torch.Tensor) -> Dict[str, torch.Tensor]`
Forward pass through the attention-based actor.

**Parameters:**
- `obs`: Observation tensor [batch_size, obs_dim]

**Returns:**
- Dictionary containing:
  - `weights`: Portfolio weights [batch_size, n_assets]
  - `logits`: Raw logits before softmax
  - `regime_info`: Regime classification results
  - `attention_weights`: Attention weights for interpretability
  - `final_representation`: Final feature representation

##### `get_regime_interpretation(regime_pred: torch.Tensor) -> str`
Convert regime prediction to human-readable format.

**Parameters:**
- `regime_pred`: Regime prediction tensor

**Returns:**
- Human-readable regime name

##### `get_attention_interpretation(attention_weights: list, feature_names: list = None) -> Dict`
Interpret attention weights for explainability.

**Parameters:**
- `attention_weights`: List of attention weight tensors
- `feature_names`: Optional list of feature group names

**Returns:**
- Dictionary with attention interpretation

### MultiHeadAttention

Multi-head self-attention mechanism for market feature analysis.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1)
```

#### Parameters
- `d_model` (int): Model dimension
- `n_heads` (int): Number of attention heads (default: 8)
- `dropout` (float): Dropout rate (default: 0.1)

#### Methods

##### `forward(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]`
Forward pass through multi-head attention.

**Parameters:**
- `x`: Input tensor [batch_size, seq_len, d_model]
- `mask`: Optional attention mask

**Returns:**
- Tuple of (output, attention_weights)

### MarketRegimeClassifier

Classifies market regimes: Bull, Bear, Volatile, Sideways.

```python
class MarketRegimeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128)
```

#### Parameters
- `input_dim` (int): Input dimension
- `hidden_dim` (int): Hidden dimension (default: 128)

#### Methods

##### `forward(x: torch.Tensor) -> Dict[str, torch.Tensor]`
Forward pass for regime classification.

**Parameters:**
- `x`: Input tensor [batch_size, input_dim]

**Returns:**
- Dictionary containing:
  - `regime_logits`: Raw logits for regime classification
  - `regime_probs`: Softmax probabilities
  - `regime_pred`: Predicted regime class

## Environment Components

### RegimeDetector

Detects market regimes based on price action and volatility.

```python
class RegimeDetector:
    def __init__(self, lookback_window: int = 20)
```

#### Parameters
- `lookback_window` (int): Number of periods to look back for regime detection (default: 20)

#### Methods

##### `detect_regime(returns: np.ndarray, prices: np.ndarray) -> MarketRegime`
Detect current market regime based on recent price action.

**Parameters:**
- `returns`: Recent returns array (can be 1D or 2D)
- `prices`: Recent prices array (can be 1D or 2D)

**Returns:**
- Detected market regime (MarketRegime enum)

##### `_calculate_trend_strength(prices: np.ndarray) -> float`
Calculate trend strength using linear regression slope.

**Parameters:**
- `prices`: Price array

**Returns:**
- Normalized trend strength

##### `_calculate_momentum(returns: np.ndarray) -> float`
Calculate momentum as weighted average of recent returns.

**Parameters:**
- `returns`: Returns array

**Returns:**
- Momentum value

### RegimeAwareReward

Regime-aware reward function that adapts to different market conditions.

```python
class RegimeAwareReward:
    def __init__(
        self,
        base_reward_weight: float = 1.0,
        regime_weights: Optional[Dict[MarketRegime, float]] = None,
        momentum_weight: float = 0.001,
        volatility_penalty: float = 0.0005,
        diversification_weight: float = 0.01,
        action_change_weight: float = 0.005
    )
```

#### Parameters
- `base_reward_weight` (float): Weight for base reward (default: 1.0)
- `regime_weights` (Dict): Regime-specific reward weights (default: None)
- `momentum_weight` (float): Weight for momentum bonus (default: 0.001)
- `volatility_penalty` (float): Weight for volatility penalty (default: 0.0005)
- `diversification_weight` (float): Weight for diversification bonus (default: 0.01)
- `action_change_weight` (float): Weight for action change bonus (default: 0.005)

#### Methods

##### `compute_reward(weights, returns, prev_weights, prices, base_reward, regime=None) -> Tuple[float, Dict[str, float]]`
Compute regime-aware reward.

**Parameters:**
- `weights`: Current portfolio weights
- `returns`: Asset returns
- `prev_weights`: Previous portfolio weights
- `prices`: Recent price history
- `base_reward`: Base portfolio return
- `regime`: Detected market regime (if None, will detect)

**Returns:**
- Tuple of (total_reward, reward_components)

### RegimePerformanceAnalyzer

Analyzes performance across different market regimes.

```python
class RegimePerformanceAnalyzer:
    def __init__(self)
```

#### Methods

##### `analyze_regime_performance(returns, prices, portfolio_returns) -> Dict[str, Dict[str, float]]`
Analyze portfolio performance across different market regimes.

**Parameters:**
- `returns`: Asset returns
- `prices`: Price history
- `portfolio_returns`: Portfolio returns

**Returns:**
- Dictionary with performance metrics by regime

## Agent Components

### AttentionSACAgent

Enhanced SAC Agent with Attention-Based Actor and Regime Awareness.

```python
class AttentionSACAgent(SACAgent):
    def __init__(self, config: SACConfig)
```

#### Parameters
- `config`: SAC configuration object

#### Methods

##### `select_action(obs: np.ndarray, greedy: bool = False, return_attention: bool = False) -> np.ndarray`
Select action using attention-based actor.

**Parameters:**
- `obs`: Observation array
- `greedy`: Whether to use greedy action selection
- `return_attention`: Whether to return attention weights

**Returns:**
- Action array (portfolio weights)

##### `update(batch_size: Optional[int] = None) -> Dict[str, float]`
Update agent with enhanced attention-based learning.

**Parameters:**
- `batch_size`: Optional batch size override

**Returns:**
- Dictionary of loss values and metrics

##### `get_attention_interpretation(obs: np.ndarray) -> Dict[str, Any]`
Get detailed attention interpretation for a single observation.

**Parameters:**
- `obs`: Observation array

**Returns:**
- Dictionary with attention interpretation

##### `get_regime_performance_summary() -> Dict[str, Any]`
Get summary of performance across different regimes.

**Returns:**
- Dictionary with regime performance summary

### SACConfig

Configuration class for SAC agent.

```python
@dataclass
class SACConfig:
    obs_dim: int
    n_assets: int
    actor_hidden: Tuple[int, int] = (256, 256)
    critic_hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    target_entropy: Optional[float] = None
    temperature_init: float = 1.0
    exploration_noise: float = 0.1
```

## Visualization Components

### AttentionVisualizer

Visualizes attention weights and regime detection for interpretability.

```python
class AttentionVisualizer:
    def __init__(self, asset_names: List[str] = None)
```

#### Parameters
- `asset_names`: List of asset names for visualization (default: None)

#### Methods

##### `plot_attention_heatmap(attention_weights, layer_idx=0, head_idx=0, save_path=None) -> go.Figure`
Create interactive attention heatmap.

**Parameters:**
- `attention_weights`: List of attention weight tensors
- `layer_idx`: Which layer to visualize (default: 0)
- `head_idx`: Which attention head to visualize (default: 0)
- `save_path`: Optional path to save the plot

**Returns:**
- Plotly figure object

##### `plot_attention_evolution(attention_history, time_steps=50, save_path=None) -> go.Figure`
Plot how attention patterns evolve over time.

**Parameters:**
- `attention_history`: List of attention weights over time
- `time_steps`: Number of time steps to show (default: 50)
- `save_path`: Optional path to save the plot

**Returns:**
- Plotly figure object

##### `plot_regime_transitions(regime_history, save_path=None) -> go.Figure`
Visualize market regime transitions over time.

**Parameters:**
- `regime_history`: List of regime names over time
- `save_path`: Optional path to save the plot

**Returns:**
- Plotly figure object

##### `plot_portfolio_weights_with_attention(weights_history, attention_history, regime_history, save_path=None) -> go.Figure`
Plot portfolio weights alongside attention patterns and regime detection.

**Parameters:**
- `weights_history`: List of portfolio weight arrays over time
- `attention_history`: List of attention weights over time
- `regime_history`: List of regime names over time
- `save_path`: Optional path to save the plot

**Returns:**
- Plotly figure object

##### `create_attention_dashboard(agent_outputs, save_path=None) -> go.Figure`
Create comprehensive attention dashboard.

**Parameters:**
- `agent_outputs`: List of agent output dictionaries
- `save_path`: Optional path to save the dashboard

**Returns:**
- Plotly figure object

### PerformanceVisualizer

Visualizes portfolio performance and regime analysis.

```python
class PerformanceVisualizer:
    def __init__(self, asset_names: List[str] = None)
```

#### Parameters
- `asset_names`: List of asset names for visualization (default: None)

#### Methods

##### `plot_performance_comparison(results, save_path=None) -> go.Figure`
Create performance comparison chart.

**Parameters:**
- `results`: Dictionary with strategy results
- `save_path`: Optional path to save the plot

**Returns:**
- Plotly figure object

##### `plot_regime_performance(regime_performance, save_path=None) -> go.Figure`
Plot performance metrics by market regime.

**Parameters:**
- `regime_performance`: Performance metrics by regime
- `save_path`: Optional path to save the plot

**Returns:**
- Plotly figure object

## Enums

### MarketRegime

Market regime enumeration.

```python
class MarketRegime(Enum):
    BULL = 0
    BEAR = 1
    VOLATILE = 2
    SIDEWAYS = 3
```

## Utility Functions

### `test_attention_actor()`
Test the attention actor implementation.

### `test_regime_rewards()`
Test the regime-aware reward system.

### `create_sample_visualizations()`
Create sample visualizations for demonstration.

## Configuration

### YAML Configuration

The system uses YAML configuration files for easy parameter tuning. See `configs/attention_sac.yaml` for a complete example.

#### Key Configuration Sections

- **env**: Environment and data configuration
- **sac**: SAC agent hyperparameters
- **regime_rewards**: Regime-aware reward parameters
- **training**: Training configuration
- **evaluation**: Evaluation metrics and baselines
- **visualization**: Visualization settings
- **logging**: Logging configuration
- **hardware**: Hardware and device settings

## Error Handling

The API includes comprehensive error handling:

- **Input validation**: All functions validate input parameters
- **Shape checking**: Tensor shapes are validated
- **Type checking**: Type hints are enforced
- **Graceful degradation**: Functions handle edge cases gracefully

## Performance Considerations

- **Batch processing**: All operations support batch processing
- **GPU acceleration**: Automatic GPU detection and usage
- **Memory efficiency**: Efficient memory usage for large datasets
- **Vectorized operations**: NumPy/PyTorch vectorized operations

## Examples

See the `examples/` directory for comprehensive usage examples:

- `demo_attention_system.py`: Complete system demonstration
- Configuration examples in `configs/`
- Test examples in `tests/`
