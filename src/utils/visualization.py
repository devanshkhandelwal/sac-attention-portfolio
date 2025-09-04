"""
Advanced Visualization Tools for Attention-Based SAC Portfolio Allocator
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AttentionVisualizer:
    """Visualizes attention weights and regime detection for interpretability."""
    
    def __init__(self, asset_names: List[str] = None):
        self.asset_names = asset_names or ['SPY', 'TLT', 'GLD', 'DBC', 'USO']
        self.feature_groups = ['Price Features', 'Volume Features', 'Technical Features', 'Momentum Features']
        self.regime_colors = {
            'Bull Market': '#2E8B57',      # Sea Green
            'Bear Market': '#DC143C',      # Crimson
            'Volatile Market': '#FF8C00',  # Dark Orange
            'Sideways Market': '#4682B4'   # Steel Blue
        }
    
    def plot_attention_heatmap(
        self, 
        attention_weights: List[torch.Tensor], 
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive attention heatmap.
        
        Args:
            attention_weights: List of attention weight tensors from each layer
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if layer_idx >= len(attention_weights):
            layer_idx = 0
        
        attn = attention_weights[layer_idx][0, head_idx].detach().numpy()  # [seq_len, seq_len]
        
        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=self.feature_groups,
            y=self.feature_groups,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title=f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}',
            xaxis_title='Key Features',
            yaxis_title='Query Features',
            width=600,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_attention_evolution(
        self, 
        attention_history: List[List[torch.Tensor]], 
        time_steps: int = 50,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot how attention patterns evolve over time.
        
        Args:
            attention_history: List of attention weights over time
            time_steps: Number of time steps to show
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if not attention_history:
            return go.Figure()
        
        # Extract attention scores for each feature group over time
        attention_scores = []
        for t, attn_weights in enumerate(attention_history[:time_steps]):
            if attn_weights:
                # Average across layers and heads
                avg_attn = torch.stack(attn_weights).mean(dim=(0, 1))  # [batch_size, seq_len, seq_len]
                feature_attention = avg_attn[0].mean(dim=-1)  # [seq_len]
                attention_scores.append(feature_attention.detach().numpy())
        
        if not attention_scores:
            return go.Figure()
        
        attention_matrix = np.array(attention_scores).T  # [n_features, n_timesteps]
        
        fig = go.Figure()
        
        for i, feature_group in enumerate(self.feature_groups):
            fig.add_trace(go.Scatter(
                x=list(range(len(attention_scores))),
                y=attention_matrix[i],
                mode='lines+markers',
                name=feature_group,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Attention Evolution Over Time',
            xaxis_title='Time Steps',
            yaxis_title='Attention Weight',
            width=800,
            height=500,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_regime_transitions(
        self, 
        regime_history: List[str], 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize market regime transitions over time.
        
        Args:
            regime_history: List of regime names over time
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if not regime_history:
            return go.Figure()
        
        # Convert regime names to numeric values
        regime_mapping = {
            'bull': 0, 'bear': 1, 'volatile': 2, 'sideways': 3
        }
        regime_values = [regime_mapping.get(regime.lower(), 3) for regime in regime_history]
        
        fig = go.Figure()
        
        # Create regime timeline
        fig.add_trace(go.Scatter(
            x=list(range(len(regime_values))),
            y=regime_values,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=regime_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Regime",
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Bull', 'Bear', 'Volatile', 'Sideways']
                )
            ),
            line=dict(width=2),
            name='Market Regime'
        ))
        
        fig.update_layout(
            title='Market Regime Transitions',
            xaxis_title='Time Steps',
            yaxis_title='Regime',
            yaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=['Bull Market', 'Bear Market', 'Volatile Market', 'Sideways Market']
            ),
            width=800,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_portfolio_weights_with_attention(
        self, 
        weights_history: List[np.ndarray], 
        attention_history: List[List[torch.Tensor]],
        regime_history: List[str],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot portfolio weights alongside attention patterns and regime detection.
        
        Args:
            weights_history: List of portfolio weight arrays over time
            attention_history: List of attention weights over time
            regime_history: List of regime names over time
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if not weights_history:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Weights', 'Attention Patterns', 'Market Regime'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Portfolio weights
        weights_matrix = np.array(weights_history).T  # [n_assets, n_timesteps]
        for i, asset in enumerate(self.asset_names):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(weights_history))),
                    y=weights_matrix[i],
                    mode='lines',
                    name=asset,
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Attention patterns
        if attention_history:
            attention_scores = []
            for attn_weights in attention_history:
                if attn_weights:
                    avg_attn = torch.stack(attn_weights).mean(dim=(0, 1))
                    feature_attention = avg_attn[0].mean(dim=-1)
                    attention_scores.append(feature_attention.detach().numpy())
            
            if attention_scores:
                attention_matrix = np.array(attention_scores).T
                for i, feature_group in enumerate(self.feature_groups):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(attention_scores))),
                            y=attention_matrix[i],
                            mode='lines',
                            name=feature_group,
                            line=dict(width=1.5),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        # Market regime
        if regime_history:
            regime_mapping = {'bull': 0, 'bear': 1, 'volatile': 2, 'sideways': 3}
            regime_values = [regime_mapping.get(regime.lower(), 3) for regime in regime_history]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(regime_values))),
                    y=regime_values,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=regime_values,
                        colorscale='Viridis'
                    ),
                    name='Regime',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='Portfolio Allocation with Attention and Regime Analysis',
            height=800,
            width=1000
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        fig.update_yaxes(title_text="Attention", row=2, col=1)
        fig.update_yaxes(
            title_text="Regime", 
            tickvals=[0, 1, 2, 3],
            ticktext=['Bull', 'Bear', 'Volatile', 'Sideways'],
            row=3, col=1
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_attention_dashboard(
        self, 
        agent_outputs: List[Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive attention dashboard.
        
        Args:
            agent_outputs: List of agent output dictionaries
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        if not agent_outputs:
            return go.Figure()
        
        # Extract data
        weights_history = [output.get('weights', []) for output in agent_outputs]
        regime_history = [output.get('regime', 'sideways') for output in agent_outputs]
        attention_history = [output.get('attention_weights', []) for output in agent_outputs]
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Portfolio Allocation', 
                'Attention Heatmap',
                'Regime Distribution', 
                'Feature Importance'
            ),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Portfolio allocation
        if weights_history and all(len(w) > 0 for w in weights_history):
            weights_matrix = np.array(weights_history).T
            for i, asset in enumerate(self.asset_names):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(weights_history))),
                        y=weights_matrix[i],
                        mode='lines',
                        name=asset,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # Attention heatmap (latest)
        if attention_history and attention_history[-1]:
            latest_attn = attention_history[-1][0][0, 0].detach().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=latest_attn,
                    x=self.feature_groups,
                    y=self.feature_groups,
                    colorscale='Blues',
                    showscale=False
                ),
                row=1, col=2
            )
        
        # Regime distribution
        if regime_history:
            regime_counts = {}
            for regime in regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            fig.add_trace(
                go.Pie(
                    labels=list(regime_counts.keys()),
                    values=list(regime_counts.values()),
                    name="Regime Distribution"
                ),
                row=2, col=1
            )
        
        # Feature importance (average attention)
        if attention_history:
            all_attention = []
            for attn_weights in attention_history:
                if attn_weights:
                    avg_attn = torch.stack(attn_weights).mean(dim=(0, 1))
                    feature_attention = avg_attn[0].mean(dim=-1)
                    all_attention.append(feature_attention.detach().numpy())
            
            if all_attention:
                avg_feature_importance = np.mean(all_attention, axis=0)
                fig.add_trace(
                    go.Bar(
                        x=self.feature_groups,
                        y=avg_feature_importance,
                        name="Feature Importance"
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='Attention-Based SAC Portfolio Allocator Dashboard',
            height=800,
            width=1200,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class PerformanceVisualizer:
    """Visualizes portfolio performance and regime analysis."""
    
    def __init__(self, asset_names: List[str] = None):
        self.asset_names = asset_names or ['SPY', 'TLT', 'GLD', 'DBC', 'USO']
    
    def plot_performance_comparison(
        self, 
        results: Dict[str, Dict[str, float]], 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create performance comparison chart.
        
        Args:
            results: Dictionary with strategy results
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        strategies = list(results.keys())
        metrics = ['sharpe_ratio', 'max_drawdown', 'total_return']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Sharpe Ratio', 'Max Drawdown', 'Total Return')
        )
        
        for i, metric in enumerate(metrics):
            values = [results[strategy].get(metric, 0) for strategy in strategies]
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Performance Comparison Across Strategies',
            height=400,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_regime_performance(
        self, 
        regime_performance: Dict[str, Dict[str, float]], 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot performance metrics by market regime.
        
        Args:
            regime_performance: Performance metrics by regime
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        regimes = list(regime_performance.keys())
        metrics = ['sharpe_ratio', 'volatility', 'win_rate']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Sharpe Ratio by Regime', 'Volatility by Regime', 'Win Rate by Regime')
        )
        
        for i, metric in enumerate(metrics):
            values = [regime_performance[regime].get(metric, 0) for regime in regimes]
            
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Performance Analysis by Market Regime',
            height=400,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    
    # Sample data
    np.random.seed(42)
    n_timesteps = 100
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
    
    # Create visualizations
    visualizer = AttentionVisualizer()
    
    # Attention heatmap
    fig1 = visualizer.plot_attention_heatmap(attention_history[0])
    fig1.show()
    
    # Attention evolution
    fig2 = visualizer.plot_attention_evolution(attention_history)
    fig2.show()
    
    # Regime transitions
    fig3 = visualizer.plot_regime_transitions(regime_history)
    fig3.show()
    
    # Portfolio weights with attention
    fig4 = visualizer.plot_portfolio_weights_with_attention(
        weights_history, attention_history, regime_history
    )
    fig4.show()
    
    print("Sample visualizations created successfully!")


if __name__ == "__main__":
    create_sample_visualizations()
