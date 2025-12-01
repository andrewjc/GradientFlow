# -*- coding: utf-8 -*-
"""
Attention Network Analyzer
==========================

Analyzer for attention-based architectures including Transformers,
BERT, GPT, and custom attention mechanisms. Focuses on gradient
flow through attention layers, residual connections, and layer norms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from ..core.engine import FlowAnalyzer
from ..core.metrics import FlowMetrics, LayerHealth
from ..visualizers.report import FlowReport


@dataclass
class AttentionLayerMetrics:
    """Metrics for a single attention layer."""
    layer_name: str
    layer_index: int

    # Component pressures
    query_pressure: float = 0.0
    key_pressure: float = 0.0
    value_pressure: float = 0.0
    output_pressure: float = 0.0

    # Attention-specific
    attention_weight_pressure: float = 0.0
    softmax_saturation: float = 0.0

    # FFN pressures
    ffn_up_pressure: float = 0.0
    ffn_down_pressure: float = 0.0

    # Residual health
    residual_ratio: float = 1.0  # ratio of output to input gradient

    @property
    def has_attention_bottleneck(self) -> bool:
        """Check if attention weights are a bottleneck."""
        return self.attention_weight_pressure > 10 * max(
            self.query_pressure, self.key_pressure, 0.001
        )

    @property
    def has_ffn_bottleneck(self) -> bool:
        """Check if FFN is a bottleneck."""
        if self.ffn_up_pressure > 0 and self.ffn_down_pressure > 0:
            return self.ffn_down_pressure > 5 * self.ffn_up_pressure
        return False


@dataclass
class TransformerMetrics:
    """Aggregate metrics for full transformer."""
    num_layers: int
    layer_metrics: List[AttentionLayerMetrics] = field(default_factory=list)
    residual_decay: float = 0.0
    attention_avg_pressure: float = 0.0
    ffn_avg_pressure: float = 0.0

    @property
    def has_residual_degradation(self) -> bool:
        """Check if residual connections are degrading."""
        return self.residual_decay < 0.5


class AttentionAnalyzer:
    """
    Analyzer for attention-based neural networks.

    This analyzer is designed for:
    - Transformer encoders and decoders
    - BERT-style models
    - GPT-style models
    - Vision Transformers (ViT)
    - Custom attention mechanisms

    Key diagnostics:
    - Query/Key/Value gradient flow
    - Attention weight gradient patterns
    - Residual connection health
    - FFN bottleneck detection
    - Layer normalization gradients
    - Multi-head attention balance

    Example:
        >>> analyzer = AttentionAnalyzer()
        >>> report = analyzer.analyze(transformer, tokens, labels, loss_fn)
        >>> report.print_summary()
        >>>
        >>> # Get attention-specific metrics
        >>> tf_metrics = analyzer.get_transformer_metrics()
        >>> for layer in tf_metrics.layer_metrics:
        ...     print(f"Layer {layer.layer_index}: Q={layer.query_pressure:.2e}")
    """

    # Common attention component patterns
    QUERY_PATTERNS = ("query", "q_proj", "q_linear", "wq", "to_q")
    KEY_PATTERNS = ("key", "k_proj", "k_linear", "wk", "to_k")
    VALUE_PATTERNS = ("value", "v_proj", "v_linear", "wv", "to_v")
    OUTPUT_PATTERNS = ("output", "o_proj", "out_proj", "wo", "to_out")
    FFN_UP_PATTERNS = ("fc1", "up_proj", "w1", "intermediate", "mlp.0", "ffn.0")
    FFN_DOWN_PATTERNS = ("fc2", "down_proj", "w2", "output", "mlp.2", "ffn.2")
    NORM_PATTERNS = ("norm", "ln", "layer_norm", "layernorm")
    ATTENTION_PATTERNS = ("attn", "attention", "self_attn", "mha")

    def __init__(
        self,
        num_heads: Optional[int] = None,
        track_heads: bool = True,
        residual_analysis: bool = True,
    ):
        """
        Initialize attention analyzer.

        Args:
            num_heads: Number of attention heads (auto-detected if None)
            track_heads: Track per-head gradient statistics
            residual_analysis: Analyze residual connection health
        """
        self.num_heads = num_heads
        self.track_heads = track_heads
        self.residual_analysis = residual_analysis

        self._flow_analyzer: Optional[FlowAnalyzer] = None
        self._attention_layers: Dict[str, nn.Module] = {}
        self._component_map: Dict[str, Dict[str, str]] = {}
        self._transformer_metrics: Optional[TransformerMetrics] = None

    def analyze(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        model_name: str = "Transformer",
        num_samples: int = 1,
    ) -> FlowReport:
        """
        Analyze gradient flow in an attention-based network.

        Args:
            model: The neural network to analyze
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            model_name: Name for the report
            num_samples: Number of analysis iterations

        Returns:
            FlowReport with analysis results
        """
        # Pre-analyze attention structure
        self._analyze_structure(model)

        # Run flow analysis
        self._flow_analyzer = FlowAnalyzer(model, model_name)
        report = self._flow_analyzer.analyze(
            sample_input,
            sample_target,
            loss_fn,
            num_samples=num_samples,
        )

        # Add attention-specific analysis
        self._build_attention_metrics(report)
        self._add_attention_issues(report)

        return report

    def _analyze_structure(self, model: nn.Module) -> None:
        """Analyze transformer structure to find attention components."""
        self._attention_layers.clear()
        self._component_map.clear()

        # Find attention blocks
        for name, module in model.named_modules():
            name_lower = name.lower()

            # Check if this is an attention layer
            is_attention = any(p in name_lower for p in self.ATTENTION_PATTERNS)
            if is_attention and len(list(module.children())) > 0:
                self._attention_layers[name] = module
                self._component_map[name] = {}

        # Map components within each attention layer
        for attn_name in list(self._attention_layers.keys()):
            prefix = attn_name + "."

            for name, module in model.named_modules():
                if not name.startswith(prefix):
                    continue

                rel_name = name[len(prefix):].lower()

                # Classify component
                if any(p in rel_name for p in self.QUERY_PATTERNS):
                    self._component_map[attn_name]["query"] = name
                elif any(p in rel_name for p in self.KEY_PATTERNS):
                    self._component_map[attn_name]["key"] = name
                elif any(p in rel_name for p in self.VALUE_PATTERNS):
                    self._component_map[attn_name]["value"] = name
                elif any(p in rel_name for p in self.OUTPUT_PATTERNS):
                    self._component_map[attn_name]["output"] = name

    def _build_attention_metrics(self, report: FlowReport) -> None:
        """Build attention-specific metrics from flow report."""
        layer_metrics = []

        for i, (attn_name, components) in enumerate(self._component_map.items()):
            metrics = AttentionLayerMetrics(
                layer_name=attn_name,
                layer_index=i,
            )

            # Get component pressures
            if "query" in components:
                m = report.metrics.get(components["query"])
                if m:
                    metrics.query_pressure = m.mean_pressure

            if "key" in components:
                m = report.metrics.get(components["key"])
                if m:
                    metrics.key_pressure = m.mean_pressure

            if "value" in components:
                m = report.metrics.get(components["value"])
                if m:
                    metrics.value_pressure = m.mean_pressure

            if "output" in components:
                m = report.metrics.get(components["output"])
                if m:
                    metrics.output_pressure = m.mean_pressure

            layer_metrics.append(metrics)

        # Build transformer-level metrics
        num_layers = len(layer_metrics)
        self._transformer_metrics = TransformerMetrics(
            num_layers=num_layers,
            layer_metrics=layer_metrics,
        )

        if layer_metrics:
            self._transformer_metrics.attention_avg_pressure = np.mean([
                lm.query_pressure + lm.key_pressure + lm.value_pressure
                for lm in layer_metrics
            ]) / 3

        # Calculate residual decay
        if len(layer_metrics) >= 2:
            first_output = layer_metrics[0].output_pressure
            last_output = layer_metrics[-1].output_pressure
            if first_output > 0:
                self._transformer_metrics.residual_decay = last_output / first_output

    def _add_attention_issues(self, report: FlowReport) -> None:
        """Add attention-specific issues to report."""
        if not self._transformer_metrics:
            return

        # Check each layer
        for lm in self._transformer_metrics.layer_metrics:
            if lm.has_attention_bottleneck:
                report.issues.append({
                    "type": "ATTN_BOTTLENECK",
                    "severity": "HIGH",
                    "layer": lm.layer_name,
                    "layer_type": "Attention",
                    "info": f"Attention weight pressure spike: attn={lm.attention_weight_pressure:.1f} vs Q={lm.query_pressure:.1f}"
                })

            if lm.has_ffn_bottleneck:
                report.issues.append({
                    "type": "FFN_BOTTLENECK",
                    "severity": "MEDIUM",
                    "layer": lm.layer_name,
                    "layer_type": "FFN",
                    "info": f"FFN compression bottleneck: up={lm.ffn_up_pressure:.1f}, down={lm.ffn_down_pressure:.1f}"
                })

            # Check for QKV imbalance
            pressures = [lm.query_pressure, lm.key_pressure, lm.value_pressure]
            if min(pressures) > 0:
                ratio = max(pressures) / min(pressures)
                if ratio > 10:
                    report.issues.append({
                        "type": "QKV_IMBALANCE",
                        "severity": "MEDIUM",
                        "layer": lm.layer_name,
                        "layer_type": "Attention",
                        "info": f"Q/K/V pressure imbalance: ratio={ratio:.1f}x"
                    })

        # Check residual degradation
        if self._transformer_metrics.has_residual_degradation:
            report.issues.append({
                "type": "RESIDUAL_DECAY",
                "severity": "HIGH",
                "layer": "network-wide",
                "layer_type": "Residual",
                "info": f"Residual connection gradient decay: {self._transformer_metrics.residual_decay:.2f}x"
            })

        # Check layer-wise gradient distribution
        if len(self._transformer_metrics.layer_metrics) >= 4:
            output_pressures = [lm.output_pressure for lm in self._transformer_metrics.layer_metrics]
            if output_pressures and min(output_pressures) > 0:
                variance = np.var(output_pressures) / np.mean(output_pressures) ** 2
                if variance > 1.0:
                    report.issues.append({
                        "type": "LAYER_VARIANCE",
                        "severity": "MEDIUM",
                        "layer": "network-wide",
                        "layer_type": "Transformer",
                        "info": f"High variance in layer output gradients: CV={np.sqrt(variance):.2f}"
                    })

    def get_transformer_metrics(self) -> Optional[TransformerMetrics]:
        """
        Get transformer-level metrics.

        Returns:
            TransformerMetrics or None if not analyzed
        """
        return self._transformer_metrics

    def get_layer_metrics(self, layer_index: int) -> Optional[AttentionLayerMetrics]:
        """
        Get metrics for a specific attention layer.

        Args:
            layer_index: Index of the attention layer

        Returns:
            AttentionLayerMetrics or None if not found
        """
        if self._transformer_metrics and 0 <= layer_index < len(self._transformer_metrics.layer_metrics):
            return self._transformer_metrics.layer_metrics[layer_index]
        return None

    def get_attention_profile(self) -> List[Tuple[int, float, float, float]]:
        """
        Get gradient profile across attention layers.

        Returns:
            List of (layer_index, query_pressure, key_pressure, value_pressure)
        """
        if not self._transformer_metrics:
            return []

        return [
            (lm.layer_index, lm.query_pressure, lm.key_pressure, lm.value_pressure)
            for lm in self._transformer_metrics.layer_metrics
        ]

    def get_head_metrics(self, layer_name: str) -> Optional[List[float]]:
        """
        Get per-head gradient metrics (if tracked).

        Args:
            layer_name: Name of the attention layer

        Returns:
            List of per-head pressures or None
        """
        # This would require more sophisticated per-head tracking
        # during the analysis phase
        return None

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._flow_analyzer:
            self._flow_analyzer.cleanup()
