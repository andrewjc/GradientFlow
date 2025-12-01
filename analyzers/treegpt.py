# -*- coding: utf-8 -*-
"""
TreeGPT-specific Analyzer
==========================

Specialized analyzer for TreeGPT language models with tree routing and memory.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class TreeRoutingStats:
    """Statistics about tree routing patterns."""
    leaf_distribution_entropy: float
    routing_diversity: float
    routing_collapse_risk: str
    avg_path_length: float
    leaf_utilization: float


@dataclass
class MemorySystemStats:
    """Statistics about memory system."""
    slot_utilization: float
    memory_diversity: float
    temporal_stability: float
    compression_quality: float
    collapse_risk: str


class TreeGPTAnalyzer:
    """
    Specialized analyzer for TreeGPT language models.

    Analyzes:
    - Tree routing patterns and leaf utilization
    - Memory slot usage and diversity
    - Causal masking effectiveness
    - RMSNorm statistics

    Usage:
        >>> analyzer = TreeGPTAnalyzer()
        >>> routing_stats = analyzer.analyze_tree_routing(leaf_ids_list)
        >>> memory_stats = analyzer.analyze_memory_system(memory_states)
    """

    def __init__(self):
        pass

    def analyze_tree_routing(
        self,
        leaf_ids: List[torch.Tensor],
        num_leaves: int
    ) -> TreeRoutingStats:
        """
        Analyze tree routing patterns.

        Args:
            leaf_ids: List of routing decisions [T x (BH, L)]
            num_leaves: Total number of leaves

        Returns:
            TreeRoutingStats with routing analysis
        """
        if not leaf_ids:
            return TreeRoutingStats(
                leaf_distribution_entropy=0.0,
                routing_diversity=0.0,
                routing_collapse_risk="UNKNOWN",
                avg_path_length=0.0,
                leaf_utilization=0.0
            )

        # Concatenate all routing decisions
        all_ids = torch.cat([ids.flatten() for ids in leaf_ids])

        # Compute leaf usage distribution
        leaf_counts = torch.bincount(all_ids, minlength=num_leaves).float()
        leaf_dist = leaf_counts / (leaf_counts.sum() + 1e-12)

        # Entropy
        entropy = -(leaf_dist * torch.log(leaf_dist + 1e-12)).sum().item()
        max_entropy = np.log(num_leaves)
        normalized_entropy = entropy / max_entropy

        # Utilization
        active_leaves = (leaf_counts > 0).sum().item()
        leaf_utilization = active_leaves / num_leaves

        # Diversity (inverse of concentration)
        routing_diversity = 1.0 - (leaf_dist.max().item())

        # Path length (log2 of num_leaves)
        avg_path_length = np.log2(num_leaves)

        # Collapse risk
        if leaf_utilization < 0.3 or normalized_entropy < 0.3:
            routing_collapse_risk = "HIGH"
        elif leaf_utilization < 0.5 or normalized_entropy < 0.5:
            routing_collapse_risk = "MEDIUM"
        else:
            routing_collapse_risk = "LOW"

        return TreeRoutingStats(
            leaf_distribution_entropy=normalized_entropy,
            routing_diversity=routing_diversity,
            routing_collapse_risk=routing_collapse_risk,
            avg_path_length=avg_path_length,
            leaf_utilization=leaf_utilization
        )

    def analyze_memory_system(
        self,
        memory_states: List[torch.Tensor]
    ) -> MemorySystemStats:
        """
        Analyze memory slot system.

        Args:
            memory_states: List of memory tensors [T x (B, M, dm)]

        Returns:
            MemorySystemStats with memory analysis
        """
        if not memory_states:
            return MemorySystemStats(
                slot_utilization=0.0,
                memory_diversity=0.0,
                temporal_stability=0.0,
                compression_quality=0.0,
                collapse_risk="UNKNOWN"
            )

        # Stack memory states
        mem_tensor = torch.stack(memory_states)  # [T, B, M, dm]
        T, B, M, dm = mem_tensor.shape

        # Slot utilization (based on norms)
        norms = torch.linalg.norm(mem_tensor, dim=-1)  # [T, B, M]
        active_slots = (norms > 0.1).float().mean().item()

        # Diversity (pairwise distances)
        final_mem = mem_tensor[-1, 0]  # [M, dm] - first batch element
        if M > 1:
            pairwise_dists = torch.pdist(final_mem)
            memory_diversity = pairwise_dists.mean().item()
        else:
            memory_diversity = 0.0

        # Temporal stability
        if T > 1:
            changes = mem_tensor[1:] - mem_tensor[:-1]
            change_norm = torch.linalg.norm(changes, dim=-1).mean().item()
            avg_norm = norms.mean().item()
            temporal_stability = 1.0 - (change_norm / (avg_norm + 1e-8))
            temporal_stability = max(0.0, min(1.0, temporal_stability))
        else:
            temporal_stability = 1.0

        # Compression quality (approximated by norm consistency)
        norm_std = norms.std().item()
        norm_mean = norms.mean().item()
        compression_quality = 1.0 - (norm_std / (norm_mean + 1e-8))
        compression_quality = max(0.0, min(1.0, compression_quality))

        # Collapse risk
        if memory_diversity < 0.01 or active_slots < 0.3:
            collapse_risk = "HIGH"
        elif memory_diversity < 0.1 or active_slots < 0.5:
            collapse_risk = "MEDIUM"
        else:
            collapse_risk = "LOW"

        return MemorySystemStats(
            slot_utilization=active_slots,
            memory_diversity=memory_diversity,
            temporal_stability=temporal_stability,
            compression_quality=compression_quality,
            collapse_risk=collapse_risk
        )
