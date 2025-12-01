# -*- coding: utf-8 -*-
"""
Flow Report Generation
======================

Central report class that aggregates analysis results and provides
multiple output formats (console, HTML, JSON, etc.).
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.metrics import FlowMetrics, LayerHealth, HealthStatus


@dataclass
class FlowReport:
    """
    Comprehensive gradient flow analysis report.

    This class aggregates metrics, health assessments, and issues
    from a gradient flow analysis session and provides multiple
    output formats.

    Attributes:
        model_name: Name of the analyzed model
        metrics: Dictionary of FlowMetrics per layer
        issues: List of detected issues
        health: Dictionary of LayerHealth per layer
        config: Analysis configuration used
    """

    model_name: str
    metrics: Dict[str, FlowMetrics]
    issues: List[Dict[str, Any]]
    health: Dict[str, LayerHealth]
    config: Any

    # Computed summary stats
    _summary: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def summary(self) -> Dict[str, Any]:
        """Compute and cache summary statistics."""
        if self._summary is not None:
            return self._summary

        health_scores = [h.score for h in self.health.values()]

        self._summary = {
            "model_name": self.model_name,
            "total_layers": len(self.metrics),
            "avg_health": np.mean(health_scores) if health_scores else 0.0,
            "min_health": np.min(health_scores) if health_scores else 0.0,
            "max_health": np.max(health_scores) if health_scores else 0.0,
            "healthy_layers": sum(1 for h in health_scores if h >= 75),
            "critical_issues": sum(1 for i in self.issues if i["severity"] == "CRITICAL"),
            "high_issues": sum(1 for i in self.issues if i["severity"] == "HIGH"),
            "total_issues": len(self.issues),
            "status": self._overall_status(),
        }

        return self._summary

    def _overall_status(self) -> str:
        """Determine overall model health status."""
        critical = sum(1 for i in self.issues if i["severity"] == "CRITICAL")
        high = sum(1 for i in self.issues if i["severity"] == "HIGH")

        if critical > 0:
            return "CRITICAL"
        elif high > 0:
            return "WARNING"
        elif self.issues:
            return "ATTENTION"
        else:
            return "HEALTHY"

    def get_worst_layers(self, n: int = 10) -> List[LayerHealth]:
        """Get the n layers with lowest health scores."""
        sorted_health = sorted(
            self.health.values(),
            key=lambda h: h.score
        )
        return sorted_health[:n]

    def get_best_layers(self, n: int = 10) -> List[LayerHealth]:
        """Get the n layers with highest health scores."""
        sorted_health = sorted(
            self.health.values(),
            key=lambda h: h.score,
            reverse=True
        )
        return sorted_health[:n]

    def get_issues_by_type(self, issue_type: str) -> List[Dict[str, Any]]:
        """Get all issues of a specific type."""
        return [i for i in self.issues if i["type"] == issue_type]

    def get_issues_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get all issues of a specific severity."""
        return [i for i in self.issues if i["severity"] == severity]

    def get_layer_metrics(self, name: str) -> Optional[FlowMetrics]:
        """Get metrics for a specific layer."""
        return self.metrics.get(name)

    def get_layer_health(self, name: str) -> Optional[LayerHealth]:
        """Get health for a specific layer."""
        return self.health.get(name)

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Export report as dictionary."""
        return {
            "summary": self.summary,
            "metrics": {
                name: m.to_dict()
                for name, m in self.metrics.items()
            },
            "health": {
                name: h.to_dict()
                for name, h in self.health.items()
            },
            "issues": self.issues,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str) -> None:
        """Save report as JSON file."""
        Path(path).write_text(self.to_json(), encoding='utf-8')

    def save_text(self, path: str) -> None:
        """Save report as plain text file."""
        from .console import ConsoleReporter
        reporter = ConsoleReporter(use_colors=False)
        text = reporter.format_report(self)
        Path(path).write_text(text, encoding='utf-8')

    def save_html(self, path: str) -> None:
        """Save report as HTML file."""
        from .html import HTMLReportGenerator
        generator = HTMLReportGenerator()
        html = generator.generate(self)
        Path(path).write_text(html, encoding='utf-8')

    # =========================================================================
    # Quick Display Methods
    # =========================================================================

    def print_summary(self) -> None:
        """Print summary to console."""
        from .console import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.print_summary(self)

    def print_issues(self) -> None:
        """Print issues to console."""
        from .console import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.print_issues(self)

    def print_layers(self, n: int = 20) -> None:
        """Print layer-by-layer summary to console."""
        from .console import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.print_layer_summary(self, n)

    def print_recommendations(self) -> None:
        """Print recommendations to console."""
        from .console import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.print_recommendations(self)

    def print_full(self) -> None:
        """Print full report to console."""
        from .console import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.print_full_report(self)
