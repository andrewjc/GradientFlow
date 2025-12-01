# -*- coding: utf-8 -*-
"""
Console Output Formatting
=========================

Beautiful console output for gradient flow reports with ANSI colors.
"""

from typing import Optional
from ..core.metrics import HealthStatus


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'


class ConsoleReporter:
    """
    Formats gradient flow reports for console output.

    Supports both colored and plain text output.
    """

    def __init__(self, use_colors: bool = True, width: int = 100):
        """
        Initialize console reporter.

        Args:
            use_colors: Whether to use ANSI colors
            width: Target width for formatting
        """
        self.use_colors = use_colors
        self.width = width

    def _c(self, color: str, text: str) -> str:
        """Apply color if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.ENDC}"
        return text

    def _header(self, text: str) -> str:
        """Format a section header."""
        line = "=" * self.width
        return f"\n{self._c(Colors.HEADER, line)}\n {text}\n{self._c(Colors.HEADER, line)}\n"

    def _subheader(self, text: str) -> str:
        """Format a subsection header."""
        line = "-" * self.width
        return f"\n{self._c(Colors.CYAN, text)}\n{line}\n"

    def _health_color(self, score: float) -> str:
        """Get color based on health score."""
        if score >= 90:
            return Colors.GREEN
        elif score >= 75:
            return Colors.GREEN
        elif score >= 50:
            return Colors.WARNING
        else:
            return Colors.FAIL

    def _severity_color(self, severity: str) -> str:
        """Get color based on severity."""
        if severity == "CRITICAL":
            return Colors.FAIL
        elif severity == "HIGH":
            return Colors.WARNING
        elif severity == "MEDIUM":
            return Colors.CYAN
        return Colors.DIM

    def format_report(self, report) -> str:
        """Format full report as string."""
        lines = []

        # Title
        lines.append(self._header(f"GRADIENT FLOW ANALYSIS - {report.model_name}"))

        # Summary
        s = report.summary
        status_color = Colors.GREEN if s["status"] == "HEALTHY" else (
            Colors.WARNING if s["status"] == "WARNING" else Colors.FAIL
        )

        lines.append(f"\n{self._c(Colors.BOLD, 'Summary:')}")
        lines.append(f"  Total layers analyzed: {s['total_layers']}")
        avg_health = s['avg_health']
        lines.append(f"  Average health: {self._c(self._health_color(avg_health), f'{avg_health:.1f}%')}")
        lines.append(f"  Healthy layers (>=75%): {s['healthy_layers']}/{s['total_layers']}")
        lines.append(f"  Issues found: {s['total_issues']} ({s['critical_issues']} critical, {s['high_issues']} high)")
        lines.append(f"  Overall status: {self._c(status_color, s['status'])}")

        # Worst layers
        lines.append(self._subheader("Layers Requiring Attention (sorted by health)"))
        worst = report.get_worst_layers(15)

        header = f"{'Layer':<45} {'Type':<18} {'Pressure':<12} {'Health':<10}"
        lines.append(header)
        lines.append("-" * len(header))

        for h in worst:
            m = report.metrics.get(h.name)
            if m:
                pressure = f"{m.mean_pressure:.2e}"
                color = self._health_color(h.score)
                name_padded = f"{h.name[:45]:<45}"
                score_str = f"{h.score:.1f}%"
                lines.append(
                    f"{self._c(color, name_padded)} "
                    f"{h.layer_type:<18} "
                    f"{pressure:<12} "
                    f"{self._c(color, score_str)}"
                )

        # Issues
        if report.issues:
            lines.append(self._subheader(f"Issues Detected ({len(report.issues)} total)"))

            header = f"{'Type':<12} {'Severity':<10} {'Layer':<40} {'Details'}"
            lines.append(header)
            lines.append("-" * self.width)

            for issue in sorted(report.issues, key=lambda x: (
                0 if x["severity"] == "CRITICAL" else
                1 if x["severity"] == "HIGH" else
                2 if x["severity"] == "MEDIUM" else 3
            )):
                color = self._severity_color(issue["severity"])
                type_padded = f"{issue['type']:<12}"
                sev_padded = f"{issue['severity']:<10}"
                layer_padded = f"{issue['layer'][:40]:<40}"
                lines.append(
                    f"{self._c(color, type_padded)} "
                    f"{self._c(color, sev_padded)} "
                    f"{layer_padded} "
                    f"{issue['info']}"
                )

        # Recommendations
        all_recs = set()
        for h in report.health.values():
            all_recs.update(h.recommendations)

        if all_recs:
            lines.append(self._subheader("Recommendations"))
            for i, rec in enumerate(sorted(all_recs), 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    def print_summary(self, report) -> None:
        """Print summary section."""
        print(self._header(f"GRADIENT FLOW ANALYSIS - {report.model_name}"))

        s = report.summary
        status_color = Colors.GREEN if s["status"] == "HEALTHY" else (
            Colors.WARNING if s["status"] in ["WARNING", "ATTENTION"] else Colors.FAIL
        )

        print(f"\n{self._c(Colors.BOLD, 'Summary:')}")
        print(f"  Total layers analyzed: {s['total_layers']}")
        avg_health = s['avg_health']
        print(f"  Average health: {self._c(self._health_color(avg_health), f'{avg_health:.1f}%')}")
        print(f"  Healthy layers (>=75%): {s['healthy_layers']}/{s['total_layers']}")
        print(f"  Issues found: {s['total_issues']} ({s['critical_issues']} critical)")
        print(f"  Overall status: {self._c(status_color, s['status'])}")

    def print_layer_summary(self, report, n: int = 20) -> None:
        """Print layer-by-layer summary."""
        print(self._subheader(f"Layer Health Summary (showing {n} layers)"))

        # Sort by health score
        sorted_health = sorted(report.health.values(), key=lambda h: h.score)

        header = f"{'Layer':<45} {'Type':<18} {'Mean':<12} {'Max':<12} {'Health':<8}"
        print(header)
        print("-" * len(header))

        for h in sorted_health[:n]:
            m = report.metrics.get(h.name)
            if m:
                mean_str = f"{m.mean_pressure:.2e}"
                max_str = f"{m.max_pressure:.2e}"
                color = self._health_color(h.score)
                name_padded = f"{h.name[:45]:<45}"
                score_str = f"{h.score:.1f}%"
                print(
                    f"{self._c(color, name_padded)} "
                    f"{h.layer_type:<18} "
                    f"{mean_str:<12} "
                    f"{max_str:<12} "
                    f"{self._c(color, score_str)}"
                )

    def print_issues(self, report) -> None:
        """Print detected issues."""
        if not report.issues:
            print(self._c(Colors.GREEN, "\nNo issues detected! Gradient flow looks healthy."))
            return

        print(self._subheader(f"Issues Detected ({len(report.issues)} total)"))

        header = f"{'Type':<12} {'Severity':<10} {'Layer Type':<18} {'Layer':<35} {'Details'}"
        print(header)
        print("-" * self.width)

        for issue in sorted(report.issues, key=lambda x: (
            0 if x["severity"] == "CRITICAL" else
            1 if x["severity"] == "HIGH" else 2
        )):
            color = self._severity_color(issue["severity"])
            type_padded = f"{issue['type']:<12}"
            sev_padded = f"{issue['severity']:<10}"
            layer_type_padded = f"{issue['layer_type']:<18}"
            layer_padded = f"{issue['layer'][:35]:<35}"
            print(
                f"{self._c(color, type_padded)} "
                f"{self._c(color, sev_padded)} "
                f"{layer_type_padded} "
                f"{layer_padded} "
                f"{issue['info']}"
            )

    def print_recommendations(self, report) -> None:
        """Print recommendations."""
        all_recs = {}
        for h in report.health.values():
            for rec in h.recommendations:
                if rec not in all_recs:
                    all_recs[rec] = []
                all_recs[rec].append(h.name)

        if not all_recs:
            print(self._c(Colors.GREEN, "\nNo recommendations - model looks healthy!"))
            return

        print(self._subheader("Recommendations"))

        for i, (rec, layers) in enumerate(sorted(all_recs.items()), 1):
            print(f"\n  {self._c(Colors.WARNING, f'{i}.')} {rec}")
            print(f"     Affects: {len(layers)} layer(s)")

    def print_full_report(self, report) -> None:
        """Print complete report."""
        print(self.format_report(report))
