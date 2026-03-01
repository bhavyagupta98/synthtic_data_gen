"""Visualization and plotting for metrics."""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


class MetricsVisualizer:
    """Visualize and plot metrics from evaluation."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.metrics_history = []
    
    def add_metrics(self, metrics_dict: Dict):
        """Add metrics snapshot to history."""
        self.metrics_history.append(metrics_dict)
    
    def plot_summary(self, output_dir: str = 'results'):
        """
        Create visualization plots of metrics.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed. Install with: pip install matplotlib")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        if not self.metrics_history:
            print("No metrics to plot")
            return
        
        latest = self.metrics_history[-1]
        
        if 'routes' not in latest:
            print("Invalid metrics format")
            return
        
        routes = latest['routes']
        route_ids = list(routes.keys())
        rc_values = [routes[rid]['rc'] for rid in route_ids]
        ds_values = [routes[rid]['ds'] for rid in route_ids]
        collisions = [routes[rid]['collisions'] for rid in route_ids]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Autonomous Driving Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Route Completion (RC)
        ax = axes[0, 0]
        colors = ['green' if rc >= 100 else 'orange' if rc >= 80 else 'red' for rc in rc_values]
        ax.barh(route_ids, rc_values, color=colors, alpha=0.7)
        ax.set_xlabel('Route Completion (%)')
        ax.set_title('Route Completion by Route')
        ax.set_xlim(0, 110)
        for i, v in enumerate(rc_values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Driving Score (DS)
        ax = axes[0, 1]
        colors = ['green' if ds >= 80 else 'orange' if ds >= 50 else 'red' for ds in ds_values]
        ax.barh(route_ids, ds_values, color=colors, alpha=0.7)
        ax.set_xlabel('Driving Score')
        ax.set_title('Driving Score by Route')
        ax.set_xlim(0, 110)
        for i, v in enumerate(ds_values):
            ax.text(v + 1, i, f'{v:.1f}', va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Collisions
        ax = axes[1, 0]
        colors = ['green' if c == 0 else 'red' for c in collisions]
        ax.bar(range(len(route_ids)), collisions, color=colors, alpha=0.7)
        ax.set_xticks(range(len(route_ids)))
        ax.set_xticklabels(route_ids, rotation=45, ha='right')
        ax.set_ylabel('Number of Collisions')
        ax.set_title('Collisions per Route')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        avg_rc = latest.get('avg_route_completion', 0)
        avg_ds = latest.get('avg_driving_score', 0)
        success_rate = (latest.get('successful_routes', 0) / latest.get('routes_tested', 1)) * 100
        total_collisions = latest.get('total_collisions', 0)
        
        summary_text = f"""
EVALUATION SUMMARY

Routes Tested: {latest.get('routes_tested', 0)}
Successful: {latest.get('successful_routes', 0)}

Avg Route Completion: {avg_rc:.1f}%
Avg Driving Score: {avg_ds:.1f}/100
Success Rate: {success_rate:.1f}%

Total Collisions: {total_collisions}
Total Violations: {latest.get('total_violations', 0)}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_path / 'metrics_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {output_file}")
        
        plt.close()
    
    def save_detailed_report(self, output_dir: str = 'results', filename: str = 'evaluation_report.txt'):
        """Save detailed text report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        
        report = []
        report.append("=" * 80)
        report.append("AUTONOMOUS DRIVING EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Routes Tested:         {latest.get('routes_tested', 0)}")
        report.append(f"Successful Routes:     {latest.get('successful_routes', 0)}")
        report.append(f"Avg Route Completion:  {latest.get('avg_route_completion', 0):.1f}%")
        report.append(f"Avg Driving Score:     {latest.get('avg_driving_score', 0):.1f}/100")
        report.append(f"Total Collisions:      {latest.get('total_collisions', 0)}")
        report.append(f"Total Violations:      {latest.get('total_violations', 0)}")
        report.append("")
        
        if 'routes' in latest:
            report.append("DETAILED ROUTE BREAKDOWN")
            report.append("-" * 80)
            report.append(f"{'Route ID':<25} {'RC %':<12} {'DS/100':<12} {'Collisions':<12} {'Violations':<12}")
            report.append("-" * 80)
            
            for rid, data in sorted(latest['routes'].items()):
                violations = data.get('violations', 0)
                report.append(
                    f"{rid:<25} {data['rc']:<12.1f} {data['ds']:<12.1f} "
                    f"{data['collisions']:<12} {violations:<12}"
                )
            
            report.append("-" * 80)
        
        report.append("")
        report.append("METRICS DEFINITIONS")
        report.append("-" * 80)
        report.append("RC (Route Completion): Percentage of route distance successfully completed")
        report.append("DS (Driving Score): 0-100 score based on performance and infractions")
        report.append("  - Collision: -60 points")
        report.append("  - Lane departure: -30 points")
        report.append("  - Speed violation: -4 points")
        report.append("Violations: Lane departures + Speed violations combined")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        output_file = output_path / filename
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"Detailed report saved to: {output_file}")
        
        return report_text
    
    def export_json(self, output_dir: str = 'results', filename: str = 'metrics.json'):
        """Export metrics as JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics_history:
            return
        
        output_file = output_path / filename
        with open(output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics exported to: {output_file}")
    
    def print_summary(self, summary_dict: Dict):
        """Print summary to console."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Routes Tested:        {summary_dict.get('routes_tested', 0)}")
        print(f"Successful Routes:    {summary_dict.get('successful_routes', 0)}")
        print(f"Success Rate:         {(summary_dict.get('successful_routes', 0) / max(1, summary_dict.get('routes_tested', 1)) * 100):.1f}%")
        print(f"Avg Route Completion: {summary_dict.get('avg_route_completion', 0):.1f}%")
        print(f"Avg Driving Score:    {summary_dict.get('avg_driving_score', 0):.1f}/100")
        print(f"Total Collisions:     {summary_dict.get('total_collisions', 0)}")
        print(f"Total Violations:     {summary_dict.get('total_violations', 0)}")
        print("=" * 80 + "\n")
