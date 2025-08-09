import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


class ThreatHuntingVisualizer:
    """Visualizes threat hunting data and analysis results"""

    def __init__(self):
        self._configure_plot_style()

    def _configure_plot_style(self):
        """Configure default plot styling"""
        sns.set_style("darkgrid")
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 100,
            'savefig.dpi': 300
        })

    def plot_anomaly_timeline(self, events: List[Dict], anomalies: List[Dict]) -> None:
        """Plot anomaly detection results over time"""
        plt.figure(figsize=(15, 8))
        
        # Convert timestamps to datetime objects
        timestamps = [event['timestamp'] for event in events]
        
        # Plot all events
        plt.scatter(timestamps, [1]*len(events), alpha=0.3, color='blue', label='Normal Activity')
        
        # Highlight anomalies
        anomaly_timestamps = [a['timestamp'] for a in anomalies]
        plt.scatter(anomaly_timestamps, [1]*len(anomaly_timestamps), 
                   color='red', label='Anomalies', s=100, edgecolor='black')
        
        plt.yticks([])
        plt.title('Threat Activity Timeline with Anomaly Detection')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_risk_assessment(self, findings: List[Dict]) -> None:
        """Visualize risk assessment of detected threats"""
        if not findings:
            return

        # Extract threat data
        titles = [f.get('title', 'Unknown')[:30] + '...' for f in findings]
        severities = [f.get('severity', 'Unknown') for f in findings]
        confidence = [f.get('confidence', 0) * 100 for f in findings]
        
        # Map severities to colors
        severity_colors = {
            'Critical': 'red',
            'High': 'darkorange',
            'Medium': 'gold',
            'Low': 'green'
        }
        colors = [severity_colors.get(s, 'gray') for s in severities]
        
        plt.figure(figsize=(12, max(6, len(findings)*0.5)))
        bars = plt.barh(titles, confidence, color=colors)
        plt.xlabel('Confidence Score (%)')
        plt.title('Threat Findings Risk Assessment')
        plt.xlim(0, 100)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 2, bar.get_y() + 0.2, f'{width:.1f}%', va='center')
        
        plt.tight_layout()
        plt.show()

    def plot_network_activity(self, network_data: List[Dict]) -> None:
        """Visualize network activity patterns"""
        # Group by protocol
        protocols = {}
        for event in network_data:
            proto = event['protocol']
            protocols[proto] = protocols.get(proto, 0) + 1
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(protocols.values(), labels=protocols.keys(), autopct='%1.1f%%')
        plt.title('Network Activity by Protocol')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_attack_timeline(self, findings: List[Dict]) -> None:
        """Create a detailed attack timeline visualization"""
        if not findings:
            return

        plt.figure(figsize=(15, 6))
        
        # Create timeline for each finding
        for i, finding in enumerate(findings):
            # Get timestamps from affected assets
            timestamps = [event['timestamp'] for event in finding.get('affected_assets', [])]
            if timestamps:
                plt.scatter(timestamps, [i]*len(timestamps), 
                          label=finding.get('title', f'Finding {i+1}')[:30] + '...')
        
        plt.yticks(range(len(findings)), [f'Finding {i+1}' for i in range(len(findings))])
        plt.title('Attack Timeline Analysis')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.show()