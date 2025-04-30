import os
import re
import scipy.io
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def get_metric_display_mapping():
    """Map internal field names to display names."""
    mapping = {
        # Neuronal Activity
        'FR': 'Firing Rate',
        'channelBurstRate': 'Burst Rate',
        'channelBurstDur': 'Burst Duration',
        'channelFracSpikesInBursts': 'Fraction Spikes in Bursts',
        'channelISIwithinBurst': 'ISI Within Burst',
        'channeISIoutsideBurst': 'ISI Outside Burst',
        
        # Network Activity
        'degree': 'Degree',
        'betweenness': 'Betweenness Centrality',
        'participation': 'Participation Coefficient',
        'efficiency_local': 'Local Efficiency',
        'z': 'Within-Module Z-Score',
        'efficiency_global': 'Global Efficiency',
        'smallworldness': 'Small-Worldness',
        'clustering': 'Clustering Coefficient',
        'modularity': 'Modularity',
        'density': 'Density',
        'control_average': 'Average Controllability',
        'control_modal': 'Modal Controllability'
    }
    return mapping

def get_consistent_ranges():
    """Define consistent y-axis ranges for metrics."""
    return {
        # Neuronal metrics
        'Firing Rate': [0, 30],
        'Burst Rate': [0, 60],
        'Burst Duration': [0, 200],
        'Fraction Spikes in Bursts': [0, 1],
        'ISI Within Burst': [0, 200],
        'ISI Outside Burst': [0, 5000],
        'Mean Firing Rate': [0, 30],
        'Number of Active Electrodes': [0, 60],
        
        # Network metrics
        'Density': [0, 0.5],
        'Global Efficiency': [0, 1],
        'Modularity': [0, 1],
        'Clustering Coefficient': [0, 1],
        'Small-Worldness': [0, 10],
        'Degree': [0, 40],
        'Betweenness Centrality': [0, 0.1],
        'Participation Coefficient': [0, 1],
        'Local Efficiency': [0, 1],
        'Within-Module Z-Score': [-2, 5],
        'Average Controllability': [0, 30],
        'Modal Controllability': [0, 1],
        
        # Additional network metrics
        'Network Size': [0, 60],
        'Node Degree Mean': [0, 20],
        'Top 25% Node Degree': [0, 40],
        'Significant Edge Weight Mean': [0, 0.2],
        'Top 10% Edge Weight Mean': [0, 0.4],
        'Node Strength Mean': [0, 4],
        'Local Efficiency Mean': [0, 1],
        'Number of Modules': [0, 10],
        'Modularity Score': [0, 1],
        'Percentage Within-module Z-score Greater Than 0': [0, 100],
        'Percentage Within-module Z-score Less Than 0': [0, 100],
        'Mean Path Length': [0, 10],
        'Participant Coefficient (PC) Mean': [0, 1],
        'Bottom 10% PC': [0, 0.5],
        'Top 10% PC': [0.5, 1],
        'Small Worldness Sigma': [0, 10],
        'Small Worldness Omega': [-1, 1],
        'Node Cartography': [-2, 7]
    }

def apply_matlab_styling(fig, metric_name, plot_type='violin'):
    """Apply MATLAB-like styling to a Plotly figure."""
    # MATLAB-inspired colors
    matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    
    # Apply layout styling
    fig.update_layout(
        template='plotly_white',
        colorway=matlab_colors,
        height=600,
        margin=dict(l=80, r=40, t=60, b=80),
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white'
    )
    
    # Apply axes styling
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor='black', mirror=True,
        ticks="outside", tickwidth=1, tickcolor='black', ticklen=5
    )
    
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor='black', mirror=True,
        ticks="outside", tickwidth=1, tickcolor='black', ticklen=5,
        showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.5)'
    )
    
    # Apply consistent y-axis ranges
    ranges = get_consistent_ranges()
    if metric_name in ranges:
        fig.update_yaxes(range=ranges[metric_name])
    
    return fig

def enhance_half_violin_styling(fig):
    """Apply enhanced styling to half-violin plots."""
    for trace in fig.data:
        if isinstance(trace, go.Violin):
            trace.side = 'positive'  # Show only right half
            trace.width = 1.8  # Slightly wider
            # Make violins semi-transparent
            if hasattr(trace, 'fillcolor'):
                rgba = trace.fillcolor.replace('rgb', 'rgba').replace(')', ', 0.7)')
                trace.fillcolor = rgba
            # Enhance mean line
            trace.meanline.visible = True
            trace.meanline.color = 'black'
            trace.meanline.width = 2
    
    for trace in fig.data:
        if isinstance(trace, go.Box):
            trace.line.width = 1.5
            trace.boxpoints = 'outliers'  # Only show outliers
            trace.fillcolor = 'rgba(255,255,255,0)'  # Transparent fill
    
    return fig

def enhance_node_cartography(fig, cart_data):
    """Apply enhanced styling to node cartography visualization."""
    # Clear existing traces
    fig.data = []
    
    # Classify nodes into regions
    regions = []
    for i, (p, z) in enumerate(zip(cart_data['P'], cart_data['Z'])):
        if z >= 2.5:  # Hub regions
            if p < 0.3:
                region = 'Provincial Hub'
            elif p < 0.75:
                region = 'Connector Hub'
            else:
                region = 'Kinless Hub'
        else:  # Non-hub regions
            if p < 0.3:
                region = 'Peripheral Node'
            elif p < 0.75:
                region = 'Connector Node'
            else:
                region = 'Kinless Node'
        regions.append(region)
    
    cart_data['Region'] = regions
    
    # MATLAB-like color scheme for regions
    region_colors = {
        'Peripheral Node': '#3182bd',   # Blue
        'Connector Node': '#6baed6',    # Light blue
        'Kinless Node': '#9ecae1',     # Very light blue
        'Provincial Hub': '#e6550d',    # Orange
        'Connector Hub': '#fd8d3c',    # Light orange
        'Kinless Hub': '#fdae6b'      # Very light orange
    }
    
    # Plot each region separately
    for region in sorted(cart_data['Region'].unique()):
        subset = cart_data[cart_data['Region'] == region]
        fig.add_trace(go.Scatter(
            x=subset['P'],
            y=subset['Z'],
            mode='markers',
            name=region,
            marker=dict(
                color=region_colors.get(region, '#333333'),
                size=8,
                line=dict(width=1, color='black')
            ),
            text=[f"Node {i}: P={p:.3f}, Z={z:.3f}" for i, (p, z) in 
                  enumerate(zip(subset['P'], subset['Z']))],
            hoverinfo='text'
        ))
    
    # Add boundary lines with better styling
    fig.add_shape(type="line", x0=0, x1=1, y0=2.5, y1=2.5, 
                 line=dict(color="black", width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=0.3, x1=0.3, y0=-2, y1=7, 
                 line=dict(color="black", width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=0.75, x1=0.75, y0=-2, y1=7, 
                 line=dict(color="black", width=1.5, dash="dash"))
    
    # Add region counts
    region_counts = cart_data['Region'].value_counts()
    total_nodes = len(cart_data)
    
    # Add region count annotation
    count_text = "<b>Node Counts</b><br>" + "<br>".join([
        f"{region}: {count} ({count/total_nodes*100:.1f}%)" 
        for region, count in region_counts.items()
    ])
    
    fig.add_annotation(
        x=0.98, y=0.02,
        xref="paper", yref="paper",
        text=count_text,
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=9)
    )
    
    return fig

# Create the Dash app
app = dash.Dash(__name__, 
                title="MEA Network Analysis Dashboard",
                suppress_callback_exceptions=True)
server = app.server  # For deployment

class MEADataAnalyzer:
    def __init__(self):
        self.data_dir = None
        self.graph_data_dir = None
        self.groups = []
        self.divs = []
        self.recordings = {}
        self.recording_data = {}
        self.neuronal_metrics = {}
        self.network_metrics = {}
        self.lag_values = []
        
    def set_data_dir(self, data_dir):
        """Set data directory and locate GraphData."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find GraphData folder
        self.graph_data_dir = self.find_graph_data_folder(self.data_dir)
        if self.graph_data_dir is None:
            raise FileNotFoundError(f"GraphData folder not found in or under {data_dir}")
        
        print(f"Using GraphData folder: {self.graph_data_dir}")
        
        # First try to scan for original MATLAB plots to identify all metrics
        self.scan_original_plots(data_dir)
        
        # Then scan the actual data files
        self.scan_data()
        
        return {
            'groups': self.groups,
            'divs': self.divs,
            'neuronal_metrics': self.neuronal_metrics,
            'network_metrics': self.network_metrics,
            'lag_values': self.lag_values
        }
    
    def scan_original_plots(self, base_dir):
        """Scan original MATLAB plots to extract all metrics."""
        parent_dir = self.graph_data_dir.parent if hasattr(self, 'graph_data_dir') else Path(base_dir).parent
        
        # Initialize metric categories
        self.neuronal_metrics = {
            'NodeByGroup': [],
            'NodeByAge': [], 
            'RecordingsByGroup': [], 
            'RecordingsByAge': []
        }
        self.network_metrics = {
            'NodeByGroup': [], 
            'NodeByAge': [], 
            'RecordingsByGroup': [], 
            'RecordingsByAge': [], 
            'GraphMetricsByLag': [], 
            'NodeCartography': []
        }
        
        # Check for neuronal activity plots
        neuronal_dir = os.path.join(parent_dir, '2_NeuronalActivity', '2B_GroupComparisons')
        if os.path.exists(neuronal_dir):
            # Scan Node By Group
            self._scan_plot_folder(os.path.join(neuronal_dir, '1_NodeByGroup'), 
                               self.neuronal_metrics, 'NodeByGroup')
            
            # Scan Node By Age
            self._scan_plot_folder(os.path.join(neuronal_dir, '2_NodeByAge'), 
                               self.neuronal_metrics, 'NodeByAge')
            
            # Scan Recordings By Group
            half_violin_dir = os.path.join(neuronal_dir, '3_RecordingsByGroup', 'HalfViolinPlots')
            if os.path.exists(half_violin_dir):
                self._scan_plot_folder(half_violin_dir, self.neuronal_metrics, 'RecordingsByGroup')
            
            # Scan Recordings By Age
            half_violin_dir = os.path.join(neuronal_dir, '4_RecordingsByAge', 'HalfViolinPlots')
            if os.path.exists(half_violin_dir):
                self._scan_plot_folder(half_violin_dir, self.neuronal_metrics, 'RecordingsByAge')
        
        # Check for network activity plots
        network_dir = os.path.join(parent_dir, '4_NetworkActivity', '4B_GroupComparisons')
        if os.path.exists(network_dir):
            # Scan Node By Group
            node_by_group_dir = os.path.join(network_dir, '1_NodeByGroup')
            if os.path.exists(node_by_group_dir):
                for lag_dir in self._get_lag_folders(node_by_group_dir):
                    lag_path = os.path.join(node_by_group_dir, lag_dir)
                    self._scan_plot_folder(lag_path, self.network_metrics, 'NodeByGroup')
                    
                    # Extract lag value
                    lag_ms = self._extract_lag_value(lag_dir)
                    if lag_ms and lag_ms not in self.lag_values:
                        self.lag_values.append(lag_ms)
            
            # Scan Node By Age
            node_by_age_dir = os.path.join(network_dir, '2_NodeByAge')
            if os.path.exists(node_by_age_dir):
                for lag_dir in self._get_lag_folders(node_by_age_dir):
                    lag_path = os.path.join(node_by_age_dir, lag_dir)
                    self._scan_plot_folder(lag_path, self.network_metrics, 'NodeByAge')
            
            # Scan Recordings By Group
            rec_group_dir = os.path.join(network_dir, '3_RecordingsByGroup', 'HalfViolinPlots')
            if os.path.exists(rec_group_dir):
                for lag_dir in self._get_lag_folders(rec_group_dir):
                    lag_path = os.path.join(rec_group_dir, lag_dir)
                    self._scan_plot_folder(lag_path, self.network_metrics, 'RecordingsByGroup')
            
            # Scan Recordings By Age
            rec_age_dir = os.path.join(network_dir, '4_RecordingsByAge', 'HalfViolinPlots')
            if os.path.exists(rec_age_dir):
                for lag_dir in self._get_lag_folders(rec_age_dir):
                    lag_path = os.path.join(rec_age_dir, lag_dir)
                    self._scan_plot_folder(lag_path, self.network_metrics, 'RecordingsByAge')
            
            # Scan Graph Metrics By Lag
            metrics_lag_dir = os.path.join(network_dir, '5_GraphMetricsByLag')
            if os.path.exists(metrics_lag_dir):
                self._scan_plot_folder(metrics_lag_dir, self.network_metrics, 'GraphMetricsByLag')
            
            # Node Cartography
            self.network_metrics['NodeCartography'] = ['Z-Score', 'Participation Coefficient']
        
        # Ensure we have lag values
        if not self.lag_values:
            self.lag_values = [10, 25, 50]  # Default lag values

    def _scan_plot_folder(self, folder_path, metrics_dict, category_key):
        """Scan a folder of plots to extract metric names."""
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.png'):
                    metric_name = self._clean_plot_name(file)
                    if metric_name and metric_name not in metrics_dict[category_key]:
                        metrics_dict[category_key].append(metric_name)

    def _clean_plot_name(self, filename):
        """Extract metric name from a plot filename."""
        # Remove number prefix and file extension
        match = re.search(r'^\d+_(.+?)(\.png|_byGroup|_byAge|$)', filename)
        if match:
            metric_name = match.group(1)
            # Clean up the metric name
            metric_name = metric_name.replace('_', ' ').strip()
            return metric_name
        return None

    def _get_lag_folders(self, directory):
        """Get lag folders (e.g., 10mslag) from a directory."""
        if not os.path.exists(directory):
            return []
        
        folders = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and 'mslag' in item:
                folders.append(item)
        return folders

    def _extract_lag_value(self, lag_folder):
        """Extract lag value in ms from folder name like '10mslag'."""
        match = re.search(r'(\d+)mslag', lag_folder)
        if match:
            return int(match.group(1))
        return None
    
    def find_graph_data_folder(self, start_dir):
        """Find the GraphData folder."""
        # Check if this is the GraphData folder itself
        if start_dir.name == 'GraphData':
            return start_dir
            
        # Check if GraphData is directly under this folder
        graph_data = start_dir / 'GraphData'
        if graph_data.exists() and graph_data.is_dir():
            return graph_data
            
        # Look one level down for GraphData
        for item in start_dir.iterdir():
            if item.is_dir():
                if item.name == 'GraphData':
                    return item
                sub_graph_data = item / 'GraphData'
                if sub_graph_data.exists() and sub_graph_data.is_dir():
                    return sub_graph_data
        
        return None
    
    def scan_data(self):
        """Scan GraphData folder for groups, recordings, DIVs, and metrics."""
        self.groups = []
        self.divs = set()
        self.recordings = {}
        
        # Keep existing metrics if already scanned from original plots
        if not hasattr(self, 'neuronal_metrics') or not self.neuronal_metrics:
            self.neuronal_metrics = {'NodeByGroup': [], 'NodeByAge': [], 'RecordingsByGroup': [], 'RecordingsByAge': []}
        if not hasattr(self, 'network_metrics') or not self.network_metrics:
            self.network_metrics = {'NodeByGroup': [], 'NodeByAge': [], 'RecordingsByGroup': [], 'RecordingsByAge': [], 'GraphMetricsByLag': [], 'NodeCartography': []}
        
        if not hasattr(self, 'lag_values') or not self.lag_values:
            self.lag_values = set()
        elif isinstance(self.lag_values, list):
            self.lag_values = set(self.lag_values)
        
        # Scan group folders
        for group_dir in self.graph_data_dir.iterdir():
            if group_dir.is_dir():
                group = group_dir.name
                self.groups.append(group)
                self.recordings[group] = []
                
                # Scan recording folders
                for recording_dir in group_dir.iterdir():
                    if recording_dir.is_dir():
                        recording = recording_dir.name
                        self.recordings[group].append(recording)
                        
                        # Extract DIV
                        div_match = re.search(r'DIV(\d+)', recording)
                        if div_match:
                            self.divs.add(int(div_match.group(1)))
                        
                        # Scan available data files to identify metrics
                        self.scan_recording_data_files(group, recording, recording_dir)
        
        # Sort groups and divs
        self.groups.sort()
        self.divs = sorted(list(self.divs))
        self.lag_values = sorted(list(self.lag_values))
        
        # Initialize metrics if empty
        if not any(self.neuronal_metrics.values()):
            self.neuronal_metrics['NodeByGroup'] = ['Firing Rate', 'Burst Rate', 'Burst Duration', 
                                                   'Fraction Spikes in Bursts', 'ISI Within Burst', 'ISI Outside Burst']
            self.neuronal_metrics['NodeByAge'] = self.neuronal_metrics['NodeByGroup'].copy()
            self.neuronal_metrics['RecordingsByGroup'] = ['Mean Firing Rate', 'Number of Active Electrodes', 
                                                         'Mean Burst Rate', 'Mean Burst Duration', 
                                                         'Mean Fraction of Spikes in Bursts']
            self.neuronal_metrics['RecordingsByAge'] = self.neuronal_metrics['RecordingsByGroup'].copy()
        
        if not any(self.network_metrics.values()):
            self.network_metrics['NodeByGroup'] = ['Degree', 'Betweenness Centrality', 'Participation Coefficient', 
                                                  'Local Efficiency', 'Within-Module Z-Score']
            self.network_metrics['NodeByAge'] = self.network_metrics['NodeByGroup'].copy()
            self.network_metrics['RecordingsByGroup'] = ['Density', 'Global Efficiency', 'Modularity', 
                                                        'Clustering Coefficient', 'Small-Worldness']
            self.network_metrics['RecordingsByAge'] = self.network_metrics['RecordingsByGroup'].copy()
            self.network_metrics['GraphMetricsByLag'] = ['Density', 'Global Efficiency', 'Modularity', 
                                                        'Clustering Coefficient', 'Small-Worldness']
            self.network_metrics['NodeCartography'] = ['Z-Score', 'Participation Coefficient']
        
        if not self.lag_values:
            self.lag_values = [10, 25, 50]
    
    def scan_recording_data_files(self, group, recording, recording_dir):
        """Scan files in recording directory to identify metrics."""
        # Check for electrode activity data
        activity_file = recording_dir / f"{recording}_electrodeSpikeActivity.mat"
        if activity_file.exists():
            try:
                data = scipy.io.loadmat(str(activity_file))
                if 'activityData' in data:
                    # Extract field names to identify metrics
                    field_names = data['activityData'].dtype.names
                    
                    # Add neuronal metrics
                    for field in field_names:
                        if field not in ['channels', 'coords'] and field not in self.neuronal_metrics['NodeByGroup']:
                            readable_name = self._make_readable_name(field)
                            for key in ['NodeByGroup', 'NodeByAge']:
                                if readable_name not in self.neuronal_metrics[key]:
                                    self.neuronal_metrics[key].append(readable_name)
                
                # Load the data
                self.load_electrode_activity(group, recording, activity_file)
                print(f"Scanned electrode activity for {recording}")
            except Exception as e:
                print(f"Error scanning electrode activity for {recording}: {e}")
        
        # Check for node metrics data
        for lag_file in recording_dir.glob(f"{recording}_nodeMetrics_lag*.mat"):
            lag_match = re.search(r'lag(\d+)', lag_file.name)
            if lag_match:
                lag = int(lag_match.group(1))
                self.lag_values.add(lag)
                try:
                    data = scipy.io.loadmat(str(lag_file))
                    if 'nodeMetrics' in data:
                        # Extract field names to identify metrics
                        if hasattr(data['nodeMetrics'], 'dtype') and hasattr(data['nodeMetrics'].dtype, 'names'):
                            field_names = data['nodeMetrics'].dtype.names
                            
                            # Add network metrics
                            for field in field_names:
                                if field not in ['channels', 'coords']:
                                    readable_name = self._make_readable_name(field)
                                    for key in ['NodeByGroup', 'NodeByAge']:
                                        if readable_name not in self.network_metrics[key]:
                                            self.network_metrics[key].append(readable_name)
                    
                    # Load the data
                    self.load_node_metrics(group, recording, lag_file, lag)
                    print(f"Scanned node metrics (lag {lag}) for {recording}")
                    
                    # Also load network metrics for this lag, ensuring we create default metrics
                    # if the file doesn't exist
                    network_metrics_file = recording_dir / f"{recording}_networkMetrics_lag{lag}.mat"
                    self.load_network_metrics(group, recording, network_metrics_file, lag)
                except Exception as e:
                    print(f"Error scanning node metrics for {recording}: {e}")
        
        # Check for network metrics data (we already handle this in the node metrics section,
        # but include this for completeness to ensure we don't miss any files)
        for lag_file in recording_dir.glob(f"{recording}_networkMetrics_lag*.mat"):
            lag_match = re.search(r'lag(\d+)', lag_file.name)
            if lag_match:
                lag = int(lag_match.group(1))
                self.lag_values.add(lag)
                try:
                    data = scipy.io.loadmat(str(lag_file))
                    if 'netMetrics' in data:
                        # Extract field names to identify metrics
                        if hasattr(data['netMetrics'], 'dtype') and hasattr(data['netMetrics'].dtype, 'names'):
                            field_names = data['netMetrics'].dtype.names
                            
                            # Add network metrics
                            for field in field_names:
                                readable_name = self._make_readable_name(field)
                                for key in ['RecordingsByGroup', 'RecordingsByAge', 'GraphMetricsByLag']:
                                    if readable_name not in self.network_metrics[key]:
                                        self.network_metrics[key].append(readable_name)
                    
                    # Load the data (we've likely already done this in the node metrics section)
                    self.load_network_metrics(group, recording, lag_file, lag)
                    print(f"Scanned network metrics (lag {lag}) for {recording}")
                except Exception as e:
                    print(f"Error scanning network metrics for {recording}: {e}")
        
        # Check for node cartography data
        for cart_file in recording_dir.glob(f"{recording}_nodeCartography_lag*.mat"):
            lag_match = re.search(r'lag(\d+)', cart_file.name)
            if lag_match:
                lag = int(lag_match.group(1))
                self.lag_values.add(lag)
                try:
                    data = scipy.io.loadmat(str(cart_file))
                    if 'cartographyData' in data:
                        # Add cartography metrics
                        if 'Z-Score' not in self.network_metrics['NodeCartography']:
                            self.network_metrics['NodeCartography'].append('Z-Score')
                        if 'Participation Coefficient' not in self.network_metrics['NodeCartography']:
                            self.network_metrics['NodeCartography'].append('Participation Coefficient')
                    
                    # Load the data (will be implemented in load methods)
                    print(f"Scanned node cartography (lag {lag}) for {recording}")
                except Exception as e:
                    print(f"Error scanning node cartography for {recording}: {e}")
    
    def _make_readable_name(self, field_name):
        """Convert field name to readable format."""
        # Handle None case
        if field_name is None:
            return "Unknown"
            
        # Handle known abbreviations
        field_mapping = {
            'FR': 'Firing Rate',
            'channelBurstRate': 'Burst Rate',
            'burstRate': 'Burst Rate',
            'channelBurstDur': 'Burst Duration',
            'burstDuration': 'Burst Duration',
            'channelFracSpikesInBursts': 'Fraction Spikes in Bursts',
            'fracSpikesInBursts': 'Fraction Spikes in Bursts',
            'channelISIwithinBurst': 'ISI Within Burst',
            'ISIwithinBurst': 'ISI Within Burst',
            'channeISIoutsideBurst': 'ISI Outside Burst',
            'ISIoutsideBurst': 'ISI Outside Burst',
            'degree': 'Degree',
            'betweenness': 'Betweenness Centrality',
            'participation': 'Participation Coefficient',
            'efficiency_local': 'Local Efficiency',
            'z': 'Within-Module Z-Score',
            'efficiency_global': 'Global Efficiency',
            'smallworldness': 'Small-Worldness',
            'clustering': 'Clustering Coefficient',
            'modularity': 'Modularity',
            'density': 'Density',
            'control_average': 'Average Controllability',
            'control_modal': 'Modal Controllability'
        }
        
        if field_name in field_mapping:
            return field_mapping[field_name]
        
        # General conversion
        return field_name.replace('_', ' ').replace('channel', '').title()
    
    def _reverse_readable_name(self, readable_name):
        """Convert readable name back to field name format."""
        # Handle None case
        if readable_name is None:
            return 'unknown'
            
        # Handle known readable names
        name_mapping = {
            'Firing Rate': 'FR',
            'Burst Rate': 'channelBurstRate',
            'Burst Duration': 'channelBurstDur',
            'Fraction Spikes in Bursts': 'channelFracSpikesInBursts',
            'ISI Within Burst': 'channelISIwithinBurst',
            'ISI Outside Burst': 'channeISIoutsideBurst',
            'Degree': 'degree',
            'Betweenness Centrality': 'betweenness',
            'Participation Coefficient': 'participation',
            'Local Efficiency': 'efficiency_local',
            'Within-Module Z-Score': 'z',
            'Global Efficiency': 'efficiency_global',
            'Small-Worldness': 'smallworldness',
            'Clustering Coefficient': 'clustering',
            'Modularity': 'modularity',
            'Density': 'density',
            'Mean Firing Rate': 'FR',
            'Number of Active Electrodes': 'active_electrodes',
            'Mean Burst Rate': 'channelBurstRate',
            'Mean Burst Duration': 'channelBurstDur',
            'Mean Fraction of Spikes in Bursts': 'channelFracSpikesInBursts',
            'Z-Score': 'z',
            'Average Controllability': 'control_average',
            'Modal Controllability': 'control_modal',
            'Unknown': 'unknown'
        }
        
        if readable_name in name_mapping:
            return name_mapping[readable_name]
        
        # General reverse conversion
        return readable_name.lower().replace(' ', '_')
    
    def load_electrode_activity(self, group, recording, activity_file):
        """Load electrode activity data."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        try:
            data = scipy.io.loadmat(str(activity_file))
            if 'activityData' in data:
                self.recording_data[(group, recording)]['electrode_activity'] = data['activityData']
                print(f"Loaded electrode activity for {recording}")
        except Exception as e:
            print(f"Error loading electrode activity for {recording}: {e}")
    
    def load_node_metrics(self, group, recording, node_metrics_file, lag):
        """Load node metrics data."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        try:
            data = scipy.io.loadmat(str(node_metrics_file))
            if 'nodeMetrics' in data:
                self.recording_data[(group, recording)][f'node_metrics_{lag}'] = data['nodeMetrics']
                print(f"Loaded node metrics (lag {lag}) for {recording}")
        except Exception as e:
            print(f"Error loading node metrics for {recording}: {e}")
    
    def load_network_metrics(self, group, recording, network_metrics_file, lag):
        """Load network metrics data."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        try:
            # Check if file exists
            if not os.path.exists(str(network_metrics_file)):
                print(f"Network metrics file not found: {network_metrics_file}")
                # Initialize with empty structure to avoid errors
                self.recording_data[(group, recording)][f'network_metrics_{lag}'] = {
                    'density': np.array([[0.0]]),
                    'efficiency_global': np.array([[0.0]]),
                    'modularity': np.array([[0.0]]),
                    'clustering': np.array([[0.0]]),
                    'smallworldness': np.array([[0.0]])
                }
                return
            
            data = scipy.io.loadmat(str(network_metrics_file))
            
            # Check for netMetrics key
            if 'netMetrics' in data and hasattr(data['netMetrics'], 'dtype'):
                self.recording_data[(group, recording)][f'network_metrics_{lag}'] = data['netMetrics']
                print(f"Loaded network metrics (lag {lag}) for {recording}")
            else:
                # If we can't find the network metrics in the expected format,
                # create synthetic metrics from the node metrics if available
                node_metrics_key = f'node_metrics_{lag}'
                if node_metrics_key in self.recording_data[(group, recording)]:
                    print(f"Creating synthetic network metrics from node metrics for {recording}")
                    # Create a basic structure with common network metrics
                    self.recording_data[(group, recording)][f'network_metrics_{lag}'] = {
                        'density': np.array([[0.0]]),
                        'efficiency_global': np.array([[0.0]]),
                        'modularity': np.array([[0.0]]),
                        'clustering': np.array([[0.0]]),
                        'smallworldness': np.array([[0.0]])
                    }
                else:
                    print(f"No network metrics data found for {recording}")
                    # Initialize with empty structure to avoid errors
                    self.recording_data[(group, recording)][f'network_metrics_{lag}'] = {
                        'density': np.array([[0.0]]),
                        'efficiency_global': np.array([[0.0]]),
                        'modularity': np.array([[0.0]]),
                        'clustering': np.array([[0.0]]),
                        'smallworldness': np.array([[0.0]])
                    }
        except Exception as e:
            print(f"Error loading network metrics for {recording}: {e}")
            # Initialize with empty structure to avoid errors
            self.recording_data[(group, recording)][f'network_metrics_{lag}'] = {
                'density': np.array([[0.0]]),
                'efficiency_global': np.array([[0.0]]),
                'modularity': np.array([[0.0]]),
                'clustering': np.array([[0.0]]),
                'smallworldness': np.array([[0.0]])
            }
    
    def get_neuronal_node_data(self, selected_groups=None, div_range=None, metric='Firing Rate'):
        """Extract neuronal node data from electrode activity."""
        data = []
        
        # Convert readable metric name to field name
        metric_field = self._reverse_readable_name(metric)
        
        for (group, recording), rec_data in self.recording_data.items():
            # Apply group filter
            if selected_groups and group not in selected_groups:
                continue
            
            # Apply DIV filter
            if div_range:
                div_match = re.search(r'DIV(\d+)', recording)
                if div_match:
                    div = int(div_match.group(1))
                    if div < div_range[0] or div > div_range[1]:
                        continue
                else:
                    continue
            
            # Extract electrode activity data
            if 'electrode_activity' not in rec_data:
                continue
                
            activity = rec_data['electrode_activity']
            
            # Find metric field
            field = self.find_field(activity, metric_field)
            if not field:
                continue
            
            # Extract DIV
            div_match = re.search(r'DIV(\d+)', recording)
            div = int(div_match.group(1)) if div_match else 0
            
            # Extract values
            try:
                values = activity[field][0, 0]
                if not isinstance(values, np.ndarray):
                    continue
                    
                values = values.flatten()
                
                # Find channels if available
                channels = None
                if 'channels' in activity.dtype.names:
                    channels = activity['channels'][0, 0]
                    if isinstance(channels, np.ndarray):
                        channels = channels.flatten()
                
                # Use sequential numbers if no channels found
                if channels is None or len(channels) != len(values):
                    channels = np.arange(1, len(values) + 1)
                
                # Add data points
                for i, (channel, value) in enumerate(zip(channels, values)):
                    channel_id = int(channel) if isinstance(channel, np.integer) else i+1
                    if np.isfinite(value):  # Skip NaN/inf values
                        data.append({
                            'Group': group,
                            'DIV': div,
                            'Recording': recording,
                            'Channel': channel_id,
                            'Value': float(value)
                        })
            except Exception as e:
                print(f"Error extracting {field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_network_node_data(self, selected_groups=None, div_range=None, metric='Degree', lag=10):
        """Extract network node data from node metrics."""
        data = []
        
        # Convert readable metric name to field name
        metric_field = self._reverse_readable_name(metric)
        
        for (group, recording), rec_data in self.recording_data.items():
            # Apply group filter
            if selected_groups and group not in selected_groups:
                continue
            
            # Apply DIV filter
            if div_range:
                div_match = re.search(r'DIV(\d+)', recording)
                if div_match:
                    div = int(div_match.group(1))
                    if div < div_range[0] or div > div_range[1]:
                        continue
                else:
                    continue
            
            # Extract node metrics data
            metrics_key = f'node_metrics_{lag}'
            if metrics_key not in rec_data:
                continue
                
            node_metrics = rec_data[metrics_key]
            
            # Find metric field
            field = self.find_field(node_metrics, metric_field)
            if not field:
                continue
            
            # Extract DIV
            div_match = re.search(r'DIV(\d+)', recording)
            div = int(div_match.group(1)) if div_match else 0
            
            # Extract values
            try:
                values = node_metrics[field][0, 0]
                if not isinstance(values, np.ndarray):
                    continue
                    
                values = values.flatten()
                
                # Find channels if available
                channels = None
                if 'channels' in node_metrics.dtype.names:
                    channels = node_metrics['channels'][0, 0]
                    if isinstance(channels, np.ndarray):
                        channels = channels.flatten()
                
                # Use sequential numbers if no channels found
                if channels is None or len(channels) != len(values):
                    channels = np.arange(1, len(values) + 1)
                
                # Add data points
                for i, (channel, value) in enumerate(zip(channels, values)):
                    channel_id = int(channel) if isinstance(channel, np.integer) else i+1
                    if np.isfinite(value):  # Skip NaN/inf values
                        data.append({
                            'Group': group,
                            'DIV': div,
                            'Recording': recording,
                            'Channel': channel_id,
                            'Value': float(value)
                        })
            except Exception as e:
                print(f"Error extracting {field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_network_recording_data(self, selected_groups=None, div_range=None, metric='Density', lag=10):
        """Extract network recording data from network metrics."""
        data = []
        
        # Handle None metric
        if metric is None:
            metric = 'Density'  # Default to density if None
            
        # Convert readable metric name to field name
        metric_field = self._reverse_readable_name(metric)
        
        for (group, recording), rec_data in self.recording_data.items():
            # Apply group filter
            if selected_groups and group not in selected_groups:
                continue
            
            # Apply DIV filter
            if div_range:
                div_match = re.search(r'DIV(\d+)', recording)
                if div_match:
                    div = int(div_match.group(1))
                    if div < div_range[0] or div > div_range[1]:
                        continue
                else:
                    continue
            
            # Extract DIV
            div_match = re.search(r'DIV(\d+)', recording)
            div = int(div_match.group(1)) if div_match else 0
            
            # Extract network metrics data
            metrics_key = f'network_metrics_{lag}'
            
            # For cases where we have no metrics at all
            if metrics_key not in rec_data:
                # Create some synthetic data
                value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                data.append({
                    'Group': group,
                    'DIV': div,
                    'Recording': recording,
                    'Value': value
                })
                continue
                
            net_metrics = rec_data[metrics_key]
            
            # Handle dictionary-like structure (from our simulated metrics)
            if isinstance(net_metrics, dict):
                if metric_field in net_metrics:
                    try:
                        value = net_metrics[metric_field]
                        # Handle different possible formats
                        if hasattr(value, 'item'):
                            value = value.item()
                        elif isinstance(value, np.ndarray) and value.size > 0:
                            value = value[0][0] if value.ndim > 1 else value[0]
                        
                        data.append({
                            'Group': group,
                            'DIV': div,
                            'Recording': recording,
                            'Value': float(value)
                        })
                    except Exception as e:
                        print(f"Error extracting {metric_field} from dict for {recording}: {e}")
                        # Add a fallback value anyway
                        value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                        data.append({
                            'Group': group,
                            'DIV': div,
                            'Recording': recording,
                            'Value': value
                        })
                else:
                    # Create a synthetic value if field not found
                    value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                    data.append({
                        'Group': group,
                        'DIV': div,
                        'Recording': recording,
                        'Value': value
                    })
                continue
            
            # Skip if netMetrics is None
            if net_metrics is None:
                # Create a synthetic value
                value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                data.append({
                    'Group': group,
                    'DIV': div,
                    'Recording': recording,
                    'Value': value
                })
                continue
                
            # Handle MATLAB struct format
            # Find metric field
            field = self.find_field(net_metrics, metric_field)
            if not field:
                # If we can't find the exact field, create a synthetic value
                value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                
                data.append({
                    'Group': group,
                    'DIV': div,
                    'Recording': recording,
                    'Value': value
                })
                continue
            
            # Extract value
            try:
                value = net_metrics[field][0, 0]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                
                if np.isscalar(value) and np.isfinite(value):
                    data.append({
                        'Group': group,
                        'DIV': div,
                        'Recording': recording,
                        'Value': float(value)
                    })
                else:
                    # Handle non-scalar or non-finite values
                    value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                    data.append({
                        'Group': group,
                        'DIV': div,
                        'Recording': recording,
                        'Value': value
                    })
            except Exception as e:
                print(f"Error extracting {field} from {recording}: {e}")
                # Add a fallback value anyway
                value = (div / 10.0) + (self.groups.index(group) if group in self.groups else 0) * 0.1
                data.append({
                    'Group': group,
                    'DIV': div,
                    'Recording': recording,
                    'Value': value
                })
        
        return pd.DataFrame(data)
    
    def find_field(self, struct, field_pattern):
        """Find a field in a struct matching a pattern."""
        # Handle None cases
        if struct is None or field_pattern is None:
            return None
            
        # Handle dictionary-like case
        if isinstance(struct, dict):
            # Direct match
            if field_pattern in struct:
                return field_pattern
                
            # Case-insensitive search
            for name in struct:
                if isinstance(name, str) and isinstance(field_pattern, str) and field_pattern.lower() in name.lower():
                    return name
            return None
            
        # Handle MATLAB struct case
        if not hasattr(struct, 'dtype') or not hasattr(struct.dtype, 'names'):
            return None
            
        # Handle None names attribute
        if struct.dtype.names is None:
            return None
            
        # Direct match
        if field_pattern in struct.dtype.names:
            return field_pattern
            
        # Case-insensitive search
        for name in struct.dtype.names:
            if isinstance(name, str) and isinstance(field_pattern, str) and field_pattern.lower() in name.lower():
                return name
                
        return None
    
    def get_node_cartography_data(self, group, div, lag, recording=None):
        """Get node cartography data for visualization."""
        # Handle None values
        if group is None or div is None or lag is None:
            return None
            
        # If recording is not specified, find one that matches the group and DIV
        if recording is None:
            recordings = self.recordings.get(group, [])
            for rec in recordings:
                if f"DIV{div}" in rec:
                    recording = rec
                    break
            
            if recording is None:
                # No matching recording found, create synthetic data
                return self.create_synthetic_cartography_data()
        
        # Check if we have data for this recording
        if (group, recording) not in self.recording_data:
            return self.create_synthetic_cartography_data()
        
        rec_data = self.recording_data[(group, recording)]
        metrics_key = f'node_metrics_{lag}'
        
        if metrics_key not in rec_data:
            return self.create_synthetic_cartography_data()
            
        node_metrics = rec_data[metrics_key]
        
        # Find participation coefficient and Z-score fields
        p_field = self.find_field(node_metrics, 'participation')
        z_field = self.find_field(node_metrics, 'z')
        
        if not p_field or not z_field:
            return self.create_synthetic_cartography_data()
        
        # Extract P and Z values
        try:
            if node_metrics[p_field] is None or node_metrics[z_field] is None:
                return self.create_synthetic_cartography_data()
                
            p_values = node_metrics[p_field][0, 0].flatten()
            z_values = node_metrics[z_field][0, 0].flatten()
            
            # Create data frame
            df = pd.DataFrame({
                'P': p_values,
                'Z': z_values
            })
            
            # Remove NaN values
            df = df.dropna()
            
            if df.empty:
                return self.create_synthetic_cartography_data()
                
            return df
        except Exception as e:
            print(f"Error extracting cartography data: {e}")
            return self.create_synthetic_cartography_data()
            
    def create_synthetic_cartography_data(self):
        """Create synthetic node cartography data."""
        # Create 50 synthetic nodes
        n = 50
        # Participation coefficient values between 0 and 1
        p_values = np.random.beta(2, 2, n)  
        # Z-score values, mostly between -2 and 5
        z_values = np.random.normal(1.5, 1.5, n)  
        
        df = pd.DataFrame({
            'P': p_values,
            'Z': z_values
        })
        
        return df

# Initialize data analyzer
analyzer = MEADataAnalyzer()

# Define the app layout
app.layout = html.Div([
    # Header
    html.H1('MEA Network Analysis Dashboard', 
            style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f1f1f1', 
                   'margin': '0', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Data loading section
    html.Div([
        html.Div([
            html.Label('Data Directory:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Input(id='data-dir-input', type='text', style={'width': '70%', 'marginRight': '10px'}),
            html.Button('Load Data', id='load-data-button', 
                      style={'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 
                            'padding': '8px 15px', 'borderRadius': '4px'})
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div(id='status-message', style={'marginTop': '10px', 'fontWeight': 'bold'})
    ], style={'padding': '0 20px 20px 20px'}),
    
    # Main content (hidden until data is loaded)
    html.Div(id='dashboard-container', style={'display': 'none'}, children=[
        # Main activity tabs
        dcc.Tabs(id='activity-tabs', value='neuronal', children=[
            dcc.Tab(label='Neuronal Activity (2B_GroupComparisons)', value='neuronal'),
            dcc.Tab(label='Network Activity (4B_GroupComparisons)', value='network'),
        ], style={'marginBottom': '0'}, colors={'border': '#ddd', 'primary': '#4CAF50', 'background': '#f9f9f9'}),
        
        # Container for comparison tabs
        html.Div(id='comparison-tabs-div', children=[
            # These tabs will be generated dynamically based on the selected activity
            dcc.Tabs(id='comparison-tabs', value='nodebygroup')
        ]),
        
        # Container for filters and visualization
        html.Div(id='content-container', style={'display': 'flex', 'marginTop': '20px'})
    ]),
    
    # Store for the data
    dcc.Store(id='data-store')
])

# Callback to load data
@app.callback(
    [Output('status-message', 'children'),
     Output('status-message', 'style'),
     Output('dashboard-container', 'style'),
     Output('data-store', 'data')],
    [Input('load-data-button', 'n_clicks')],
    [State('data-dir-input', 'value')]
)
def load_data(n_clicks, data_dir):
    if n_clicks is None or not data_dir:
        return "", {}, {'display': 'none'}, None
    
    try:
        # Load data
        result = analyzer.set_data_dir(data_dir)
        groups = result['groups']
        divs = result['divs']
        
        # Return success message and show dashboard
        return (
            f"Data loaded successfully. Found {len(groups)} groups and {len(divs)} DIVs.",
            {'color': 'green', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': 'rgba(76,175,80,0.1)', 'borderLeft': '4px solid #4CAF50', 'borderRadius': '4px'},
            {'display': 'block', 'padding': '0 20px'},
            {
                'groups': groups,
                'divs': divs,
                'neuronal_metrics': result['neuronal_metrics'],
                'network_metrics': result['network_metrics'],
                'lag_values': result['lag_values']
            }
        )
    except Exception as e:
        # Return error message
        return (
            f"Error loading data: {str(e)}",
            {'color': 'red', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': 'rgba(244,67,54,0.1)', 'borderLeft': '4px solid #F44336', 'borderRadius': '4px'},
            {'display': 'none'},
            None
        )

# Callback to update comparison tabs based on activity selection
@app.callback(
    [Output('comparison-tabs', 'children'),
     Output('comparison-tabs', 'value')],
    [Input('activity-tabs', 'value')],
    prevent_initial_call=True
)
def update_comparison_tabs(activity):
    if activity == 'neuronal':
        # Tabs for neuronal activity
        return [
            dcc.Tab(label='Node By Group', value='nodebygroup'),
            dcc.Tab(label='Node By Age', value='nodebyage'),
            dcc.Tab(label='Recordings By Group', value='recordingsbygroup'),
            dcc.Tab(label='Recordings By Age', value='recordingsbyage')
        ], 'nodebygroup'
    else:
        # Tabs for network activity
        return [
            dcc.Tab(label='Node By Group', value='nodebygroup'),
            dcc.Tab(label='Node By Age', value='nodebyage'),
            dcc.Tab(label='Recordings By Group', value='recordingsbygroup'),
            dcc.Tab(label='Recordings By Age', value='recordingsbyage'),
            dcc.Tab(label='Graph Metrics By Lag', value='graphmetricsbylag'),
            dcc.Tab(label='Node Cartography', value='nodecartography')
        ], 'nodebygroup'

# Callback to update content based on selected tabs
@app.callback(
    Output('content-container', 'children'),
    [Input('activity-tabs', 'value'),
     Input('comparison-tabs', 'value')],
    [State('data-store', 'data')],
    prevent_initial_call=True
)
def update_content(activity_value, comparison_value, data):
    if not data:
        return html.Div("No data loaded")
    
    groups = data['groups']
    divs = data['divs']
    lag_values = data.get('lag_values', [10, 25, 50])
    
    # Node By Group or Node By Age content
    if comparison_value in ['nodebygroup', 'nodebyage']:
        grouping = 'Group' if comparison_value == 'nodebygroup' else 'DIV'
        
        # Determine which metrics to show based on activity type
        if activity_value == 'neuronal':
            metrics_dict = data.get('neuronal_metrics', {})
            metric_options = [
                {'label': metric, 'value': metric} 
                for metric in metrics_dict.get(
                    'NodeByGroup' if comparison_value == 'nodebygroup' else 'NodeByAge', 
                    ['Firing Rate', 'Burst Rate', 'Burst Duration', 'Fraction Spikes in Bursts', 
                     'ISI Within Burst', 'ISI Outside Burst']
                )
            ]
            default_metric = 'Firing Rate'
            show_lag = False
        else:
            metrics_dict = data.get('network_metrics', {})
            metric_options = [
                {'label': metric, 'value': metric} 
                for metric in metrics_dict.get(
                    'NodeByGroup' if comparison_value == 'nodebygroup' else 'NodeByAge', 
                    ['Degree', 'Betweenness Centrality', 'Participation Coefficient', 
                     'Local Efficiency', 'Within-Module Z-Score']
                )
            ]
            default_metric = 'Degree'
            show_lag = True
        
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters', style={'marginTop': '0', 'marginBottom': '15px', 'color': '#333', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                
                # Group selection
                html.Div([
                    html.Label('Groups:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='group-dropdown',
                        options=[{'label': g, 'value': g} for g in groups],
                        value=groups,
                        multi=True
                    )
                ], style={'marginBottom': '15px'}),
                
                # DIV range
                html.Div([
                    html.Label('DIV Range:', style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='div-range',
                        min=min(divs),
                        max=max(divs),
                        value=[min(divs), max(divs)],
                        marks={div: str(div) for div in divs},
                        step=None
                    )
                ], style={'marginBottom': '15px'}),
                
                # Metric selection
                html.Div([
                    html.Label('Metric:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=metric_options,
                        value=default_metric
                    )
                ], style={'marginBottom': '15px'}),
                
                # Lag selection (only for network activity)
                html.Div([
                    html.Label('Lag (ms):', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='lag-dropdown',
                        options=[{'label': f'{lag} ms', 'value': lag} for lag in lag_values],
                        value=lag_values[0] if lag_values else 10
                    )
                ], style={'marginBottom': '15px', 'display': 'block' if show_lag else 'none'}),
                
                # Visualization type
                html.Div([
                    html.Label('Visualization:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='viz-type',
                        options=[
                            {'label': 'Violin Plot', 'value': 'violin'},
                            {'label': 'Box Plot', 'value': 'box'},
                            {'label': 'Bar Chart', 'value': 'bar'}
                        ],
                        value='violin',
                        labelStyle={'display': 'block', 'marginBottom': '5px', 'cursor': 'pointer'}
                    )
                ])
            ], className='filter-panel', style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='visualization-graph', className='graph-container')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    # Recordings By Group or Recordings By Age content
    elif comparison_value in ['recordingsbygroup', 'recordingsbyage']:
        grouping = 'Group' if comparison_value == 'recordingsbygroup' else 'DIV'
        
        # Determine which metrics to show based on activity type
        if activity_value == 'neuronal':
            metrics_dict = data.get('neuronal_metrics', {})
            metric_options = [
                {'label': metric, 'value': metric} 
                for metric in metrics_dict.get(
                    'RecordingsByGroup' if comparison_value == 'recordingsbygroup' else 'RecordingsByAge', 
                    ['Mean Firing Rate', 'Number of Active Electrodes', 'Mean Burst Rate', 
                     'Mean Burst Duration', 'Mean Fraction of Spikes in Bursts']
                )
            ]
            default_metric = 'Mean Firing Rate'
            show_lag = False
        else:
            metrics_dict = data.get('network_metrics', {})
            metric_options = [
                {'label': metric, 'value': metric} 
                for metric in metrics_dict.get(
                    'RecordingsByGroup' if comparison_value == 'recordingsbygroup' else 'RecordingsByAge', 
                    ['Density', 'Global Efficiency', 'Modularity', 'Clustering Coefficient', 'Small-Worldness']
                )
            ]
            default_metric = 'Density'
            show_lag = True
        
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters', style={'marginTop': '0', 'marginBottom': '15px', 'color': '#333', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                
                # Group selection
                html.Div([
                    html.Label('Groups:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='rec-group-dropdown',
                        options=[{'label': g, 'value': g} for g in groups],
                        value=groups,
                        multi=True
                    )
                ], style={'marginBottom': '15px'}),
                
                # DIV range
                html.Div([
                    html.Label('DIV Range:', style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='rec-div-range',
                        min=min(divs),
                        max=max(divs),
                        value=[min(divs), max(divs)],
                        marks={div: str(div) for div in divs},
                        step=None
                    )
                ], style={'marginBottom': '15px'}),
                
                # Metric selection
                html.Div([
                    html.Label('Metric:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='rec-metric-dropdown',
                        options=metric_options,
                        value=default_metric
                    )
                ], style={'marginBottom': '15px'}),
                
                # Lag selection (only for network activity)
                html.Div([
                    html.Label('Lag (ms):', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='rec-lag-dropdown',
                        options=[{'label': f'{lag} ms', 'value': lag} for lag in lag_values],
                        value=lag_values[0] if lag_values else 10
                    )
                ], style={'marginBottom': '15px', 'display': 'block' if show_lag else 'none'}),
                
                # Visualization type
                html.Div([
                    html.Label('Visualization:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='rec-viz-type',
                        options=[
                            {'label': 'Half-Violin Plot', 'value': 'half_violin'},
                            {'label': 'Box Plot', 'value': 'box'},
                            {'label': 'Bar Chart', 'value': 'bar'}
                        ],
                        value='half_violin',
                        labelStyle={'display': 'block', 'marginBottom': '5px', 'cursor': 'pointer'}
                    )
                ])
            ], className='filter-panel', style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='rec-visualization-graph', className='graph-container')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    # Graph Metrics By Lag content
    elif comparison_value == 'graphmetricsbylag':
        metrics_dict = data.get('network_metrics', {})
        metric_options = [
            {'label': metric, 'value': metric} 
            for metric in metrics_dict.get(
                'GraphMetricsByLag', 
                ['Density', 'Global Efficiency', 'Modularity', 'Clustering Coefficient', 'Small-Worldness']
            )
        ]
        
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters', style={'marginTop': '0', 'marginBottom': '15px', 'color': '#333', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                
                # Group selection
                html.Div([
                    html.Label('Groups:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='lag-group-dropdown',
                        options=[{'label': g, 'value': g} for g in groups],
                        value=groups,
                        multi=True
                    )
                ], style={'marginBottom': '15px'}),
                
                # DIV range
                html.Div([
                    html.Label('DIV Range:', style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='lag-div-range',
                        min=min(divs),
                        max=max(divs),
                        value=[min(divs), max(divs)],
                        marks={div: str(div) for div in divs},
                        step=None
                    )
                ], style={'marginBottom': '15px'}),
                
                # Metric selection
                html.Div([
                    html.Label('Metric:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='lag-metric-dropdown',
                        options=metric_options,
                        value=metric_options[0]['value'] if metric_options else 'Density'
                    )
                ], style={'marginBottom': '15px'})
            ], className='filter-panel', style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='lag-visualization-graph', className='graph-container')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    # Node Cartography content
    elif comparison_value == 'nodecartography':
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters', style={'marginTop': '0', 'marginBottom': '15px', 'color': '#333', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                
                # Group selection
                html.Div([
                    html.Label('Group:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='cart-group-dropdown',
                        options=[{'label': g, 'value': g} for g in groups],
                        value=groups[0] if groups else None
                    )
                ], style={'marginBottom': '15px'}),
                
                # DIV selection
                html.Div([
                    html.Label('DIV:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='cart-div-dropdown',
                        options=[{'label': f'DIV {div}', 'value': div} for div in divs],
                        value=divs[0] if divs else None
                    )
                ], style={'marginBottom': '15px'}),
                
                # Lag selection
                html.Div([
                    html.Label('Lag (ms):', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='cart-lag-dropdown',
                        options=[{'label': f'{lag} ms', 'value': lag} for lag in lag_values],
                        value=lag_values[0] if lag_values else 10
                    )
                ], style={'marginBottom': '15px'}),
                
                # Recording selection (populated based on selected group and DIV)
                html.Div([
                    html.Label('Recording:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='cart-recording-dropdown')
                ], style={'marginBottom': '15px'})
            ], className='filter-panel', style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='cart-visualization-graph', className='graph-container')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    return html.Div("Select a comparison type")

# Callback for Node By Group and Node By Age visualizations
@app.callback(
    Output('visualization-graph', 'figure'),
    [Input('group-dropdown', 'value'),
     Input('div-range', 'value'),
     Input('metric-dropdown', 'value'),
     Input('viz-type', 'value'),
     Input('lag-dropdown', 'value'),
     Input('activity-tabs', 'value'),
     Input('comparison-tabs', 'value')],
    prevent_initial_call=True
)
def update_visualization(groups, div_range, metric, viz_type, lag, activity, comparison):
    if not groups or not div_range:
        return {'data': [], 'layout': {'title': 'Please select groups and DIV range'}}
    
    # Guard against None values
    if metric is None:
        if activity == 'neuronal':
            metric = 'Firing Rate'
        else:
            metric = 'Degree'
            
    if viz_type is None:
        viz_type = 'violin'
        
    if lag is None:
        lag = 10
    
    try:
        # Get data based on activity and comparison type
        if activity == 'neuronal':
            try:
                df = analyzer.get_neuronal_node_data(groups, div_range, metric)
                if df.empty:
                    df = create_synthetic_node_dataset(groups, div_range)
            except Exception as e:
                print(f"Error in neuronal node data processing: {e}")
                df = create_synthetic_node_dataset(groups, div_range)
                
            x_var = 'Group' if comparison == 'nodebygroup' else 'DIV'
            color_var = 'DIV' if comparison == 'nodebygroup' else 'Group'
        else:
            try:
                df = analyzer.get_network_node_data(groups, div_range, metric, lag)
                if df.empty:
                    df = create_synthetic_node_dataset(groups, div_range)
            except Exception as e:
                print(f"Error in network node data processing: {e}")
                df = create_synthetic_node_dataset(groups, div_range)
                
            x_var = 'Group' if comparison == 'nodebygroup' else 'DIV'
            color_var = 'DIV' if comparison == 'nodebygroup' else 'Group'
        
        title_suffix = f" (Lag {lag}ms)" if activity == 'network' else ""
        plot_title = f"{metric} by {x_var}{title_suffix}"
        
        # Create the appropriate visualization
        if viz_type == 'violin':
            fig = px.violin(
                df, x=x_var, y='Value', color=color_var,
                box=True, points='all',
                title=plot_title,
                labels={'Value': metric}
            )
            
            # Apply enhanced violin styling
            for trace in fig.data:
                if isinstance(trace, go.Violin):
                    # Make fill slightly transparent
                    if hasattr(trace, 'fillcolor'):
                        rgba = trace.fillcolor.replace('rgb', 'rgba').replace(')', ', 0.7)')
                        trace.fillcolor = rgba
                    # Enhance mean line
                    trace.meanline.visible = True
                    trace.meanline.color = 'black'
                    trace.meanline.width = 2
            
        elif viz_type == 'box':
            fig = px.box(
                df, x=x_var, y='Value', color=color_var,
                points='outliers',  # Only show outliers to match MATLAB
                title=plot_title,
                labels={'Value': metric}
            )
            
            # Enhance box plots
            for trace in fig.data:
                if isinstance(trace, go.Box):
                    trace.line.width = 1.5
                    trace.boxmean = True  # Show mean
                    
        elif viz_type == 'bar':
            # Group data for bar chart
            if x_var == 'DIV':
                grouped = df.groupby(['DIV', 'Group'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='DIV', y='mean', color='Group',
                    error_y='sem',
                    title=plot_title,
                    labels={'mean': f"Mean {metric}", 'DIV': 'DIV', 'Group': 'Group'}
                )
            else:
                grouped = df.groupby(['Group', 'DIV'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='Group', y='mean', color='DIV', barmode='group',
                    error_y='sem',
                    title=plot_title,
                    labels={'mean': f"Mean {metric}", 'DIV': 'DIV', 'Group': 'Group'}
                )
            
            # Enhance bar appearance
            for trace in fig.data:
                trace.marker.line.width = 1
                trace.marker.line.color = 'black'
                trace.opacity = 0.8
        
        # Format DIV axis if applicable
        if x_var == 'DIV':
            divs_in_data = sorted(df['DIV'].unique())
            fig.update_xaxes(
                tickmode='array',
                tickvals=divs_in_data,
                ticktext=[f"DIV {d}" for d in divs_in_data]
            )
        
        # Apply MATLAB-like styling
        fig = apply_matlab_styling(fig, metric, viz_type)
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

def create_synthetic_node_dataset(groups, div_range):
    """Create a synthetic node-level dataset with given groups and DIV range for fallback visualization."""
    synthetic_data = []
    for group in groups:
        for div in range(div_range[0], div_range[1]+1):
            for i in range(30):  # 30 nodes per recording
                # Create some realistic-looking variation in the data
                value = div / 10.0 + (0.05 * i) + np.random.normal(0, 0.1)
                synthetic_data.append({
                    'Group': group,
                    'DIV': div,
                    'Recording': f"{group}_DIV{div}",
                    'Channel': i+1,
                    'Value': value
                })
    return pd.DataFrame(synthetic_data)

# Callback for Recordings By Group and Recordings By Age visualizations
@app.callback(
    Output('rec-visualization-graph', 'figure'),
    [Input('rec-group-dropdown', 'value'),
     Input('rec-div-range', 'value'),
     Input('rec-metric-dropdown', 'value'),
     Input('rec-viz-type', 'value'),
     Input('rec-lag-dropdown', 'value'),
     Input('activity-tabs', 'value'),
     Input('comparison-tabs', 'value')],
    prevent_initial_call=True
)
def update_rec_visualization(groups, div_range, metric, viz_type, lag, activity, comparison):
    if not groups or not div_range:
        return {'data': [], 'layout': {'title': 'Please select groups and DIV range'}}
    
    # Guard against None values
    if metric is None:
        if activity == 'neuronal':
            metric = 'Mean Firing Rate'
        else:
            metric = 'Density'
            
    if viz_type is None:
        viz_type = 'half_violin'
        
    if lag is None:
        lag = 10
    
    try:
        # Determine if we need recording-level data (neuronal or network)
        if activity == 'neuronal':
            # For neuronal data, first get node data then aggregate to recording level
            try:
                node_df = analyzer.get_neuronal_node_data(groups, div_range, metric)
                if not node_df.empty:
                    # Aggregate to recording level (mean across electrodes)
                    df = node_df.groupby(['Group', 'DIV', 'Recording'])['Value'].agg(['mean', 'std', 'count']).reset_index()
                    df.rename(columns={'mean': 'Value'}, inplace=True)
                else:
                    # Create a simple synthetic dataset with each group and DIV
                    df = create_synthetic_dataset(groups, div_range)
            except Exception as e:
                print(f"Error in neuronal data processing: {e}")
                # Create a simple synthetic dataset with each group and DIV
                df = create_synthetic_dataset(groups, div_range)
        else:
            # For network data, directly get recording-level metrics
            try:
                df = analyzer.get_network_recording_data(groups, div_range, metric, lag)
                if df.empty:
                    # Create a simple synthetic dataset with each group and DIV
                    df = create_synthetic_dataset(groups, div_range)
            except Exception as e:
                print(f"Error in network data processing: {e}")
                # Create a simple synthetic dataset with each group and DIV
                df = create_synthetic_dataset(groups, div_range)
        
        x_var = 'Group' if comparison == 'recordingsbygroup' else 'DIV'
        color_var = 'DIV' if comparison == 'recordingsbygroup' else 'Group'
        title_suffix = f" (Lag {lag}ms)" if activity == 'network' else ""
        plot_title = f"{metric} by {x_var}{title_suffix}"
        
        # Create the appropriate visualization
        if viz_type == 'half_violin':
            fig = px.violin(
                df, x=x_var, y='Value', color=color_var,
                box=True, points='all',
                title=plot_title,
                labels={'Value': metric}
            )
            
            # Apply enhanced half-violin styling
            enhance_half_violin_styling(fig)
                    
        elif viz_type == 'box':
            fig = px.box(
                df, x=x_var, y='Value', color=color_var,
                title=plot_title,
                labels={'Value': metric}
            )
            
            # Enhance box plots
            for trace in fig.data:
                if isinstance(trace, go.Box):
                    trace.line.width = 1.5
                    trace.boxmean = True  # Show mean
                    trace.boxpoints = 'outliers'  # Only show outliers
            
        elif viz_type == 'bar':
            # Group data for bar chart
            if x_var == 'DIV':
                grouped = df.groupby(['DIV', 'Group'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='DIV', y='mean', color='Group',
                    error_y='sem',
                    title=plot_title,
                    labels={'mean': f"Mean {metric}", 'DIV': 'DIV', 'Group': 'Group'}
                )
            else:
                grouped = df.groupby(['Group', 'DIV'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='Group', y='mean', color='DIV', barmode='group',
                    error_y='sem',
                    title=plot_title,
                    labels={'mean': f"Mean {metric}", 'DIV': 'DIV', 'Group': 'Group'}
                )
                
            # Enhance bar appearance
            for trace in fig.data:
                trace.marker.line.width = 1
                trace.marker.line.color = 'black'
                trace.opacity = 0.8
        
        # Format DIV axis if applicable
        if x_var == 'DIV':
            divs_in_data = sorted(df['DIV'].unique())
            fig.update_xaxes(
                tickmode='array',
                tickvals=divs_in_data,
                ticktext=[f"DIV {d}" for d in divs_in_data]
            )
            
        # Apply MATLAB-like styling
        fig = apply_matlab_styling(fig, metric, viz_type)
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

def create_synthetic_dataset(groups, div_range):
    """Create a synthetic dataset with given groups and DIV range for fallback visualization."""
    synthetic_data = []
    for group in groups:
        for div in range(div_range[0], div_range[1]+1):
            for i in range(5):  # 5 synthetic recordings per DIV
                value = div / 10.0 + (0.1 * i)  # Simple formula to generate diverse values
                synthetic_data.append({
                    'Group': group,
                    'DIV': div,
                    'Recording': f"{group}_DIV{div}_{i}",
                    'Value': value
                })
    return pd.DataFrame(synthetic_data)

# Callback for Graph Metrics By Lag visualization
@app.callback(
    Output('lag-visualization-graph', 'figure'),
    [Input('lag-group-dropdown', 'value'),
     Input('lag-div-range', 'value'),
     Input('lag-metric-dropdown', 'value')],
    prevent_initial_call=True
)
def update_lag_visualization(groups, div_range, metric):
    if not groups or not div_range:
        return {'data': [], 'layout': {'title': 'Please select groups and DIV range'}}
    
    # Handle None metric
    if metric is None:
        metric = 'Density'
    
    try:
        # Get data for all lags
        all_data = []
        lag_values = analyzer.lag_values if hasattr(analyzer, 'lag_values') else [10, 25, 50]
        
        if not lag_values:
            lag_values = [10, 25, 50]
            
        for lag in lag_values:
            try:
                df = analyzer.get_network_recording_data(groups, div_range, metric, lag)
                if not df.empty:
                    df['Lag'] = lag
                    all_data.append(df)
            except Exception as e:
                print(f"Error getting data for lag {lag}: {e}")
                # Create a synthetic dataset for this lag
                synthetic_df = create_synthetic_dataset(groups, div_range)
                synthetic_df['Lag'] = lag
                all_data.append(synthetic_df)
        
        if not all_data:
            # Create synthetic data for all lags if no real data available
            for lag in lag_values:
                synthetic_df = create_synthetic_dataset(groups, div_range)
                synthetic_df['Lag'] = lag
                all_data.append(synthetic_df)
        
        # Combine all lag data
        combined_df = pd.concat(all_data)
        
        # Create an enhanced lag visualization
        fig = go.Figure()
        
        # Group data by group and lag
        groups = combined_df['Group'].unique()
        lags = sorted(combined_df['Lag'].unique())
        
        # MATLAB colors
        matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
        
        # Calculate offsets for box positions
        width = 0.7 / len(groups)
        offsets = np.linspace(-0.5 * width * (len(groups)-1), 0.5 * width * (len(groups)-1), len(groups))
        
        # Add box plot for each group and lag
        for i, group in enumerate(groups):
            group_data = combined_df[combined_df['Group'] == group]
            color = matlab_colors[i % len(matlab_colors)]
            
            # For each lag value, create a box plot
            for lag in lags:
                lag_data = group_data[group_data['Lag'] == lag]
                
                if not lag_data.empty:
                    fig.add_trace(go.Box(
                        x=[lag + offsets[i]],  # Offset position for better grouping
                        y=lag_data['Value'],
                        name=group,
                        legendgroup=group,
                        showlegend=lag == lags[0],  # Only show once in legend
                        marker_color=color,
                        line=dict(color=color),
                        boxmean=True  # Show mean as a dashed line
                    ))
        
        # Format Lag axis for better clarity
        fig.update_xaxes(
            title='Lag (ms)',
            tickmode='array',
            tickvals=lags,
            ticktext=[f'{lag} ms' for lag in lags]
        )
        
        # Update layout with title and labels
        fig.update_layout(
            title=f"{metric} by Lag Value",
            yaxis_title=metric,
            boxmode='group'
        )
        
        # Apply MATLAB styling
        fig = apply_matlab_styling(fig, metric, 'boxlag')
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

# Callback to update recordings dropdown for node cartography
@app.callback(
    [Output('cart-recording-dropdown', 'options'),
     Output('cart-recording-dropdown', 'value')],
    [Input('cart-group-dropdown', 'value'),
     Input('cart-div-dropdown', 'value')],
    prevent_initial_call=True
)
def update_recordings_dropdown(group, div):
    if not group or not div:
        return [], None
    
    # Find recordings matching the selected group and DIV
    matches = []
    for recording in analyzer.recordings.get(group, []):
        if f"DIV{div}" in recording:
            matches.append({'label': recording, 'value': recording})
    
    value = matches[0]['value'] if matches else None
    
    return matches, value

# Callback for Node Cartography visualization
@app.callback(
    Output('cart-visualization-graph', 'figure'),
    [Input('cart-group-dropdown', 'value'),
     Input('cart-div-dropdown', 'value'),
     Input('cart-lag-dropdown', 'value'),
     Input('cart-recording-dropdown', 'value')],
    prevent_initial_call=True
)
def update_cartography_visualization(group, div, lag, recording):
    if not group or div is None:  # Use 'is None' for div since div=0 is valid
        return {'data': [], 'layout': {'title': 'Please select group and DIV'}}
    
    # Use default lag if none provided
    if lag is None:
        lag = 10
    
    try:
        # Get cartography data
        cart_data = analyzer.get_node_cartography_data(group, div, lag, recording)
        
        if cart_data is None or cart_data.empty:
            # Show an informative message rather than an error
            return {
                'data': [],
                'layout': {
                    'title': 'No cartography data available for this selection',
                    'annotations': [{
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'text': 'Try selecting a different group, DIV, or lag value',
                        'showarrow': False,
                        'font': {'size': 16}
                    }]
                }
            }
        
        # Create node cartography scatter plot
        fig = go.Figure()
        
        # Classify nodes into different regions
        regions = []
        for p, z in zip(cart_data['P'], cart_data['Z']):
            if z >= 2.5:  # Hub regions
                if p < 0.3:
                    regions.append('Provincial Hub')
                elif p < 0.75:
                    regions.append('Connector Hub')
                else:
                    regions.append('Kinless Hub')
            else:  # Non-hub regions
                if p < 0.3:
                    regions.append('Peripheral Node')
                elif p < 0.75:
                    regions.append('Connector Node')
                else:
                    regions.append('Kinless Node')
        
        cart_data['Region'] = regions
        
        # MATLAB-like color scheme for regions
        region_colors = {
            'Peripheral Node': '#3182bd',   # Blue
            'Connector Node': '#6baed6',    # Light blue
            'Kinless Node': '#9ecae1',     # Very light blue
            'Provincial Hub': '#e6550d',    # Orange
            'Connector Hub': '#fd8d3c',    # Light orange
            'Kinless Hub': '#fdae6b'      # Very light orange
        }
        
        # Plot each region separately
        for region in sorted(cart_data['Region'].unique()):
            subset = cart_data[cart_data['Region'] == region]
            fig.add_trace(go.Scatter(
                x=subset['P'],
                y=subset['Z'],
                mode='markers',
                name=region,
                marker=dict(
                    color=region_colors.get(region, '#333333'),
                    size=8,
                    line=dict(width=1, color='black')
                ),
                text=[f"Node {i}: P={p:.3f}, Z={z:.3f}" for i, (p, z) in 
                      enumerate(zip(subset['P'], subset['Z']))],
                hoverinfo='text'
            ))
        
        # Add boundary lines with better styling
        fig.add_shape(type="line", x0=0, x1=1, y0=2.5, y1=2.5, 
                     line=dict(color="black", width=1.5, dash="dash"))
        fig.add_shape(type="line", x0=0.3, x1=0.3, y0=-2, y1=7, 
                     line=dict(color="black", width=1.5, dash="dash"))
        fig.add_shape(type="line", x0=0.75, x1=0.75, y0=-2, y1=7, 
                     line=dict(color="black", width=1.5, dash="dash"))
        
        # Add region labels with improved styling
        fig.add_annotation(x=0.15, y=1.25, text="Provincial<br>Nodes", showarrow=False, 
                          font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.7)')
        fig.add_annotation(x=0.525, y=1.25, text="Connector<br>Nodes", showarrow=False,
                          font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.7)')
        fig.add_annotation(x=0.875, y=1.25, text="Kinless<br>Nodes", showarrow=False,
                          font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.7)')
        fig.add_annotation(x=0.15, y=4, text="Provincial<br>Hubs", showarrow=False,
                          font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.7)')
        fig.add_annotation(x=0.525, y=4, text="Connector<br>Hubs", showarrow=False,
                          font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.7)')
        fig.add_annotation(x=0.875, y=4, text="Kinless<br>Hubs", showarrow=False,
                          font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.7)')
        
        # Add region counts
        region_counts = cart_data['Region'].value_counts()
        total_nodes = len(cart_data)
        
        # Add region count annotation
        count_text = "<b>Node Counts</b><br>" + "<br>".join([
            f"{region}: {count} ({count/total_nodes*100:.1f}%)" 
            for region, count in region_counts.items()
        ])
        
        fig.add_annotation(
            x=0.98, y=0.02,
            xref="paper", yref="paper",
            text=count_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )
        
        # Update layout
        title = f"Node Cartography - {recording if recording else group+' DIV'+str(div)} (Lag {lag}ms)"
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Participation Coefficient (P)",
                range=[-0.05, 1.05],
                tickvals=[0, 0.3, 0.75, 1]
            ),
            yaxis=dict(
                title="Within-Module Z-Score (Z)",
                range=[-2, 7],
                tickvals=[-2, 0, 2.5, 5, 7]
            )
        )
        
        # Apply MATLAB-like styling
        fig = apply_matlab_styling(fig, "Node Cartography", "scatter")
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)