import os
import re
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask

# Create Flask server
server = Flask(__name__)

# Create assets folder if it doesn't exist
assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
os.makedirs(assets_dir, exist_ok=True)

# Create custom CSS file
css_content = """
.js-plotly-plot, .plot-container, .plotly {
    width: 100% !important;
    height: 100% !important;
    display: block !important;
}

.main-svg {
    display: block !important;
}

.dashboard-container {
    padding: 15px;
    background-color: #f9f9f9;
}

.header {
    background-color: #4CAF50;
    color: white;
    padding: 15px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.filter-container {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}

.visualization-container {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 15px;
    height: 600px;
}

.graph-wrapper {
    height: 100%;
    width: 100%;
}
"""

with open(os.path.join(assets_dir, 'custom.css'), 'w') as f:
    f.write(css_content)

# Create Dash app
app = dash.Dash(
    __name__, 
    server=server,
    title="MEA Network Analysis Dashboard",
    suppress_callback_exceptions=True,
    assets_folder=assets_dir
)

class MEADataProcessor:
    """Class for loading and processing MEA-NAP data."""
    
    def __init__(self):
        """Initialize the MEA data processor."""
        self.data_dir = None
        self.graph_data_dir = None
        self.groups = []
        self.divs = []
        self.recordings = {}
        self.recording_data = {}
        self.lag_values = []
        
        # Neuronal and network metrics based on sample output folders
        self.neuronal_metrics = {
            'NodeByGroup': [
                'Firing Rate', 
                'Active Firing Rate', 
                'Burst Rate', 
                'Within-Burst Firing Rate',
                'Burst Duration', 
                'ISI Within Burst', 
                'ISI Outside Burst', 
                'Fraction Spikes in Bursts'
            ],
            'NodeByAge': [
                'Firing Rate', 
                'Active Firing Rate', 
                'Burst Rate', 
                'Within-Burst Firing Rate',
                'Burst Duration', 
                'ISI Within Burst', 
                'ISI Outside Burst', 
                'Fraction Spikes in Bursts'
            ],
            'RecordingsByGroup': [
                'Number of Active Electrodes',
                'Mean Firing Rate',
                'Median Firing Rate',
                'Network Burst Rate',
                'Mean Channels in Network Bursts',
                'Mean Network Burst Length',
                'Mean ISI Within Network Burst',
                'Mean ISI Outside Network Bursts',
                'CV of Inter-Network Burst Intervals',
                'Fraction of Bursts in Network Bursts',
                'Single-Electrode Burst Rate',
                'Single-Electrode Burst Duration',
                'Single-Electrode ISI Within Burst',
                'Single-Electrode ISI Outside Burst',
                'Mean Fraction Spikes in Bursts'
            ],
            'RecordingsByAge': [
                'Number of Active Electrodes',
                'Mean Firing Rate',
                'Median Firing Rate',
                'Network Burst Rate',
                'Mean Channels in Network Bursts',
                'Mean Network Burst Length',
                'Mean ISI Within Network Burst',
                'Mean ISI Outside Network Bursts',
                'CV of Inter-Network Burst Intervals',
                'Fraction of Bursts in Network Bursts',
                'Single-Electrode Burst Rate',
                'Single-Electrode Burst Duration',
                'Single-Electrode ISI Within Burst',
                'Single-Electrode ISI Outside Burst',
                'Mean Fraction Spikes in Bursts'
            ]
        }
        
        self.network_metrics = {
            'NodeByGroup': [
                'Node Degree',
                'Edge Weight',
                'Node Strength',
                'Local Efficiency',
                'Within-Module Degree Z-Score',
                'Betweenness Centrality',
                'Participation Coefficient',
                'Average Controllability',
                'Modal Controllability'
            ],
            'NodeByAge': [
                'Node Degree',
                'Edge Weight',
                'Node Strength',
                'Local Efficiency',
                'Within-Module Degree Z-Score',
                'Betweenness Centrality',
                'Participation Coefficient',
                'Average Controllability',
                'Modal Controllability'
            ],
            'RecordingsByGroup': [
                'Network Size',
                'Density',
                'Node Degree Mean',
                'Top 25% Node Degree',
                'Significant Edge Weight Mean',
                'Top 10% Edge Weight Mean',
                'Node Strength Mean',
                'Local Efficiency Mean',
                'Clustering Coefficient',
                'Number of Modules',
                'Modularity Score',
                'Percentage Z-Score > 0',
                'Percentage Z-Score < 0',
                'Mean Path Length',
                'Participation Coefficient Mean',
                'Bottom 10% PC',
                'Top 10% PC',
                'Global Efficiency',
                'Peripheral Nodes',
                'Non-hub Connectors',
                'Non-hub Kinless',
                'Provincial Hubs',
                'Connector Hubs',
                'Kinless Hubs',
                'Small Worldness Sigma',
                'Small Worldness Omega',
                'Mean Average Controllability',
                'Num NMF Components',
                'NMF Div Network Size',
                'Effective Rank'
            ],
            'RecordingsByAge': [
                'Network Size',
                'Density',
                'Node Degree Mean',
                'Top 25% Node Degree',
                'Significant Edge Weight Mean',
                'Top 10% Edge Weight Mean',
                'Node Strength Mean',
                'Local Efficiency Mean',
                'Clustering Coefficient',
                'Number of Modules',
                'Modularity Score',
                'Percentage Z-Score > 0',
                'Percentage Z-Score < 0',
                'Mean Path Length',
                'Participation Coefficient Mean',
                'Bottom 10% PC',
                'Top 10% PC',
                'Global Efficiency',
                'Peripheral Nodes',
                'Non-hub Connectors',
                'Non-hub Kinless',
                'Provincial Hubs',
                'Connector Hubs',
                'Kinless Hubs',
                'Small Worldness Sigma',
                'Small Worldness Omega',
                'Mean Average Controllability',
                'Num NMF Components',
                'NMF Div Network Size',
                'Effective Rank'
            ],
            'GraphMetricsByLag': [
                'Network Size',
                'Density',
                'Node Degree Mean',
                'Top 25% Node Degree',
                'Significant Edge Weight Mean',
                'Top 10% Edge Weight Mean',
                'Node Strength Mean',
                'Local Efficiency Mean',
                'Clustering Coefficient',
                'Number of Modules',
                'Modularity Score',
                'Percentage Z-Score > 0',
                'Percentage Z-Score < 0',
                'Mean Path Length',
                'Participation Coefficient Mean',
                'Bottom 10% PC',
                'Top 10% PC',
                'Global Efficiency',
                'Small Worldness Sigma',
                'Small Worldness Omega'
            ],
            'NodeCartography': ['Z-Score', 'Participation Coefficient']
        }
        
        # Field mapping for MATLAB data
        self.field_mapping = {
            # Neuronal metrics - node level
            'Firing Rate': 'FR',
            'Active Firing Rate': 'activeFR',
            'Burst Rate': 'channelBurstRate',
            'Within-Burst Firing Rate': 'channelWithinBurstFR',
            'Burst Duration': 'channelBurstDur',
            'ISI Within Burst': 'channelISIwithinBurst',
            'ISI Outside Burst': 'channelISIoutsideBurst',
            'Fraction Spikes in Bursts': 'channelFracSpikesInBursts',
            
            # Neuronal metrics - recording level
            'Number of Active Electrodes': 'active_electrodes',
            'Mean Firing Rate': 'FR',
            'Median Firing Rate': 'medianFR',
            'Network Burst Rate': 'networkBurstRate',
            'Mean Channels in Network Bursts': 'meanNumChannelsInNB',
            'Mean Network Burst Length': 'meanNetworkBurstLength',
            'Mean ISI Within Network Burst': 'meanISIwithinNetworkBurst',
            'Mean ISI Outside Network Bursts': 'meanISIoutsideNetworkBurst',
            'CV of Inter-Network Burst Intervals': 'CVofINBI',
            'Fraction of Bursts in Network Bursts': 'fractionBurstsInNetworkBursts',
            'Single-Electrode Burst Rate': 'burstRate',
            'Single-Electrode Burst Duration': 'burstDuration',
            'Single-Electrode ISI Within Burst': 'ISIwithinBurst',
            'Single-Electrode ISI Outside Burst': 'ISIoutsideBurst',
            'Mean Fraction Spikes in Bursts': 'fracSpikesInBursts',
            
            # Network metrics - node level
            'Node Degree': 'degree',
            'Edge Weight': 'edgeWeight',
            'Node Strength': 'strength',
            'Local Efficiency': 'efficiency_local',
            'Within-Module Degree Z-Score': 'z',
            'Betweenness Centrality': 'betweenness',
            'Participation Coefficient': 'participation',
            'Average Controllability': 'averageControlability',
            'Modal Controllability': 'modalControlability',
            
            # Network metrics - recording level
            'Network Size': 'network_size',
            'Density': 'density',
            'Node Degree Mean': 'nodeDegree_mean',
            'Top 25% Node Degree': 'nodeDegree_top25',
            'Significant Edge Weight Mean': 'edgeWeight_mean',
            'Top 10% Edge Weight Mean': 'edgeWeight_top10',
            'Node Strength Mean': 'nodeStrength_mean',
            'Local Efficiency Mean': 'efficiency_local_mean',
            'Clustering Coefficient': 'clustering',
            'Number of Modules': 'nModules',
            'Modularity Score': 'modularity',
            'Percentage Z-Score > 0': 'percentageZScorePos',
            'Percentage Z-Score < 0': 'percentageZScoreNeg',
            'Mean Path Length': 'path_length',
            'Participation Coefficient Mean': 'PC_mean',
            'Bottom 10% PC': 'PC_bottom10',
            'Top 10% PC': 'PC_top10',
            'Global Efficiency': 'efficiency_global',
            'Peripheral Nodes': 'NC1PeripheralNodes',
            'Non-hub Connectors': 'NC2NonhubConnectors',
            'Non-hub Kinless': 'NC3NonhubKinless',
            'Provincial Hubs': 'NC4ProvincialHubs',
            'Connector Hubs': 'NC5ConnectorHubs',
            'Kinless Hubs': 'NC6KinlessHubs',
            'Small Worldness Sigma': 'smallworldness_sigma',
            'Small Worldness Omega': 'smallworldness_omega',
            'Mean Average Controllability': 'meanAverageControlability',
            'Num NMF Components': 'nNMF',
            'NMF Div Network Size': 'nNMFDivNetworkSize',
            'Effective Rank': 'effectiveRank',
            
            # Simplified mapping for cartography
            'Z-Score': 'z'
        }
    
    def set_data_directory(self, data_dir):
        """Set the data directory and locate the GraphData folder."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find GraphData folder
        self.graph_data_dir = self._find_graph_data_folder()
        if self.graph_data_dir is None:
            raise FileNotFoundError(f"GraphData folder not found in {data_dir}")
        
        print(f"Using GraphData folder: {self.graph_data_dir}")
        
        # Scan data and load it
        self._scan_data()
        self._load_all_data()
        
        return {
            'groups': self.groups,
            'divs': self.divs,
            'neuronal_metrics': self.neuronal_metrics,
            'network_metrics': self.network_metrics,
            'lag_values': self.lag_values
        }
    
    def _find_graph_data_folder(self):
        """Find the GraphData folder within the data directory."""
        # Check if this is the GraphData folder itself
        if self.data_dir.name == 'GraphData':
            return self.data_dir
        
        # Check direct child
        graph_data = self.data_dir / 'GraphData'
        if graph_data.exists() and graph_data.is_dir():
            return graph_data
        
        # Look one level down
        for item in self.data_dir.iterdir():
            if item.is_dir():
                if item.name == 'GraphData':
                    return item
                sub_graph_data = item / 'GraphData'
                if sub_graph_data.exists() and sub_graph_data.is_dir():
                    return sub_graph_data
        
        return None
    
    def _scan_data(self):
        """Scan the GraphData folder for groups, recordings, DIVs, and lag values."""
        self.groups = []
        self.divs = set()
        self.recordings = {}
        found_lags = set()
        
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
                        
                        # Extract DIV using regex
                        div_match = re.search(r'DIV(\d+)', recording)
                        if div_match:
                            self.divs.add(int(div_match.group(1)))
                        
                        # Find lag values
                        for lag_file in recording_dir.glob(f"{recording}_networkMetrics_lag*.mat"):
                            lag_match = re.search(r'lag(\d+)', lag_file.name)
                            if lag_match:
                                found_lags.add(int(lag_match.group(1)))
        
        # Sort groups and divs
        self.groups.sort()
        self.divs = sorted(list(self.divs))
        
        # Set lag values
        if found_lags:
            self.lag_values = sorted(list(found_lags))
        else:
            self.lag_values = [10, 25, 50]  # Default if none found
    
    def _load_all_data(self):
        """Load data for all recordings."""
        for group in self.groups:
            for recording in self.recordings[group]:
                recording_dir = self.graph_data_dir / group / recording
                
                # Load electrode activity data
                activity_file = recording_dir / f"{recording}_electrodeSpikeActivity.mat"
                if activity_file.exists():
                    self._load_electrode_activity(group, recording, activity_file)
                
                # Load network metrics data for each lag value
                for lag in self.lag_values:
                    node_metrics_file = recording_dir / f"{recording}_nodeMetrics_lag{lag}.mat"
                    network_metrics_file = recording_dir / f"{recording}_networkMetrics_lag{lag}.mat"
                    
                    if node_metrics_file.exists():
                        self._load_node_metrics(group, recording, node_metrics_file, lag)
                    
                    if network_metrics_file.exists():
                        self._load_network_metrics(group, recording, network_metrics_file, lag)
    
    def _load_electrode_activity(self, group, recording, file_path):
        """Load electrode activity data from a .mat file."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        try:
            data = scipy.io.loadmat(str(file_path))
            if 'activityData' in data:
                self.recording_data[(group, recording)]['electrode_activity'] = data['activityData']
                print(f"Loaded electrode activity for {recording}")
        except Exception as e:
            print(f"Error loading electrode activity for {recording}: {e}")
    
    def _load_node_metrics(self, group, recording, file_path, lag):
        """Load node metrics data from a .mat file."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        try:
            data = scipy.io.loadmat(str(file_path))
            if 'nodeMetrics' in data:
                self.recording_data[(group, recording)][f'node_metrics_{lag}'] = data['nodeMetrics']
                print(f"Loaded node metrics (lag {lag}) for {recording}")
        except Exception as e:
            print(f"Error loading node metrics for {recording}: {e}")
            # Create fallback structure with zeros
            self.recording_data[(group, recording)][f'node_metrics_{lag}'] = {
                'degree': np.array([[0.0]]),
                'betweenness': np.array([[0.0]]),
                'participation': np.array([[0.0]]),
                'efficiency_local': np.array([[0.0]]),
                'z': np.array([[0.0]])
            }
    
    def _load_network_metrics(self, group, recording, file_path, lag):
        """Load network metrics data from a .mat file."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        default_metrics = {
            'density': np.array([[0.0]]),
            'efficiency_global': np.array([[0.0]]),
            'modularity': np.array([[0.0]]),
            'clustering': np.array([[0.0]]),
            'smallworldness': np.array([[0.0]])
        }
        
        try:
            if not os.path.exists(str(file_path)):
                print(f"Network metrics file not found: {file_path}")
                self.recording_data[(group, recording)][f'network_metrics_{lag}'] = default_metrics
                return
            
            data = scipy.io.loadmat(str(file_path))
            
            if 'netMetrics' in data and hasattr(data['netMetrics'], 'dtype'):
                self.recording_data[(group, recording)][f'network_metrics_{lag}'] = data['netMetrics']
                print(f"Loaded network metrics (lag {lag}) for {recording}")
            else:
                self.recording_data[(group, recording)][f'network_metrics_{lag}'] = default_metrics
        except Exception as e:
            print(f"Error loading network metrics for {recording}: {e}")
            self.recording_data[(group, recording)][f'network_metrics_{lag}'] = default_metrics
    
    def _get_field_name(self, readable_name):
        """Get the field name corresponding to a readable metric name."""
        if readable_name in self.field_mapping:
            return self.field_mapping[readable_name]
        
        # Fallback: convert to lowercase and replace spaces with underscores
        return readable_name.lower().replace(' ', '_')
    
    def _find_field(self, struct, field_pattern):
        """Find a field in a MATLAB struct that matches a pattern."""
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
        if not hasattr(struct, 'dtype') or not hasattr(struct.dtype, 'names') or struct.dtype.names is None:
            return None
        
        # Direct match
        if field_pattern in struct.dtype.names:
            return field_pattern
        
        # Case-insensitive search
        for name in struct.dtype.names:
            if isinstance(name, str) and isinstance(field_pattern, str) and field_pattern.lower() in name.lower():
                return name
        
        return None
    
    def get_neuronal_node_data(self, selected_groups=None, selected_divs=None, metric='Firing Rate'):
        """Extract neuronal node data from electrode activity data."""
        data = []
        
        # Handle empty selected_divs
        if not selected_divs:
            selected_divs = self.divs
        
        # Get field name
        metric_field = self._get_field_name(metric)
        
        # Process each recording
        for (group, recording), rec_data in self.recording_data.items():
            # Apply group filter
            if selected_groups and group not in selected_groups:
                continue
            
            # Extract DIV from recording name
            div_match = re.search(r'DIV(\d+)', recording)
            if not div_match:
                continue
            
            div = int(div_match.group(1))
            
            # Apply DIV filter
            if div not in selected_divs:
                continue
            
            # Get electrode activity data
            if 'electrode_activity' not in rec_data:
                continue
            
            activity = rec_data['electrode_activity']
            
            # Find the requested metric
            field = self._find_field(activity, metric_field)
            if not field:
                continue
            
            # Extract values
            try:
                values = activity[field][0, 0]
                if not isinstance(values, np.ndarray):
                    continue
                
                values = values.flatten()
                
                # Find channels if available
                channels = None
                if hasattr(activity, 'dtype') and 'channels' in activity.dtype.names:
                    channels = activity['channels'][0, 0]
                    if isinstance(channels, np.ndarray):
                        channels = channels.flatten()
                
                # Use sequential numbers if no channels found
                if channels is None or len(channels) != len(values):
                    channels = np.arange(1, len(values) + 1)
                
                # Add data points
                for i, (channel, value) in enumerate(zip(channels, values)):
                    if np.isfinite(value):  # Skip NaN/inf values
                        data.append({
                            'Group': group,
                            'DIV': div,
                            'Recording': recording,
                            'Channel': int(channel) if isinstance(channel, np.integer) else i+1,
                            'Value': float(value)
                        })
            except Exception as e:
                print(f"Error extracting {field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_network_node_data(self, selected_groups=None, selected_divs=None, metric='Degree', lag=10):
        """Extract network node data from node metrics data."""
        data = []
        
        # Handle empty selected_divs
        if not selected_divs:
            selected_divs = self.divs
        
        # Get field name
        metric_field = self._get_field_name(metric)
        
        # Process each recording
        for (group, recording), rec_data in self.recording_data.items():
            # Apply group filter
            if selected_groups and group not in selected_groups:
                continue
            
            # Extract DIV from recording name
            div_match = re.search(r'DIV(\d+)', recording)
            if not div_match:
                continue
            
            div = int(div_match.group(1))
            
            # Apply DIV filter
            if div not in selected_divs:
                continue
            
            # Get node metrics data
            metrics_key = f'node_metrics_{lag}'
            if metrics_key not in rec_data:
                continue
            
            node_metrics = rec_data[metrics_key]
            
            # Find the requested metric
            field = self._find_field(node_metrics, metric_field)
            if not field:
                continue
            
            # Extract values
            try:
                values = node_metrics[field][0, 0]
                if not isinstance(values, np.ndarray):
                    continue
                
                values = values.flatten()
                
                # Find channels if available
                channels = None
                if hasattr(node_metrics, 'dtype') and 'channels' in node_metrics.dtype.names:
                    channels = node_metrics['channels'][0, 0]
                    if isinstance(channels, np.ndarray):
                        channels = channels.flatten()
                
                # Use sequential numbers if no channels found
                if channels is None or len(channels) != len(values):
                    channels = np.arange(1, len(values) + 1)
                
                # Add data points
                for i, (channel, value) in enumerate(zip(channels, values)):
                    if np.isfinite(value):  # Skip NaN/inf values
                        data.append({
                            'Group': group,
                            'DIV': div,
                            'Recording': recording,
                            'Channel': int(channel) if isinstance(channel, np.integer) else i+1,
                            'Value': float(value)
                        })
            except Exception as e:
                print(f"Error extracting {field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_network_recording_data(self, selected_groups=None, selected_divs=None, metric='Density', lag=10):
        """Extract network recording data from network metrics data."""
        data = []
        
        # Handle empty selected_divs
        if not selected_divs:
            selected_divs = self.divs
        
        # Handle None metric
        if metric is None:
            metric = 'Density'
        
        # Get field name
        metric_field = self._get_field_name(metric)
        
        # Process each recording
        for (group, recording), rec_data in self.recording_data.items():
            # Apply group filter
            if selected_groups and group not in selected_groups:
                continue
            
            # Extract DIV from recording name
            div_match = re.search(r'DIV(\d+)', recording)
            if not div_match:
                continue
            
            div = int(div_match.group(1))
            
            # Apply DIV filter
            if div not in selected_divs:
                continue
            
            # Get network metrics data
            metrics_key = f'network_metrics_{lag}'
            if metrics_key not in rec_data:
                continue
            
            net_metrics = rec_data[metrics_key]
            
            # Handle dictionary-like structure
            if isinstance(net_metrics, dict):
                if metric_field in net_metrics:
                    try:
                        value = net_metrics[metric_field]
                        # Convert value to scalar
                        if hasattr(value, 'item'):
                            value = value.item()
                        elif isinstance(value, np.ndarray) and value.size > 0:
                            value = value[0][0] if value.ndim > 1 else value[0]
                        
                        # Only add finite values
                        if np.isfinite(value):
                            data.append({
                                'Group': group,
                                'DIV': div,
                                'Recording': recording,
                                'Value': float(value)
                            })
                    except Exception as e:
                        print(f"Error extracting {metric_field} from dict for {recording}: {e}")
                continue
            
            # Skip if netMetrics is None
            if net_metrics is None:
                continue
            
            # Find the requested metric field
            field = self._find_field(net_metrics, metric_field)
            if not field:
                continue
            
            # Extract value
            try:
                value = net_metrics[field][0, 0]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                
                # Only add finite scalars
                if np.isscalar(value) and np.isfinite(value):
                    data.append({
                        'Group': group,
                        'DIV': div,
                        'Recording': recording,
                        'Value': float(value)
                    })
            except Exception as e:
                print(f"Error extracting {field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_node_cartography_data(self, group, div, lag, recording=None):
        """Get node cartography data (Participation Coefficient vs. Z-Score)."""
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
                return None  # No matching recording found
        
        # Check if we have data for this recording
        if (group, recording) not in self.recording_data:
            return None
        
        rec_data = self.recording_data[(group, recording)]
        metrics_key = f'node_metrics_{lag}'
        
        if metrics_key not in rec_data:
            return None
        
        node_metrics = rec_data[metrics_key]
        
        # Find participation coefficient and Z-score fields
        p_field = self._find_field(node_metrics, 'participation')
        z_field = self._find_field(node_metrics, 'z')
        
        if not p_field or not z_field:
            return None
        
        # Extract P and Z values
        try:
            if node_metrics[p_field] is None or node_metrics[z_field] is None:
                return None
            
            p_values = node_metrics[p_field][0, 0].flatten()
            z_values = node_metrics[z_field][0, 0].flatten()
            
            # Create data frame with P and Z
            df = pd.DataFrame({
                'P': p_values,
                'Z': z_values
            })
            
            # Remove NaN values
            df = df.dropna()
            
            if df.empty:
                return None
            
            return df
        except Exception as e:
            print(f"Error extracting cartography data: {e}")
            return None


# Initialize data processor
data_processor = MEADataProcessor()

# Create app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('MEA Network Analysis Dashboard', style={'margin': '0'}),
        html.P("Interactive visualization of Microelectrode Array data from the MEA-NAP pipeline",
               style={'margin': '0 0 15px 0'})
    ], className='header'),
    
    # Main content container
    html.Div([
        # Data loading section
        html.Div([
            html.Div([
                html.Label('Data Directory:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Input(
                    id='data-dir-input', 
                    type='text', 
                    placeholder='Enter path to folder containing GraphData',
                    style={'width': '70%', 'marginRight': '10px', 'padding': '8px'}
                ),
                html.Button(
                    'Load Data', 
                    id='load-data-button', 
                    style={'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 
                          'padding': '8px 15px', 'borderRadius': '4px', 'cursor': 'pointer'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='status-message', style={'marginTop': '10px', 'fontWeight': 'bold'})
        ], className='filter-container'),
        
        # Main dashboard (hidden until data is loaded)
        html.Div(id='dashboard-container', style={'display': 'none'}, children=[
            # Tab organization
            html.Div([
                # Main Tabs: Activity Type
                dcc.Tabs(id='activity-tabs', value='neuronal', children=[
                    # Tab 1: Neuronal Activity
                    dcc.Tab(label='Neuronal Activity', value='neuronal', children=[
                        # Subtabs for Neuronal Activity
                        dcc.Tabs(id='neuronal-tabs', value='nodebygroup', children=[
                            dcc.Tab(label='Node By Group', value='nodebygroup'),
                            dcc.Tab(label='Node By Age', value='nodebyage'),
                            dcc.Tab(label='Recordings By Group', value='recordingsbygroup'),
                            dcc.Tab(label='Recordings By Age', value='recordingsbyage')
                        ])
                    ]),
                    
                    # Tab 2: Network Activity
                    dcc.Tab(label='Network Activity', value='network', children=[
                        # Subtabs for Network Activity
                        dcc.Tabs(id='network-tabs', value='nodebygroup', children=[
                            dcc.Tab(label='Node By Group', value='nodebygroup'),
                            dcc.Tab(label='Node By Age', value='nodebyage'),
                            dcc.Tab(label='Recordings By Group', value='recordingsbygroup'),
                            dcc.Tab(label='Recordings By Age', value='recordingsbyage'),
                            dcc.Tab(label='Graph Metrics By Lag', value='graphmetricsbylag'),
                            dcc.Tab(label='Node Cartography', value='nodecartography')
                        ])
                    ])
                ], style={'marginBottom': '20px'})
            ], className='filter-container'),
            
            # Filters and visualization container
            html.Div([
                # Left column - filters
                html.Div([
                    html.H3('Filters', style={'marginTop': '0', 'marginBottom': '15px'}),
                    
                    # Group selection
                    html.Div([
                        html.Label('Groups:', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='group-dropdown',
                            multi=True
                        )
                    ], style={'marginBottom': '15px'}),
                    
                    # DIV selection
                    html.Div([
                        html.Label('DIVs:', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='div-dropdown',
                            multi=True
                        )
                    ], style={'marginBottom': '15px'}),
                    
                    # Metric selection
                    html.Div([
                        html.Label('Metric:', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='metric-dropdown'
                        )
                    ], style={'marginBottom': '15px'}),
                    
                    # Lag selection
                    html.Div([
                        html.Label('Lag (ms):', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='lag-dropdown'
                        )
                    ], id='lag-dropdown-container', style={'marginBottom': '15px', 'display': 'none'}),
                    
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
                            labelStyle={'display': 'block', 'marginBottom': '5px'}
                        )
                    ])
                ], style={'width': '25%', 'padding': '20px'}),
                
                # Right column - visualization
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='visualization-graph',
                            style={'height': '100%', 'width': '100%'},
                            config={'displayModeBar': True, 'responsive': True}
                        )
                    ], className='graph-wrapper')
                ], className='visualization-container', style={'width': '75%'})
            ], style={'display': 'flex'})
        ]),
        
        # Stores for various data
        dcc.Store(id='data-store'),
        dcc.Store(id='current-comparison-store')
    ], className='dashboard-container')
])

# Callback to load data
@app.callback(
    [Output('status-message', 'children'),
     Output('status-message', 'style'),
     Output('dashboard-container', 'style'),
     Output('data-store', 'data'),
     Output('group-dropdown', 'options'),
     Output('group-dropdown', 'value'),
     Output('div-dropdown', 'options'),
     Output('div-dropdown', 'value'),
     Output('lag-dropdown', 'options'),
     Output('lag-dropdown', 'value')],
    [Input('load-data-button', 'n_clicks')],
    [State('data-dir-input', 'value')]
)
def load_data(n_clicks, data_dir):
    if n_clicks is None or not data_dir:
        return (
            "", 
            {}, 
            {'display': 'none'}, 
            None,
            [], [], [], [], [], []
        )
    
    try:
        # Load data
        result = data_processor.set_data_directory(data_dir)
        groups = result['groups']
        divs = result['divs']
        lag_values = result['lag_values']
        
        # Prepare dropdown options
        group_options = [{'label': g, 'value': g} for g in groups]
        div_options = [{'label': f'DIV {div}', 'value': div} for div in divs]
        lag_options = [{'label': f'{lag} ms', 'value': lag} for lag in lag_values]
        
        # Return success message and show dashboard
        return (
            f"Data loaded successfully. Found {len(groups)} groups and {len(divs)} DIVs.",
            {'color': 'green', 'fontWeight': 'bold', 'padding': '10px', 
             'backgroundColor': 'rgba(76,175,80,0.1)', 'borderLeft': '4px solid #4CAF50', 
             'borderRadius': '4px'},
            {'display': 'block'},
            {
                'groups': groups,
                'divs': divs,
                'lag_values': lag_values,
                'neuronal_metrics': result['neuronal_metrics'],
                'network_metrics': result['network_metrics']
            },
            group_options,
            groups,  # Select all groups by default
            div_options,
            divs,  # Select all DIVs by default
            lag_options,
            lag_values[0] if lag_values else 10  # Default to first lag value
        )
    except Exception as e:
        # Return error message
        return (
            f"Error loading data: {str(e)}",
            {'color': 'red', 'fontWeight': 'bold', 'padding': '10px', 
             'backgroundColor': 'rgba(244,67,54,0.1)', 'borderLeft': '4px solid #F44336', 
             'borderRadius': '4px'},
            {'display': 'none'},
            None,
            [], [], [], [], [], []
        )

# Callback to store the current comparison selection
@app.callback(
    Output('current-comparison-store', 'data'),
    [Input('neuronal-tabs', 'value'),
     Input('network-tabs', 'value'),
     Input('activity-tabs', 'value')]
)
def store_current_comparison(neuronal_tab, network_tab, activity_tab):
    """Store the currently selected tab combination"""
    comparison_tab = neuronal_tab if activity_tab == 'neuronal' else network_tab
    return {
        'activity': activity_tab,
        'comparison': comparison_tab
    }

# Callback to update metric options
@app.callback(
    [Output('metric-dropdown', 'options'),
     Output('metric-dropdown', 'value'),
     Output('lag-dropdown-container', 'style')],
    [Input('current-comparison-store', 'data'),
     Input('data-store', 'data')],
    prevent_initial_call=True
)
def update_metric_options(current_selection, data):
    """Update the metric dropdown based on the current activity and comparison tabs"""
    if not data or not current_selection:
        return [], None, {'display': 'none'}
    
    activity = current_selection.get('activity')
    comparison = current_selection.get('comparison')
    
    if not activity or not comparison:
        return [], None, {'display': 'none'}
    
    # Get metrics based on activity and comparison
    if activity == 'neuronal':
        metrics_dict = data.get('neuronal_metrics', {})
        if comparison in ['nodebygroup', 'nodebyage']:
            metrics = metrics_dict.get('NodeByGroup', ['Firing Rate'])
            default_metric = 'Firing Rate' if 'Firing Rate' in metrics else metrics[0] if metrics else None
            show_lag = False
        else:  # recordingsbygroup, recordingsbyage
            metrics = metrics_dict.get('RecordingsByGroup', ['Mean Firing Rate'])
            default_metric = 'Mean Firing Rate' if 'Mean Firing Rate' in metrics else metrics[0] if metrics else None
            show_lag = False
    else:  # network
        metrics_dict = data.get('network_metrics', {})
        if comparison in ['nodebygroup', 'nodebyage']:
            metrics = metrics_dict.get('NodeByGroup', ['Node Degree'])
            default_metric = 'Node Degree' if 'Node Degree' in metrics else metrics[0] if metrics else None
            show_lag = True
        elif comparison in ['recordingsbygroup', 'recordingsbyage']:
            metrics = metrics_dict.get('RecordingsByGroup', ['Density'])
            default_metric = 'Density' if 'Density' in metrics else metrics[0] if metrics else None
            show_lag = True
        elif comparison == 'graphmetricsbylag':
            metrics = metrics_dict.get('GraphMetricsByLag', ['Density'])
            default_metric = 'Density' if 'Density' in metrics else metrics[0] if metrics else None
            show_lag = False  # No lag dropdown for lag comparison
        elif comparison == 'nodecartography':
            metrics = []  # No metric selection for cartography
            default_metric = None
            show_lag = True
    
    # Create options and set default value
    metric_options = [{'label': metric, 'value': metric} for metric in metrics]
    
    # Show/hide lag dropdown
    lag_style = {'display': 'block' if show_lag else 'none', 'marginBottom': '15px'}
    
    return metric_options, default_metric, lag_style

# Callback for visualization
@app.callback(
    Output('visualization-graph', 'figure'),
    [Input('group-dropdown', 'value'),
     Input('div-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('viz-type', 'value'),
     Input('lag-dropdown', 'value'),
     Input('current-comparison-store', 'data')],
    prevent_initial_call=True
)
def update_visualization(groups, selected_divs, metric, viz_type, lag, current_selection):
    """Generate visualizations based on selected filters."""
    # Check if we have groups and current selection
    if not groups or not current_selection:
        return {
            'data': [], 
            'layout': {
                'title': 'Please select at least one group',
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
        }
    
    activity = current_selection.get('activity')
    comparison = current_selection.get('comparison')
    
    if not activity or not comparison:
        return {
            'data': [], 
            'layout': {
                'title': 'No activity or comparison type selected',
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
        }
    
    try:
        # Get data based on activity and comparison type
        if comparison == 'nodecartography':
            # Node cartography is a special case
            if not groups or not selected_divs:
                return {
                    'data': [], 
                    'layout': {
                        'title': 'Please select group and DIV for node cartography',
                        'plot_bgcolor': 'white',
                        'paper_bgcolor': 'white'
                    }
                }
            
            # Get cartography data
            cart_data = data_processor.get_node_cartography_data(groups[0], selected_divs[0], lag)
            
            if cart_data is None or cart_data.empty:
                return {
                    'data': [], 
                    'layout': {
                        'title': 'No cartography data available for this selection',
                        'plot_bgcolor': 'white',
                        'paper_bgcolor': 'white'
                    }
                }
            
            # Classify nodes into regions
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
            
            # Create figure for cartography
            fig = px.scatter(
                cart_data, x='P', y='Z', color='Region',
                title=f"Node Cartography - {groups[0]} DIV{selected_divs[0]} (Lag {lag}ms)",
                labels={'P': 'Participation Coefficient', 'Z': 'Within-Module Z-Score'}
            )
            
            # Add boundary lines
            fig.add_shape(type="line", x0=0, x1=1, y0=2.5, y1=2.5, line=dict(color="black", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0.3, x1=0.3, y0=-2, y1=7, line=dict(color="black", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0.75, x1=0.75, y0=-2, y1=7, line=dict(color="black", width=1, dash="dash"))
            
            # Set axis ranges
            fig.update_xaxes(range=[-0.05, 1.05], title='Participation Coefficient (P)')
            fig.update_yaxes(range=[-2, 7], title='Within-Module Z-Score (Z)')
            
        elif comparison == 'graphmetricsbylag':
            # Graph metrics by lag visualization
            all_data = []
            
            for lag_val in data_processor.lag_values:
                df = data_processor.get_network_recording_data(groups, selected_divs, metric, lag_val)
                if not df.empty:
                    df['Lag'] = lag_val
                    all_data.append(df)
            
            if not all_data:
                return {
                    'data': [], 
                    'layout': {
                        'title': 'No data available for the selected filters',
                        'plot_bgcolor': 'white',
                        'paper_bgcolor': 'white'
                    }
                }
            
            combined_df = pd.concat(all_data)
            
            # Create line plot for lag comparison
            fig = px.line(
                combined_df, x='Lag', y='Value', color='Group', markers=True,
                title=f"{metric} by Lag Value",
                labels={'Value': metric, 'Lag': 'Lag (ms)'}
            )
            
        else:
            # Get data for regular node/recording metrics
            if activity == 'neuronal':
                if comparison in ['nodebygroup', 'nodebyage']:
                    df = data_processor.get_neuronal_node_data(groups, selected_divs, metric)
                else:
                    # For recordings, aggregate node data
                    node_df = data_processor.get_neuronal_node_data(groups, selected_divs, metric)
                    if not node_df.empty:
                        df = node_df.groupby(['Group', 'DIV', 'Recording'])['Value'].agg(['mean']).reset_index()
                        df.rename(columns={'mean': 'Value'}, inplace=True)
                    else:
                        df = pd.DataFrame()
            else:
                if comparison in ['nodebygroup', 'nodebyage']:
                    df = data_processor.get_network_node_data(groups, selected_divs, metric, lag)
                else:
                    df = data_processor.get_network_recording_data(groups, selected_divs, metric, lag)
            
            if df.empty:
                return {
                    'data': [], 
                    'layout': {
                        'title': 'No data available for the selected filters',
                        'plot_bgcolor': 'white',
                        'paper_bgcolor': 'white'
                    }
                }
            
            # Determine x and color variables
            x_var = 'Group' if comparison in ['nodebygroup', 'recordingsbygroup'] else 'DIV'
            color_var = 'DIV' if comparison in ['nodebygroup', 'recordingsbygroup'] else 'Group'
            
            # Set title
            title_suffix = f" (Lag {lag}ms)" if activity == 'network' else ""
            plot_title = f"{metric} by {x_var}{title_suffix}"
            
            # Create visualization based on type
            if viz_type == 'violin':
                fig = px.violin(
                    df, x=x_var, y='Value', color=color_var,
                    box=True, points='all',
                    title=plot_title,
                    labels={'Value': metric}
                )
                
                # Add marker trace for all points (ensures visibility)
                for group in df[color_var].unique():
                    subset = df[df[color_var] == group]
                    fig.add_trace(go.Scatter(
                        x=subset[x_var],
                        y=subset['Value'],
                        mode='markers',
                        marker=dict(
                            size=4
                        ),
                        name=f"{group} (points)",
                        showlegend=False
                    ))
                
            elif viz_type == 'box':
                fig = px.box(
                    df, x=x_var, y='Value', color=color_var,
                    points='outliers',
                    title=plot_title,
                    labels={'Value': metric}
                )
                
                # Add marker trace for all data points
                for group in df[color_var].unique():
                    subset = df[df[color_var] == group]
                    fig.add_trace(go.Scatter(
                        x=subset[x_var],
                        y=subset['Value'],
                        mode='markers',
                        marker=dict(
                            size=4
                        ),
                        name=f"{group} (points)",
                        showlegend=False
                    ))
                
            elif viz_type == 'bar':
                # Group data by x and color variables
                if x_var == 'DIV':
                    grouped = df.groupby(['DIV', 'Group'])['Value'].agg(['mean', 'sem']).reset_index()
                    fig = px.bar(
                        grouped, x='DIV', y='mean', color='Group',
                        error_y='sem',
                        title=plot_title,
                        labels={'mean': f"Mean {metric}", 'DIV': 'DIV', 'Group': 'Group'},
                        barmode='group'
                    )
                else:
                    grouped = df.groupby(['Group', 'DIV'])['Value'].agg(['mean', 'sem']).reset_index()
                    fig = px.bar(
                        grouped, x='Group', y='mean', color='DIV',
                        error_y='sem',
                        title=plot_title,
                        labels={'mean': f"Mean {metric}", 'DIV': 'DIV', 'Group': 'Group'},
                        barmode='group'
                    )
        
        # Apply common styling to all figures
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            margin=dict(l=80, r=40, t=60, b=80)
        )
        
        # Ensure axes are visible
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            mirror=True,
            showgrid=False
        )
        
        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            mirror=True,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(211,211,211,0.5)'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        return {
            'data': [], 
            'layout': {
                'title': f'Error: {str(e)}',
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'annotations': [{
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': f'An error occurred: {str(e)}',
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            }
        }

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)