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
        
        # Scan GraphData folder
        self.scan_data()
        
        return {
            'groups': self.groups,
            'divs': self.divs
        }
    
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
        """Scan GraphData folder for groups, recordings, and DIVs."""
        self.groups = []
        self.divs = set()
        self.recordings = {}
        
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
                        
                        # Load data for this recording
                        self.load_recording_data(group, recording, recording_dir)
        
        # Sort groups and divs
        self.groups.sort()
        self.divs = sorted(list(self.divs))
    
    def load_recording_data(self, group, recording, recording_dir):
        """Load data for a recording."""
        if (group, recording) not in self.recording_data:
            self.recording_data[(group, recording)] = {}
        
        # Load electrode activity data
        activity_file = recording_dir / f"{recording}_electrodeSpikeActivity.mat"
        if activity_file.exists():
            try:
                data = scipy.io.loadmat(str(activity_file))
                if 'activityData' in data:
                    self.recording_data[(group, recording)]['electrode_activity'] = data['activityData']
                    print(f"Loaded electrode activity for {recording}")
            except Exception as e:
                print(f"Error loading electrode activity for {recording}: {e}")
        
        # Load node metrics data
        for lag_file in recording_dir.glob(f"{recording}_nodeMetrics_lag*.mat"):
            lag_match = re.search(r'lag(\d+)', lag_file.name)
            if lag_match:
                lag = int(lag_match.group(1))
                try:
                    data = scipy.io.loadmat(str(lag_file))
                    if 'nodeMetrics' in data:
                        self.recording_data[(group, recording)][f'node_metrics_{lag}'] = data['nodeMetrics']
                        print(f"Loaded node metrics (lag {lag}) for {recording}")
                except Exception as e:
                    print(f"Error loading node metrics for {recording}: {e}")
        
        # Load network metrics data
        for lag_file in recording_dir.glob(f"{recording}_networkMetrics_lag*.mat"):
            lag_match = re.search(r'lag(\d+)', lag_file.name)
            if lag_match:
                lag = int(lag_match.group(1))
                try:
                    data = scipy.io.loadmat(str(lag_file))
                    if 'netMetrics' in data:
                        self.recording_data[(group, recording)][f'network_metrics_{lag}'] = data['netMetrics']
                        print(f"Loaded network metrics (lag {lag}) for {recording}")
                except Exception as e:
                    print(f"Error loading network metrics for {recording}: {e}")
    
    def get_neuronal_node_data(self, selected_groups=None, div_range=None, metric='FR'):
        """Extract neuronal node data from electrode activity."""
        data = []
        
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
            metric_field = self.find_field(activity, metric)
            if not metric_field:
                continue
            
            # Extract DIV
            div_match = re.search(r'DIV(\d+)', recording)
            div = int(div_match.group(1)) if div_match else 0
            
            # Extract values
            try:
                values = activity[metric_field][0, 0]
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
                print(f"Error extracting {metric_field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_network_node_data(self, selected_groups=None, div_range=None, metric='degree', lag=10):
        """Extract network node data from node metrics."""
        data = []
        
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
            metric_field = self.find_field(node_metrics, metric)
            if not metric_field:
                continue
            
            # Extract DIV
            div_match = re.search(r'DIV(\d+)', recording)
            div = int(div_match.group(1)) if div_match else 0
            
            # Extract values
            try:
                values = node_metrics[metric_field][0, 0]
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
                print(f"Error extracting {metric_field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def get_network_recording_data(self, selected_groups=None, div_range=None, metric='density', lag=10):
        """Extract network recording data from network metrics."""
        data = []
        
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
            
            # Extract network metrics data
            metrics_key = f'network_metrics_{lag}'
            if metrics_key not in rec_data:
                continue
                
            net_metrics = rec_data[metrics_key]
            
            # Find metric field
            metric_field = self.find_field(net_metrics, metric)
            if not metric_field:
                continue
            
            # Extract DIV
            div_match = re.search(r'DIV(\d+)', recording)
            div = int(div_match.group(1)) if div_match else 0
            
            # Extract value
            try:
                value = net_metrics[metric_field][0, 0]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                
                if np.isscalar(value) and np.isfinite(value):
                    data.append({
                        'Group': group,
                        'DIV': div,
                        'Recording': recording,
                        'Value': float(value)
                    })
            except Exception as e:
                print(f"Error extracting {metric_field} from {recording}: {e}")
        
        return pd.DataFrame(data)
    
    def find_field(self, struct, field_pattern):
        """Find a field in a struct matching a pattern."""
        if not hasattr(struct, 'dtype') or not hasattr(struct.dtype, 'names'):
            return None
            
        # Direct match
        if field_pattern in struct.dtype.names:
            return field_pattern
            
        # Case-insensitive search
        for name in struct.dtype.names:
            if field_pattern.lower() in name.lower():
                return name
                
        return None

# Initialize data analyzer
analyzer = MEADataAnalyzer()

# Define the app layout
app.layout = html.Div([
    # Header
    html.H1('MEA Network Analysis Dashboard', 
            style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f1f1f1'}),
    
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
        ]),
        
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
            {'color': 'green', 'fontWeight': 'bold'},
            {'display': 'block', 'padding': '0 20px'},
            {'groups': groups, 'divs': divs}
        )
    except Exception as e:
        # Return error message
        return (
            f"Error loading data: {str(e)}",
            {'color': 'red', 'fontWeight': 'bold'},
            {'display': 'none'},
            None
        )

# Callback to update comparison tabs based on activity selection
@app.callback(
    Output('comparison-tabs', 'children'),
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
        ]
    else:
        # Tabs for network activity
        return [
            dcc.Tab(label='Node By Group', value='nodebygroup'),
            dcc.Tab(label='Node By Age', value='nodebyage'),
            dcc.Tab(label='Recordings By Group', value='recordingsbygroup'),
            dcc.Tab(label='Recordings By Age', value='recordingsbyage'),
            dcc.Tab(label='Graph Metrics By Lag', value='graphmetricsbylag'),
            dcc.Tab(label='Node Cartography', value='nodecartography')
        ]

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
    
    # Node By Group or Node By Age content
    if comparison_value in ['nodebygroup', 'nodebyage']:
        grouping = 'Group' if comparison_value == 'nodebygroup' else 'DIV'
        
        # Determine which metrics to show based on activity type
        if activity_value == 'neuronal':
            metric_options = [
                {'label': 'Firing Rate', 'value': 'FR'},
                {'label': 'Burst Rate', 'value': 'channelBurstRate'},
                {'label': 'Burst Duration', 'value': 'channelBurstDur'},
                {'label': 'Fraction Spikes in Bursts', 'value': 'channelFracSpikesInBursts'},
                {'label': 'ISI Within Burst', 'value': 'channelISIwithinBurst'},
                {'label': 'ISI Outside Burst', 'value': 'channeISIoutsideBurst'}
            ]
            default_metric = 'FR'
            show_lag = False
        else:
            metric_options = [
                {'label': 'Degree', 'value': 'degree'},
                {'label': 'Betweenness Centrality', 'value': 'betweenness'},
                {'label': 'Participation Coefficient', 'value': 'participation'},
                {'label': 'Local Efficiency', 'value': 'efficiency_local'},
                {'label': 'Within-Module Z-Score', 'value': 'z'}
            ]
            default_metric = 'degree'
            show_lag = True
        
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters'),
                
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
                        options=[
                            {'label': '10 ms', 'value': 10},
                            {'label': '25 ms', 'value': 25},
                            {'label': '50 ms', 'value': 50}
                        ],
                        value=10
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
                        value='violin'
                    )
                ])
            ], style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='visualization-graph')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    # Recordings By Group or Recordings By Age content
    elif comparison_value in ['recordingsbygroup', 'recordingsbyage']:
        grouping = 'Group' if comparison_value == 'recordingsbygroup' else 'DIV'
        
        # Determine which metrics to show based on activity type
        if activity_value == 'neuronal':
            metric_options = [
                {'label': 'Mean Firing Rate', 'value': 'FR'},
                {'label': 'Number of Active Electrodes', 'value': 'active_electrodes'},
                {'label': 'Mean Burst Rate', 'value': 'channelBurstRate'},
                {'label': 'Mean Burst Duration', 'value': 'channelBurstDur'},
                {'label': 'Mean Fraction of Spikes in Bursts', 'value': 'channelFracSpikesInBursts'}
            ]
            default_metric = 'FR'
            show_lag = False
        else:
            metric_options = [
                {'label': 'Density', 'value': 'density'},
                {'label': 'Global Efficiency', 'value': 'efficiency_global'},
                {'label': 'Modularity', 'value': 'modularity'},
                {'label': 'Clustering Coefficient', 'value': 'clustering'},
                {'label': 'Small-Worldness', 'value': 'smallworldness'}
            ]
            default_metric = 'density'
            show_lag = True
        
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters'),
                
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
                        options=[
                            {'label': '10 ms', 'value': 10},
                            {'label': '25 ms', 'value': 25},
                            {'label': '50 ms', 'value': 50}
                        ],
                        value=10
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
                        value='half_violin'
                    )
                ])
            ], style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='rec-visualization-graph')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    # Graph Metrics By Lag content
    elif comparison_value == 'graphmetricsbylag':
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters'),
                
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
                        options=[
                            {'label': 'Density', 'value': 'density'},
                            {'label': 'Global Efficiency', 'value': 'efficiency_global'},
                            {'label': 'Modularity', 'value': 'modularity'},
                            {'label': 'Clustering Coefficient', 'value': 'clustering'},
                            {'label': 'Small-Worldness', 'value': 'smallworldness'}
                        ],
                        value='density'
                    )
                ], style={'marginBottom': '15px'})
            ], style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='lag-visualization-graph')
            ], style={'width': '75%', 'paddingLeft': '20px'})
        ]
    
    # Node Cartography content
    elif comparison_value == 'nodecartography':
        return [
            # Left column - filters
            html.Div([
                html.H3('Filters'),
                
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
                        options=[
                            {'label': '10 ms', 'value': 10},
                            {'label': '25 ms', 'value': 25},
                            {'label': '50 ms', 'value': 50}
                        ],
                        value=10
                    )
                ], style={'marginBottom': '15px'}),
                
                # Recording selection (populated based on selected group and DIV)
                html.Div([
                    html.Label('Recording:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='cart-recording-dropdown')
                ], style={'marginBottom': '15px'})
            ], style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Right column - visualization
            html.Div([
                dcc.Graph(id='cart-visualization-graph')
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
    
    try:
        # Get data based on activity and comparison type
        if activity == 'neuronal':
            df = analyzer.get_neuronal_node_data(groups, div_range, metric)
            x_var = 'Group' if comparison == 'nodebygroup' else 'DIV'
            color_var = 'DIV' if comparison == 'nodebygroup' else 'Group'
        else:
            df = analyzer.get_network_node_data(groups, div_range, metric, lag)
            x_var = 'Group' if comparison == 'nodebygroup' else 'DIV'
            color_var = 'DIV' if comparison == 'nodebygroup' else 'Group'
        
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data found with the selected filters'}}
        
        metric_display = metric.replace('_', ' ').title()
        title_suffix = f" (Lag {lag}ms)" if activity == 'network' else ""
        
        # Create the appropriate visualization
        if viz_type == 'violin':
            fig = px.violin(
                df, x=x_var, y='Value', color=color_var,
                box=True, points='all',
                title=f"{metric_display} by {x_var}{title_suffix}",
                labels={'Value': metric_display}
            )
        elif viz_type == 'box':
            fig = px.box(
                df, x=x_var, y='Value', color=color_var,
                points='all',
                title=f"{metric_display} by {x_var}{title_suffix}",
                labels={'Value': metric_display}
            )
        elif viz_type == 'bar':
            # Group data for bar chart
            if x_var == 'DIV':
                grouped = df.groupby(['DIV', 'Group'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='DIV', y='mean', color='Group',
                    error_y='sem',
                    title=f"Mean {metric_display} by {x_var}{title_suffix}",
                    labels={'mean': f"Mean {metric_display}", 'DIV': 'DIV', 'Group': 'Group'}
                )
            else:
                grouped = df.groupby(['Group', 'DIV'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='Group', y='mean', color='DIV', barmode='group',
                    error_y='sem',
                    title=f"Mean {metric_display} by {x_var}{title_suffix}",
                    labels={'mean': f"Mean {metric_display}", 'DIV': 'DIV', 'Group': 'Group'}
                )
        
        # Format DIV axis if applicable
        if x_var == 'DIV':
            divs_in_data = sorted(df['DIV'].unique())
            fig.update_xaxes(
                tickmode='array',
                tickvals=divs_in_data,
                ticktext=[f"DIV {d}" for d in divs_in_data]
            )
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

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
    
    try:
        # Determine if we need recording-level data (neuronal or network)
        if activity == 'neuronal':
            # For neuronal data, first get node data then aggregate to recording level
            node_df = analyzer.get_neuronal_node_data(groups, div_range, metric)
            if not node_df.empty:
                # Aggregate to recording level (mean across electrodes)
                df = node_df.groupby(['Group', 'DIV', 'Recording'])['Value'].agg(['mean', 'std', 'count']).reset_index()
                df.rename(columns={'mean': 'Value'}, inplace=True)
            else:
                df = pd.DataFrame()
        else:
            # For network data, directly get recording-level metrics
            df = analyzer.get_network_recording_data(groups, div_range, metric, lag)
        
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data found with the selected filters'}}
        
        x_var = 'Group' if comparison == 'recordingsbygroup' else 'DIV'
        color_var = 'DIV' if comparison == 'recordingsbygroup' else 'Group'
        metric_display = metric.replace('_', ' ').title()
        title_suffix = f" (Lag {lag}ms)" if activity == 'network' else ""
        
        # Create the appropriate visualization
        if viz_type == 'half_violin':
            fig = px.violin(
                df, x=x_var, y='Value', color=color_var,
                box=True, points='all',
                title=f"{metric_display} by {x_var}{title_suffix}",
                labels={'Value': metric_display}
            )
            
            # Adjust to show only the right half of violins
            for trace in fig.data:
                if isinstance(trace, go.Violin):
                    trace.side = 'positive'  # Show only right half
                    
        elif viz_type == 'box':
            fig = px.box(
                df, x=x_var, y='Value', color=color_var,
                title=f"{metric_display} by {x_var}{title_suffix}",
                labels={'Value': metric_display}
            )
            
        elif viz_type == 'bar':
            # Group data for bar chart
            if x_var == 'DIV':
                grouped = df.groupby(['DIV', 'Group'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='DIV', y='mean', color='Group',
                    error_y='sem',
                    title=f"Mean {metric_display} by {x_var}{title_suffix}",
                    labels={'mean': f"Mean {metric_display}", 'DIV': 'DIV', 'Group': 'Group'}
                )
            else:
                grouped = df.groupby(['Group', 'DIV'])['Value'].agg(['mean', 'sem']).reset_index()
                fig = px.bar(
                    grouped, x='Group', y='mean', color='DIV', barmode='group',
                    error_y='sem',
                    title=f"Mean {metric_display} by {x_var}{title_suffix}",
                    labels={'mean': f"Mean {metric_display}", 'DIV': 'DIV', 'Group': 'Group'}
                )
        
        # Format DIV axis if applicable
        if x_var == 'DIV':
            divs_in_data = sorted(df['DIV'].unique())
            fig.update_xaxes(
                tickmode='array',
                tickvals=divs_in_data,
                ticktext=[f"DIV {d}" for d in divs_in_data]
            )
            
        # Apply common layout updates
        fig.update_layout(
            template='plotly_white',
            xaxis_title=x_var,
            yaxis_title=metric_display
        )
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

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
    
    try:
        # Get data for all lags
        all_data = []
        for lag in [10, 25, 50]:
            df = analyzer.get_network_recording_data(groups, div_range, metric, lag)
            if not df.empty:
                df['Lag'] = lag
                all_data.append(df)
        
        if not all_data:
            return {'data': [], 'layout': {'title': 'No data found with the selected filters'}}
        
        # Combine all lag data
        combined_df = pd.concat(all_data)
        
        if combined_df.empty:
            return {'data': [], 'layout': {'title': 'No data found with the selected filters'}}
        
        metric_display = metric.replace('_', ' ').title()
        
        # Create grouped box plots by lag and group
        fig = px.box(
            combined_df, x='Lag', y='Value', color='Group',
            title=f"{metric_display} by Lag Value",
            labels={'Value': metric_display, 'Lag': 'Lag (ms)'}
        )
        
        # Format Lag axis for better clarity
        fig.update_xaxes(
            tickmode='array',
            tickvals=[10, 25, 50],
            ticktext=['10 ms', '25 ms', '50 ms']
        )
        
        # Apply common layout updates
        fig.update_layout(
            template='plotly_white',
            xaxis_title='Lag (ms)',
            yaxis_title=metric_display,
            boxmode='group'  # Group boxes by color
        )
        
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
    if not group or not div or not lag or not recording:
        return {'data': [], 'layout': {'title': 'Please select all required filters'}}
    
    try:
        # Get node metrics data for P-Z scatter plot
        if (group, recording) not in analyzer.recording_data:
            return {'data': [], 'layout': {'title': f'Data for {recording} not found'}}
        
        node_metrics_key = f'node_metrics_{lag}'
        if node_metrics_key not in analyzer.recording_data[(group, recording)]:
            return {'data': [], 'layout': {'title': f'No node metrics with lag {lag}ms found for {recording}'}}
        
        node_metrics = analyzer.recording_data[(group, recording)][node_metrics_key]
        
        # Find participation coefficient and Z-score fields
        p_field = analyzer.find_field(node_metrics, 'participation')
        z_field = analyzer.find_field(node_metrics, 'z')
        
        if not p_field or not z_field:
            return {'data': [], 'layout': {'title': 'Participation coefficient or Z-score not found'}}
        
        # Extract P and Z values
        p_values = node_metrics[p_field][0, 0].flatten()
        z_values = node_metrics[z_field][0, 0].flatten()
        
        # Create data frame for plotting
        df = pd.DataFrame({
            'P': p_values,
            'Z': z_values
        })
        
        # Remove NaN values
        df = df.dropna()
        
        if df.empty:
            return {'data': [], 'layout': {'title': 'No valid P-Z values found'}}
        
        # Create node cartography scatter plot
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=df['P'],
            y=df['Z'],
            mode='markers',
            marker=dict(
                color=df.index,
                colorscale='viridis',
                size=10,
                line=dict(width=1, color='black')
            ),
            text=[f"Node {i}: P={p:.3f}, Z={z:.3f}" for i, (p, z) in enumerate(zip(df['P'], df['Z']))],
            hoverinfo='text'
        ))
        
        # Add boundary lines
        fig.add_shape(type="line", x0=0, x1=1, y0=2.5, y1=2.5, line=dict(color="gray", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0.3, x1=0.3, y0=-2, y1=7, line=dict(color="gray", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0.75, x1=0.75, y0=-2, y1=7, line=dict(color="gray", width=1, dash="dash"))
        
        # Add region labels
        fig.add_annotation(x=0.15, y=1.25, text="Provincial<br>Nodes", showarrow=False)
        fig.add_annotation(x=0.525, y=1.25, text="Connector<br>Nodes", showarrow=False)
        fig.add_annotation(x=0.875, y=1.25, text="Kinless<br>Nodes", showarrow=False)
        fig.add_annotation(x=0.15, y=4, text="Provincial<br>Hubs", showarrow=False)
        fig.add_annotation(x=0.525, y=4, text="Connector<br>Hubs", showarrow=False)
        fig.add_annotation(x=0.875, y=4, text="Kinless<br>Hubs", showarrow=False)
        
        # Update layout
        fig.update_layout(
            title=f"Node Cartography - {recording} (Lag {lag}ms)",
            xaxis=dict(
                title="Participation Coefficient (P)",
                range=[0, 1]
            ),
            yaxis=dict(
                title="Within-Module Z-Score (Z)",
                range=[-2, 7]
            ),
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)