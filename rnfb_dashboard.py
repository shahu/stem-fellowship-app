import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
from datetime import datetime, timedelta
import model_load
import io
import os

# --- Global Models ---
LR_MODEL = None
RF_MODEL = None
LR_INFO = ""
RF_INFO = ""

# Initialize the Dash app
app = dash.Dash(__name__, external_scripts=[{'src': 'https://cdn.tailwindcss.com'}])
app.title = "RNFB Price Predictor"

# --- Constants & Data ---
RAW_CSV_DATA = """Date,Actual RNFB,Hybrid Predicted RNFB
2013-01-01,401.55,395.20
2013-04-01,393.57,399.16
2013-07-01,391.05,394.17
2013-10-01,400.19,391.72
2014-01-01,395.94,399.87
2014-04-01,402.61,397.98
2014-07-01,401.17,400.61
2014-10-01,403.66,402.25
2015-01-01,406.05,400.99
2015-04-01,405.06,405.20
2015-07-01,405.6,406.54
2015-10-01,410.46,406.47
2016-01-01,414.58,409.73
2016-04-01,429.4,411.35
2016-07-01,421.5,425.11
2016-10-01,413.15,420.64
2017-01-01,413.19,420.68
2017-04-01,403.33,415.76
2017-07-01,404.38,406.12
2017-10-01,415.63,410.14
2018-01-01,422.12,416.03
2018-04-01,423.91,420.78
2018-07-01,422.3,421.83
2018-10-01,428.12,424.13
2019-01-01,422.07,425.35
2019-04-01,422.04,426.93
2019-07-01,421.69,423.34
2019-10-01,418.72,429.44
2020-01-01,416.65,428.80
2020-04-01,434.97,425.11
2020-07-01,425.02,444.21
2020-10-01,420.18,426.87
2021-01-01,419.99,426.12
2021-04-01,420.05,424.18
2021-07-01,424.83,424.51
2021-10-01,434.03,426.05"""

def get_base_data():
    lines = RAW_CSV_DATA.split('\n')[1:]
    data = []
    for line in lines:
        parts = line.split(',')
        date_str = parts[0]
        actual = float(parts[1]) if parts[1] else None
        predicted = float(parts[2]) if parts[2] else None
        
        data.append({
            'name': date_str[:7],
            'actual': actual,
            'predicted': predicted,
            'type': 'historical'
        })
    
    # Extrapolation logic
    last_real = data[-1]
    current_date = datetime.strptime(last_real['name'] + "-01", "%Y-%m-%d")
    last_pred_val = last_real['predicted']
    
    target_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    
    # Simple consistent randomization for demo
    np.random.seed(42) 
    
    while current_date < target_date:
        # Add 3 months
        # Logic to advance quarters roughly
        month = current_date.month
        year = current_date.year
        new_month = month + 3
        if new_month > 12:
            new_month -= 12
            year += 1
        current_date = datetime(year, new_month, 1)
        
        date_str = current_date.strftime("%Y-%m")
        
        # Trend simulation
        seasonality = math.sin((new_month - 1) / 6 * math.pi) * 2
        last_pred_val += 0.8 + (np.random.random() - 0.3) + seasonality
        
        data.append({
            'name': date_str,
            'actual': None,
            'predicted': round(last_pred_val, 2),
            'type': 'projected'
        })
        
    return pd.DataFrame(data)

df_base = get_base_data()

MONTHS = [
    {'value': '01', 'label': 'January'}, {'value': '02', 'label': 'February'},
    {'value': '03', 'label': 'March'}, {'value': '04', 'label': 'April'},
    {'value': '05', 'label': 'May'}, {'value': '06', 'label': 'June'},
    {'value': '07', 'label': 'July'}, {'value': '08', 'label': 'August'},
    {'value': '09', 'label': 'September'}, {'value': '10', 'label': 'October'},
    {'value': '11', 'label': 'November'}, {'value': '12', 'label': 'December'}
]
YEARS = [2022, 2023, 2024, 2025, 2026, 2027]

# --- Icons (SVG) ---
# --- Icons (SVG) ---
import base64

# Tailwind Color Map for SVG Fill/Stroke substitution
TAILWIND_COLORS = {
    'text-indigo-500': '#6366f1',
    'text-indigo-600': '#4f46e5',
    'text-indigo-100': '#e0e7ff',
    'text-amber-500': '#f59e0b',
    'text-amber-100': '#fef3c7',
    'text-slate-400': '#94a3b8',
    'white': '#ffffff',
    'current': 'currentColor' # fallback
}

def icon_wrapper(content, size=16, className=""):
    """
    Wrapper to create standard SVG icons using Base64 Data URIs.
    This ensures they render correctly in any Dash environment.
    """
    
    # 1. Determine Color
    # Simple extraction of known color classes
    color = "#475569" # slate-600 default
    
    # Check for specific text colors in className
    parts = className.split(' ')
    for part in parts:
        if part in TAILWIND_COLORS:
            color = TAILWIND_COLORS[part]
            break
        # Fallback for dynamic colors not in map, though difficult to parse
        # If we see 'text-white' manually handle etc
        if part == 'text-white':
            color = '#ffffff'
    
    # 2. construct SVG string
    # Replace single quotes with double for attribute stability
    svg_str = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{content}</svg>'
    
    # 3. Base64 encode
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    data_uri = f"data:image/svg+xml;base64,{encoded}"
    
    # 4. Return Image
    return html.Img(src=data_uri, className=className, style={'width': size, 'height': size})

def icon_calculator(size=16, className=""):
    return icon_wrapper(
        '<rect x="4" y="2" width="16" height="20" rx="2"></rect>'
        '<line x1="8" y1="6" x2="16" y2="6"></line>'
        '<line x1="16" y1="14" x2="16" y2="18"></line>'
        '<path d="M16 10h.01"></path><path d="M12 10h.01"></path><path d="M8 10h.01"></path>'
        '<path d="M12 14h.01"></path><path d="M8 14h.01"></path>'
        '<path d="M12 18h.01"></path><path d="M8 18h.01"></path>', 
        size, className
    )

def icon_activity(size=16, className=""):
    return icon_wrapper('<path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>', size, className)

def icon_settings(size=18, className=""):
    return icon_wrapper(
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>'
        '<circle cx="12" cy="12" r="3"></circle>',
        size, className
    )

def icon_calendar(size=14, className=""):
    return icon_wrapper(
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>'
        '<line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line>'
        '<line x1="3" y1="10" x2="21" y2="10"></line>',
        size, className
    )

def icon_alert_triangle(size=16, className=""):
    return icon_wrapper(
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>'
        '<line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line>',
        size, className
    )

def icon_bar_chart(size=14, className=""):
    return icon_wrapper(
        '<line x1="18" y1="20" x2="18" y2="10"></line>'
        '<line x1="12" y1="20" x2="12" y2="4"></line>'
        '<line x1="6" y1="20" x2="6" y2="14"></line>',
        size, className
    )

def icon_truck(size=14, className=""):
    return icon_wrapper(
        '<rect x="1" y="3" width="15" height="13" rx="2"></rect>'
        '<polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon>'
        '<circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle>',
        size, className
    )

def icon_thermometer(size=14, className=""):
    return icon_wrapper(
        '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path>',
        size, className
    )

def icon_trending_up(size=14, className=""):
    return icon_wrapper(
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>'
        '<polyline points="17 6 23 6 23 12"></polyline>',
        size, className
    )

def icon_target(size=12, className=""):
    return icon_wrapper(
        '<circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle>'
        '<circle cx="12" cy="12" r="2"></circle>',
        size, className
    )


# --- Components ---

def input_field(id_name, label, value, type="number", step=None, min=None, max=None):
    return html.Div([
        html.Label(label, className="block text-xs font-medium text-slate-600 mb-1"),
        dcc.Input(
            id=id_name,
            type=type,
            value=value,
            step=step,
            min=min,
            max=max,
            className="w-full bg-slate-50 border border-slate-200 rounded px-2 py-1.5 text-sm focus:ring-1 focus:ring-indigo-500 outline-none"
        )
    ])

def slider_field(id_name, label, value, min, max, step):
    return html.Div([
        html.Div([
            html.Label(label, className="text-xs font-medium text-slate-600"),
            html.Span(id=f"{id_name}-val", children=str(value), className="text-xs font-mono text-slate-500")
        ], className="flex justify-between mb-1"),
        dcc.Slider(
            id=id_name,
            min=min,
            max=max,
            step=step,
            value=value,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            className="w-full"
        )
    ])

# --- Layout ---
app.layout = html.Div(className="min-h-screen bg-slate-50 text-slate-800 font-sans", children=[
    
    # Header
    html.Header(className="bg-white border-b border-slate-200 sticky top-0 z-20 px-6 py-4 flex items-center justify-between shadow-sm", children=[
        html.Div(className="flex items-center gap-3", children=[
            html.Div(className="bg-indigo-600 p-2 rounded-lg text-white", children=[icon_calculator()]),
            html.Div([
                html.H1("RNFB Price Predictor", className="text-xl font-bold text-slate-900"),
                html.P("Revised Northern Food Basket System", className="text-xs text-slate-500")
            ])
        ]),
        
        # Model Data Loading Section
        html.Div(className="flex items-center gap-4", children=[
            # LR Upload
            html.Div(className="flex items-center gap-2", children=[
                dcc.Upload(
                    id='upload-lr-model',
                    children=html.Button([
                        html.Span("Load LR Model", className="text-xs font-semibold")
                    ], className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded transition-colors text-xs"),
                    multiple=False
                ),
                html.Div(id='lr-status-indicator', className="w-3 h-3 rounded-full bg-slate-300", title="LR Model Status")
            ]),
            
            # RF Upload
            html.Div(className="flex items-center gap-2", children=[
                dcc.Upload(
                    id='upload-rf-model',
                    children=html.Button([
                        html.Span("Load RF Model", className="text-xs font-semibold")
                    ], className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded transition-colors text-xs"),
                    multiple=False
                ),
                html.Div(id='rf-status-indicator', className="w-3 h-3 rounded-full bg-slate-300", title="RF Model Status")
            ]),

            html.Div(className="hidden md:flex items-center gap-4 text-sm text-slate-600 bg-slate-100 px-4 py-2 rounded-full", children=[
                html.Span(className="flex items-center gap-1", children=[
                    icon_activity(16, "text-indigo-500"), "Model: Linear Regression + RF Regressor"
                ])
            ]),
            
            # Debug Toggle
            html.Button(
                id='toggle-debug-sidebar',
                children=[icon_settings(18, "text-slate-600")],
                className="p-2 rounded-full hover:bg-slate-100 transition-colors",
                title="Toggle Debug Log"
            )
        ])
    ]),

    # Main Content
    html.Main(className="max-w-7xl mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6", children=[
        
        # Left Panel (Controls)
        html.Section(className="lg:col-span-4 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col h-full overflow-hidden", children=[
            html.Div(className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center", children=[
                html.H2(className="font-semibold flex items-center gap-2 text-slate-700", children=[icon_settings(), "Model Parameters"]),
                html.Span("Input Panel", className="text-[10px] bg-indigo-50 text-indigo-700 px-2 py-1 rounded font-mono uppercase tracking-wide")
            ]),
            
            html.Div(className="p-5 flex-1 overflow-y-auto space-y-6 custom-scrollbar", children=[
                
                # Date & Crisis
                html.Div(className="bg-indigo-50 p-4 rounded-lg border border-indigo-100", children=[
                    html.Label(className="block text-xs font-bold text-indigo-900 uppercase tracking-wide mb-2 flex items-center gap-2", children=[
                         icon_calendar(14, "text-indigo-600"), " Forecast Target Date"
                    ]),
                    
                    html.Div(className="flex gap-2 mb-3", children=[
                        dcc.Dropdown(
                            id='month-select',
                            options=[{'label': m['label'], 'value': m['value']} for m in MONTHS],
                            value='12',
                            clearable=False,
                            className="flex-1 text-sm text-slate-700 w-32" 
                        ),
                        dcc.Dropdown(
                            id='year-select',
                            options=[{'label': str(y), 'value': str(y)} for y in YEARS],
                            value='2023',
                            clearable=False,
                             className="w-24 text-sm text-slate-700"
                        )
                    ]),
                    
                    # Crisis Toggle
                    dcc.Checklist(
                        id='crisis-mode-toggle',
                        options=[{'label': ' Crisis Regime (Pandemic / Supply Shock)', 'value': 'crisis'}],
                        value=[],
                        className="text-xs font-bold text-slate-600 accent-amber-500",
                        inputClassName="mr-2 cursor-pointer",
                        labelClassName="flex items-center"
                    )
                ]),
                
                # Core Indicators
                html.Div(className="space-y-3", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_bar_chart(14, "text-indigo-500"), " Core Indicators"
                    ]),
                    html.Div(className="grid grid-cols-2 gap-3", children=[
                        input_field('input-cpi', 'CPI (Index)', 158.3, step=0.1),
                        input_field('input-rate', 'CAD/USD Ex.Rate', 1.36, step=0.01),
                    ])
                ]),
                
                # Logistics
                html.Div(className="space-y-3", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_truck(14, "text-indigo-500"), " Logistics (Fuel)"
                    ]),
                    html.Div(className="space-y-3", children=[
                        slider_field('input-diesel', 'Diesel Price ($/L)', 1.85, 1.0, 3.0, 0.01),
                        slider_field('input-jet', 'Jet Fuel Price ($/L)', 2.10, 1.0, 3.50, 0.01),
                    ])
                ]),

                # Meteorological
                html.Div(className="space-y-3", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_thermometer(14, "text-indigo-500"), " Meteorological"
                    ]),
                    html.Div(className="grid grid-cols-2 gap-3", children=[
                         input_field('input-temp', 'Temp (2m) °C', -15),
                         input_field('input-snow', 'Snowfall (cm)', 25, min=0),
                    ])
                ]),
                
                # Economic
                html.Div(className="space-y-3", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_trending_up(14, "text-indigo-500"), " CME Futures"
                    ]),
                     html.Div(className="grid grid-cols-2 gap-3", children=[
                         input_field('input-cattle-live', 'Live Cattle', 185.50, step=0.1),
                         input_field('input-cattle-feeder', 'Feeder Cattle', 255.20, step=0.1),
                         input_field('input-wheat', 'Wheat', 580.00, step=0.1),
                         input_field('input-milk', 'Milk III', 17.50, step=0.01),
                    ])
                ])
                
            ]), # End scroll
            
            # Button
            html.Div(className="p-4 border-t border-slate-200 bg-slate-50", children=[
                html.Button([
                    icon_activity(16),
                    html.Span("Run Prediction", className="ml-2")
                ], id='run-btn', className="w-full py-2.5 rounded-lg text-white text-sm font-semibold shadow-md bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg transition-all flex items-center justify-center")
            ])
            
        ]),
        
        # Right Panel (Results)
        html.Section(className="lg:col-span-8 flex flex-col gap-6", children=[
            
            # Metrics
            html.Div(className="grid grid-cols-1 md:grid-cols-4 gap-4", children=[
                html.Div(id='metric-card-main', className="md:col-span-1 p-4 rounded-xl shadow-md text-white flex flex-col justify-between min-h-[100px] transition-colors duration-300 bg-gradient-to-br from-indigo-600 to-blue-700", children=[
                    html.Div([
                        html.P(id='target-date-label', children="2023-12 Forecast", className="text-indigo-100 text-[10px] font-bold uppercase tracking-wider mb-1"),
                        html.Span(id='prediction-value', children="$0", className="text-2xl font-bold")
                    ]),
                    html.Div(className="flex items-center gap-1 mt-2 text-indigo-100 text-[10px]", children=[
                         icon_trending_up(12), html.Span(id='status-text', children="Dynamic Projection")
                    ])
                ]),
                
                # RMSE
                html.Div(className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-center", children=[
                     html.P(className="text-slate-500 text-[10px] font-bold uppercase tracking-wide mb-1 flex items-center gap-1", children=[icon_activity(12), " RMSE"]),
                     html.Div(className="flex items-end gap-2", children=[html.Span("5.46", className="text-xl font-bold text-slate-800")])
                ]),
                # MAE
                html.Div(className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-center", children=[
                     html.P(className="text-slate-500 text-[10px] font-bold uppercase tracking-wide mb-1 flex items-center gap-1", children=[icon_target(12), " MAE"]),
                     html.Div(className="flex items-end gap-2", children=[html.Span("4.18", className="text-xl font-bold text-slate-800")])
                ]),
                # R2
                html.Div(className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-center", children=[
                     html.P(className="text-slate-500 text-[10px] font-bold uppercase tracking-wide mb-1 flex items-center gap-1", children=[icon_bar_chart(12), " R² Score"]),
                     html.Div(className="flex items-end gap-2", children=[html.Span("0.76", className="text-xl font-bold text-slate-800")])
                ])
            ]),
            
            # Chart
            html.Div(className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex-1 min-h-[400px] flex flex-col", children=[
                html.Div(className="flex justify-between items-center mb-6", children=[
                    html.Div([
                        html.H3("RNFB Price Trend", className="text-lg font-bold text-slate-800"),
                        html.P("Actual (2013-2021) vs. Hybrid Predicted (Fitted to 2023)", className="text-sm text-slate-500")
                    ])
                ]),
                
                dcc.Loading(type="default", children=[
                    dcc.Graph(id='price-chart', style={'height': '100%'}, config={'displayModeBar': False})
                ])
            ])
        ])
        
        
    ]),

    # Debug Sidebar (Fixed Right)
    html.Div(id='debug-sidebar', className="fixed right-0 top-0 h-full w-80 bg-white shadow-2xl transform translate-x-full transition-transform duration-300 z-50 flex flex-col border-l border-slate-200", children=[
        html.Div(className="p-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center", children=[
            html.H3("System Debug Log", className="font-bold text-slate-800 flex items-center gap-2"),
            html.Button("x", id="close-sidebar", className="text-slate-500 hover:text-slate-700 font-bold")
        ]),
        html.Div(className="p-4 flex-1 overflow-y-auto font-mono text-xs text-slate-600 whitespace-pre-wrap", id='debug-log-content', children="No models loaded.")
    ]),

    # Store for sidebar state
    dcc.Store(id='sidebar-state', data={'open': False})
])

# --- Callbacks ---

@app.callback(
    [Output('lr-status-indicator', 'className'),
     Output('rf-status-indicator', 'className'),
     Output('debug-log-content', 'children'),
     Output('debug-sidebar', 'className')], # Controls visibility via translate class
    [Input('upload-lr-model', 'contents'),
     Input('upload-rf-model', 'contents'),
     Input('close-sidebar', 'n_clicks'),
     Input('toggle-debug-sidebar', 'n_clicks')],
    [State('upload-lr-model', 'filename'),
     State('upload-rf-model', 'filename'),
     State('debug-log-content', 'children'),
     State('debug-sidebar', 'className')] 
)
def handle_model_uploads(lr_contents, rf_contents, close_msg, toggle_msg, lr_filename, rf_filename, current_log, current_sidebar_class_state):
    global LR_MODEL, RF_MODEL, LR_INFO, RF_INFO
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Status classes
    success_class = "w-3 h-3 rounded-full bg-green-500 shadow-sm ring-2 ring-green-200"
    error_class = "w-3 h-3 rounded-full bg-red-500 shadow-sm ring-2 ring-red-200"
    neutral_class = "w-3 h-3 rounded-full bg-slate-300"
    
    lr_status = neutral_class if LR_MODEL is None else success_class
    rf_status = neutral_class if RF_MODEL is None else success_class
    
    # Sidebar classes
    sidebar_base_class = "fixed right-0 top-0 h-full w-80 bg-white shadow-2xl transform transition-transform duration-300 z-50 flex flex-col border-l border-slate-200 "
    sidebar_open_class = sidebar_base_class + "translate-x-0"
    sidebar_closed_class = sidebar_base_class + "translate-x-full"
    
    new_sidebar_class = current_sidebar_class_state if current_sidebar_class_state else sidebar_closed_class
    
    # Handle Toggle/Close
    if triggered_id == 'close-sidebar':
        new_sidebar_class = sidebar_closed_class
        # No log update
        return lr_status, rf_status, current_log, new_sidebar_class
        
    if triggered_id == 'toggle-debug-sidebar':
        # Toggle based on current state check (checking string presence is simple enough)
        if "translate-x-0" in new_sidebar_class:
            new_sidebar_class = sidebar_closed_class
        else:
            new_sidebar_class = sidebar_open_class
        return lr_status, rf_status, current_log, new_sidebar_class

    log_updates = []
    
    # Handle LR Upload
    if triggered_id == 'upload-lr-model' and lr_contents:
        try:
            content_type, content_string = lr_contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save to temp file
            temp_path = "temp_lr_model.pkl"
            with open(temp_path, "wb") as f:
                f.write(decoded)
            
            model, info = model_load.load_lr_model(temp_path)
            
            if model:
                LR_MODEL = model
                LR_INFO = info
                lr_status = success_class
                log_updates.append(f"--- LR Model Loaded [{datetime.now().strftime('%H:%M:%S')}] ---\n{info}")
                new_sidebar_class = sidebar_open_class # Open sidebar on success
            else:
                lr_status = error_class
                log_updates.append(f"--- LR Model Load Failed ---\n{info}")
                
        except Exception as e:
            lr_status = error_class
            log_updates.append(f"Error processing LR file: {str(e)}")
            
    # Handle RF Upload
    if triggered_id == 'upload-rf-model' and rf_contents:
        try:
            content_type, content_string = rf_contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save to temp file
            temp_path = "temp_rf_model.pkl"
            with open(temp_path, "wb") as f:
                f.write(decoded)
            
            model, info = model_load.load_rd_model(temp_path)
            
            if model:
                RF_MODEL = model
                RF_INFO = info
                rf_status = success_class
                log_updates.append(f"--- RF Model Loaded [{datetime.now().strftime('%H:%M:%S')}] ---\n{info}")
                new_sidebar_class = sidebar_open_class # Open sidebar on success
            else:
                rf_status = error_class
                log_updates.append(f"--- RF Model Load Failed ---\n{info}")

        except Exception as e:
            rf_status = error_class
            log_updates.append(f"Error processing RF file: {str(e)}")

    # Combine logs (Append mode)
    if log_updates:
        # If current_log is the placeholder, replace it. Otherwise append.
        base_log = "" if current_log == "No models loaded." else current_log
        new_entry = "\n\n".join(log_updates)
        new_log = base_log + ("\n\n" if base_log else "") + new_entry
    else:
        new_log = current_log
        if triggered_id not in ['upload-lr-model', 'upload-rf-model']:
             return lr_status, rf_status, new_log, new_sidebar_class

    return lr_status, rf_status, new_log, new_sidebar_class
@app.callback(
    Output('input-diesel-val', 'children'),
    Input('input-diesel', 'value')
)
def update_diesel_label(val):
    return f"{val}"

@app.callback(
    Output('input-jet-val', 'children'),
    Input('input-jet', 'value')
)
def update_jet_label(val):
    return f"{val}"

@app.callback(
    [Output('price-chart', 'figure'),
     Output('prediction-value', 'children'),
     Output('target-date-label', 'children'),
     Output('metric-card-main', 'className'),
     Output('target-date-label', 'className'),
     Output('status-text', 'children')],
    [Input('run-btn', 'n_clicks'),
     Input('month-select', 'value'),
     Input('year-select', 'value'),
     Input('crisis-mode-toggle', 'value')],
    [State('input-cpi', 'value'),
     State('input-rate', 'value'),
     State('input-diesel', 'value'),
     State('input-jet', 'value'),
     State('input-temp', 'value'),
     State('input-snow', 'value'),
     State('input-cattle-live', 'value'),
     State('input-cattle-feeder', 'value'),
     State('input-wheat', 'value'),
     State('input-milk', 'value')]
)
def update_chart(n_clicks, month, year, crisis_mode_val, 
                 cpi, ex_rate, diesel, jet, temp, snow, cattle_l, cattle_f, wheat, milk):
    
    selected_date = f"{year}-{month}"
    is_crisis = 'crisis' in crisis_mode_val if crisis_mode_val else False
    
    # 1. Base Logic from TS
    data = df_base.copy()
    
    # Logic to find base value
    existing_point = data[data['name'] == selected_date]
    
    if not existing_point.empty:
        base_value = existing_point['predicted'].values[0]
    else:
        # Simple linear projection if not found (extrapolation for dates beyond generated)
        last_point = data.iloc[-1]
        last_date = datetime.strptime(last_point['name'] + "-01", "%Y-%m-%d")
        sel_date_dt = datetime.strptime(selected_date + "-01", "%Y-%m-%d")
        
        diff_months = (sel_date_dt.year - last_date.year) * 12 + (sel_date_dt.month - last_date.month)
        trend_per_month = 0.3
        base_value = last_point['predicted'] + (diff_months * trend_per_month)

    # 2. Delta Calculation
    core_impact = (cpi - 158.3) * 0.4 + (ex_rate - 1.36) * 15
    logistics_impact = (diesel - 1.85) * 10 + (jet - 2.10) * 15
    
    weather_impact = 0
    if temp < -20:
        weather_impact += 5
    weather_impact += (snow - 25) * 0.1
    
    commodities_impact = (cattle_l - 185.5) * 0.02 + (wheat - 580) * 0.005
    
    regime_impact = 25.5 if is_crisis else 0
    
    total_delta = core_impact + logistics_impact + weather_impact + commodities_impact + regime_impact
    predicted_value = base_value + total_delta
    
    # 3. Update Chart Data
    # If selected date exists, update it? Or append? Matches TS logic: "newPredictionPoint"
    
    # We want to show the full line, and this specific point as the "endpoint" or highlighted point
    # To match TS visual:
    # "if selectedDate > lastBaseDate ... push ... else ... newChartData = [...baseData, newPredictionPoint]" 
    # Actually in TS it replaces or appends for the chart display only. 
    
    # Let's just append this point to the dataframe for plotting purposes if it's new, 
    # or just create a special trace for it.
    
    # Filter to not show future beyond selected date? TS didn't seem to truncate, just added the point.
    
    fig = go.Figure()
    
    # Actual Trace
    fig.add_trace(go.Scatter(
        x=data[data['actual'].notnull()]['name'],
        y=data[data['actual'].notnull()]['actual'],
        mode='lines',
        name='Actual RNFB',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Predicted Trace (dashed)
    fig.add_trace(go.Scatter(
        x=data['name'],
        y=data['predicted'],
        mode='lines',
        name='Hybrid Predicted',
        line=dict(color='#ef4444' if not is_crisis else '#d97706', width=2, dash='dash')
    ))
    
    # Forecast Point
    fig.add_trace(go.Scatter(
        x=[selected_date],
        y=[predicted_value],
        mode='markers+text',
        name='Forecast',
        marker=dict(color='#ef4444' if not is_crisis else '#d97706', size=12, line=dict(color='white', width=2)),
        text=[selected_date],
        textposition="top center"
    ))
    
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=20, r=20, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    
    # Styles based on crisis mode
    card_class = "md:col-span-1 p-4 rounded-xl shadow-md text-white flex flex-col justify-between min-h-[100px] transition-colors duration-300 "
    card_class += "bg-gradient-to-br from-amber-600 to-orange-700" if is_crisis else "bg-gradient-to-br from-indigo-600 to-blue-700"
    
    text_class = "text-[10px] font-bold uppercase tracking-wider mb-1 "
    text_class += "text-amber-100" if is_crisis else "text-indigo-100"
    
    status_text = "Crisis Impact Applied" if is_crisis else "Dynamic Projection"
    
    return fig, f"${predicted_value:.2f}", f"{selected_date} Forecast", card_class, text_class, status_text


if __name__ == '__main__':
    app.run(debug=True)
