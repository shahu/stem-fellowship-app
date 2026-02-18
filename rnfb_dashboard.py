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

# Global Models 
LR_MODEL = None
RF_MODEL = None
LR_INFO = ""
RF_INFO = ""

# --- Auto-load models at startup ---
STARTUP_LOG_LINES = []

# WRSI Anomaly average from all_samples_clean_final.csv (used as mock value for prototype)
WRSI_ANOMALY_AVG = 171.29

def _auto_load_models():
    global LR_MODEL, RF_MODEL, LR_INFO, RF_INFO
    STARTUP_LOG_LINES.append(f"=== Dashboard Startup [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
    STARTUP_LOG_LINES.append("Auto-loading models from local directory...\n")

    # Load LR model
    STARTUP_LOG_LINES.append(f"[LR] Searching: {model_load.LR_MODEL_PATH}")
    try:
        lr_model, lr_info = model_load.load_lr_model()
        if lr_model:
            LR_MODEL = lr_model
            LR_INFO = lr_info
            STARTUP_LOG_LINES.append(f"[LR] ✅ Loaded successfully")
            STARTUP_LOG_LINES.append(lr_info)
        else:
            STARTUP_LOG_LINES.append(f"[LR] ❌ Failed: {lr_info}")
    except Exception as e:
        STARTUP_LOG_LINES.append(f"[LR] ❌ Error: {str(e)}")

    STARTUP_LOG_LINES.append("")  # blank line separator

    # Load RF model
    STARTUP_LOG_LINES.append(f"[RF] Searching: {model_load.RF_MODEL_PATH}")
    try:
        rf_model, rf_info = model_load.load_rd_model()
        if rf_model:
            RF_MODEL = rf_model
            RF_INFO = rf_info
            STARTUP_LOG_LINES.append(f"[RF] ✅ Loaded successfully")
            STARTUP_LOG_LINES.append(rf_info)
        else:
            STARTUP_LOG_LINES.append(f"[RF] ❌ Failed: {rf_info}")
    except Exception as e:
        STARTUP_LOG_LINES.append(f"[RF] ❌ Error: {str(e)}")

    STARTUP_LOG_LINES.append("\n=== Auto-load Complete ===")

_auto_load_models()
STARTUP_LOG = "\n".join(STARTUP_LOG_LINES)

# Initialize the Dash app
app = dash.Dash(__name__, external_scripts=[{'src': 'https://cdn.tailwindcss.com'}], suppress_callback_exceptions=True)
app.title = "RNFB Price Predictor"

# Constants & Data 
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

# Icons (SVG)
# Icons (SVG) 
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
    color = "#475569" 
    
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

def icon_warning_light(size=16, className=""):
    """Yellow warning light icon for under-development features."""
    svg_str = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" fill="#FEF3C7" stroke="#D97706"></path><line x1="12" y1="9" x2="12" y2="13" stroke="#D97706"></line><line x1="12" y1="17" x2="12.01" y2="17" stroke="#D97706"></line></svg>'
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    data_uri = f"data:image/svg+xml;base64,{encoded}"
    return html.Img(src=data_uri, className=className, style={'width': size, 'height': size, 'cursor': 'help'})

def rnfb_logo(size=36):
    """Custom RNFB logo — stylized food basket with northern snowflake element."""
    svg_str = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 40 40">
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#4F46E5"/>
      <stop offset="100%" stop-color="#3B82F6"/>
    </linearGradient>
  </defs>
  <rect width="40" height="40" rx="10" fill="url(#lg)"/>
  <g transform="translate(8,7)" stroke="#fff" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="M2 8 L4 22 C4 24 6 25 8 25 L16 25 C18 25 20 24 20 22 L22 8" />
    <line x1="0" y1="8" x2="24" y2="8"/>
    <path d="M7 8 L9 3" />
    <path d="M17 8 L15 3" />
    <line x1="9" y1="3" x2="15" y2="3"/>
    <circle cx="12" cy="1" r="1" fill="#93C5FD" stroke="#93C5FD" stroke-width="1"/>
    <line x1="12" y1="13" x2="12" y2="19" stroke="#93C5FD" stroke-width="1" opacity="0.8"/>
    <line x1="9" y1="16" x2="15" y2="16" stroke="#93C5FD" stroke-width="1" opacity="0.8"/>
    <line x1="9.8" y1="13.8" x2="14.2" y2="18.2" stroke="#93C5FD" stroke-width="1" opacity="0.5"/>
    <line x1="14.2" y1="13.8" x2="9.8" y2="18.2" stroke="#93C5FD" stroke-width="1" opacity="0.5"/>
  </g>
</svg>"""
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    data_uri = f"data:image/svg+xml;base64,{encoded}"
    return html.Img(src=data_uri, style={'width': size, 'height': size})

def icon_target(size=12, className=""):
    return icon_wrapper(
        '<circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle>'
        '<circle cx="12" cy="12" r="2"></circle>',
        size, className
    )


# Components

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
app.layout = html.Div(className="h-screen bg-slate-50 text-slate-800 font-sans flex flex-col overflow-hidden", children=[
    
    # Header
    html.Header(className="bg-white border-b border-slate-200 z-20 px-6 py-3 flex items-center justify-between shadow-sm flex-shrink-0", children=[
        html.Div(className="flex items-center gap-3", children=[
            rnfb_logo(36),
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
            
            # Help Button (re-trigger onboarding tour)
            html.Button(
                id='help-tour-btn',
                children=[html.Span("?", style={'fontSize': '14px', 'fontWeight': '700', 'lineHeight': '1'})],
                className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 hover:bg-indigo-200 transition-colors flex items-center justify-center",
                title="Show User Guide"
            ),
            # Debug Toggle
            html.Button(
                id='toggle-debug-sidebar',
                children=[icon_settings(18, "text-slate-600")],
                className="p-2 rounded-full hover:bg-slate-100 transition-colors",
                title="Toggle Debug Log"
            )
        ])
    ]),

    # Body: Main Content + Debug Sidebar in a flex row
    html.Div(className="flex flex-1 overflow-hidden", children=[

    # Main Content (shrinks when sidebar is open)
    html.Main(id='main-content', className="flex-1 overflow-y-auto transition-all duration-300", children=[
    html.Div(className="max-w-7xl mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6", children=[
        
        # Left Panel (Controls)
        html.Section(className="lg:col-span-4 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col h-full overflow-hidden", children=[
            html.Div(className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center", children=[
                html.H2(className="font-semibold flex items-center gap-2 text-slate-700", children=[icon_settings(), "Model Parameters"]),
                html.Span("Input Panel", className="text-[10px] bg-indigo-50 text-indigo-700 px-2 py-1 rounded font-mono uppercase tracking-wide")
            ]),
            
            html.Div(className="p-3 flex-1 overflow-y-auto space-y-3 custom-scrollbar", children=[
                
                # Date & Crisis — single compact row
                html.Div(className="bg-indigo-50 px-3 py-2 rounded-lg border border-indigo-100", children=[
                    html.Div(className="flex items-center gap-2 flex-wrap", children=[
                        html.Label(className="text-xs font-bold text-indigo-900 uppercase tracking-wide flex items-center gap-1 flex-shrink-0", children=[
                             icon_calendar(12, "text-indigo-600"), "Forecast"
                        ]),
                        dcc.Dropdown(
                            id='month-select',
                            options=[{'label': m['label'][:3] + '.', 'value': m['value']} for m in MONTHS],
                            value='12',
                            clearable=False,
                            className="text-sm text-slate-700",
                            style={'width': '90px'}
                        ),
                        dcc.Dropdown(
                            id='year-select',
                            options=[{'label': str(y), 'value': str(y)} for y in YEARS],
                            value='2023',
                            clearable=False,
                            className="text-sm text-slate-700",
                            style={'width': '80px'}
                        ),
                        html.Div(className="ml-auto flex-shrink-0", title="Crisis Regime (Pandemic / Supply Shock)", children=[
                            dcc.Checklist(
                                id='crisis-mode-toggle',
                                options=[{'label': ' Crisis', 'value': 'crisis'}],
                                value=[],
                                className="text-[10px] font-bold text-slate-600 accent-amber-500",
                                inputClassName="mr-1 cursor-pointer",
                                labelClassName="flex items-center"
                            )
                        ])
                    ])
                ]),
                
                # Core Indicators
                html.Div(className="space-y-2", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_bar_chart(14, "text-indigo-500"), " Core Indicators"
                    ]),
                    html.Div(className="grid grid-cols-2 gap-2", children=[
                        input_field('input-cpi', 'CPI (Index)', 158.3, step=0.1),
                        input_field('input-rate', 'CAD/USD Ex.Rate', 0.82, step=0.001, min=0.60, max=1.10),
                    ])
                ]),
                
                # Logistics
                html.Div(className="space-y-2", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_truck(14, "text-indigo-500"), " Logistics (Fuel)"
                    ]),
                    html.Div(className="space-y-2", children=[
                        slider_field('input-diesel', 'Diesel Price ($/L)', 1.85, 1.0, 3.0, 0.01),
                        slider_field('input-jet', 'Jet Fuel Price ($/L)', 2.10, 1.0, 3.50, 0.01),
                    ])
                ]),

                # Meteorological (Disabled - Prototype Phase)
                html.Div(className="space-y-2 opacity-50", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_thermometer(14, "text-indigo-500"), " Meteorological",
                        html.Span(className="ml-auto", title="🚧 Under development — Meteorological features are not active in this prototype phase.", children=[
                            icon_warning_light(14)
                        ])
                    ]),
                    html.Div(className="grid grid-cols-2 gap-2", children=[
                         html.Div([
                             html.Label('Temp (2m) °C', className="block text-xs font-medium text-slate-400 mb-1"),
                             dcc.Input(id='input-temp', type='number', value=-15, disabled=True,
                                       className="w-full bg-slate-100 border border-slate-200 rounded px-2 py-1.5 text-sm text-slate-400 cursor-not-allowed")
                         ]),
                         html.Div([
                             html.Label('Snowfall (cm)', className="block text-xs font-medium text-slate-400 mb-1"),
                             dcc.Input(id='input-snow', type='number', value=25, min=0, disabled=True,
                                       className="w-full bg-slate-100 border border-slate-200 rounded px-2 py-1.5 text-sm text-slate-400 cursor-not-allowed")
                         ]),
                    ])
                ]),
                
                # Economic
                html.Div(className="space-y-2", children=[
                    html.H3(className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2 pb-1 border-b border-slate-100", children=[
                        icon_trending_up(14, "text-indigo-500"), " CME Futures"
                    ]),
                     html.Div(className="grid grid-cols-2 gap-2", children=[
                         input_field('input-cattle-live', 'Live Cattle', 185.50, step=0.1),
                         input_field('input-cattle-feeder', 'Feeder Cattle', 255.20, step=0.1),
                         input_field('input-wheat', 'Wheat', 580.00, step=0.1),
                         input_field('input-milk', 'Milk III', 17.50, step=0.01),
                    ])
                ])
                
            ]), # End scroll
            
            # Button
            html.Div(className="p-3 border-t border-slate-200 bg-slate-50 flex-shrink-0", children=[
                html.Button([
                    icon_activity(16),
                    html.Span("Run Prediction", className="ml-2")
                ], id='run-btn', className="w-full py-2 rounded-lg text-white text-sm font-semibold shadow-md bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg transition-all flex items-center justify-center")
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
        
        
    ])
    ]), # End Main Content scroll wrapper

    # Debug Sidebar (Inline right panel, default open)
    html.Div(id='debug-sidebar', className="w-80 bg-white border-l border-slate-200 flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden", children=[
        html.Div(className="p-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center", children=[
            html.H3("System Debug Log", className="font-bold text-sm text-slate-800 flex items-center gap-2"),
            html.Button("✕", id="close-sidebar", className="text-slate-400 hover:text-slate-700 font-bold text-lg leading-none")
        ]),
        html.Div(className="p-4 flex-1 overflow-y-auto font-mono text-[11px] text-slate-600 whitespace-pre-wrap leading-relaxed", id='debug-log-content', children=STARTUP_LOG)
    ]),

    ]), # End flex row (main + sidebar)

    # Store for sidebar state
    dcc.Store(id='sidebar-state', data={'open': True}),

    # Stores for log management (avoids duplicate Output on debug-log-content)
    dcc.Store(id='upload-log-store', data=''),
    dcc.Store(id='prediction-log-store', data=''),
    # Hidden div to trigger auto-scroll
    html.Div(id='auto-scroll-trigger', style={'display': 'none'}),

    # --- Onboarding Tour ---
    dcc.Store(id='tour-seen', storage_type='local', data=False),
    dcc.Store(id='tour-step', data=0),
    dcc.Store(id='tour-active', data=False),

    # Tour Overlay
    html.Div(id='tour-overlay', style={'display': 'none'}, children=[
        # Semi-transparent backdrop
        html.Div(id='tour-backdrop', style={
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'backgroundColor': 'rgba(15, 23, 42, 0.7)', 'zIndex': 9998,
            'transition': 'opacity 0.3s ease'
        }),
        # Spotlight cutout (positioned dynamically via JS)
        html.Div(id='tour-spotlight', style={
            'position': 'fixed', 'zIndex': 9999,
            'borderRadius': '12px',
            'boxShadow': '0 0 0 9999px rgba(15, 23, 42, 0.7)',
            'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            'pointerEvents': 'none'
        }),
        # Tooltip card
        html.Div(id='tour-tooltip', style={
            'position': 'fixed', 'zIndex': 10000,
            'backgroundColor': 'white', 'borderRadius': '16px',
            'padding': '24px', 'width': '380px',
            'boxShadow': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
            'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'
        }, children=[
            # Step indicator
            html.Div(id='tour-step-indicator', className="flex items-center gap-1.5 mb-3"),
            # Title
            html.H3(id='tour-title', style={
                'fontSize': '16px', 'fontWeight': '700', 'color': '#1e293b', 'marginBottom': '8px'
            }),
            # Description
            html.P(id='tour-desc', style={
                'fontSize': '13px', 'color': '#64748b', 'lineHeight': '1.6', 'marginBottom': '20px'
            }),
            # Navigation buttons
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                html.Button("Skip", id='tour-skip-btn', style={
                    'background': 'none', 'border': 'none', 'color': '#94a3b8',
                    'fontSize': '13px', 'cursor': 'pointer', 'fontWeight': '500'
                }),
                html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
                    html.Button("← Back", id='tour-prev-btn', style={
                        'padding': '8px 16px', 'borderRadius': '8px', 'border': '1px solid #e2e8f0',
                        'background': 'white', 'color': '#475569', 'fontSize': '13px',
                        'cursor': 'pointer', 'fontWeight': '600'
                    }),
                    html.Button("Next →", id='tour-next-btn', style={
                        'padding': '8px 20px', 'borderRadius': '8px', 'border': 'none',
                        'background': 'linear-gradient(135deg, #4F46E5, #3B82F6)',
                        'color': 'white', 'fontSize': '13px', 'cursor': 'pointer', 'fontWeight': '600'
                    })
                ])
            ])
        ])
    ])
])

# --- Callbacks ---

# --- Upload callback: writes to upload-log-store ---
@app.callback(
    [Output('lr-status-indicator', 'className'),
     Output('rf-status-indicator', 'className'),
     Output('upload-log-store', 'data'),
     Output('debug-sidebar', 'className')],
    [Input('upload-lr-model', 'contents'),
     Input('upload-rf-model', 'contents'),
     Input('close-sidebar', 'n_clicks'),
     Input('toggle-debug-sidebar', 'n_clicks')],
    [State('upload-lr-model', 'filename'),
     State('upload-rf-model', 'filename'),
     State('upload-log-store', 'data'),
     State('debug-sidebar', 'className')]
)
def handle_model_uploads(lr_contents, rf_contents, close_msg, toggle_msg, lr_filename, rf_filename, current_upload_log, current_sidebar_class_state):
    global LR_MODEL, RF_MODEL, LR_INFO, RF_INFO

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Status classes
    success_class = "w-3 h-3 rounded-full bg-green-500 shadow-sm ring-2 ring-green-200"
    error_class = "w-3 h-3 rounded-full bg-red-500 shadow-sm ring-2 ring-red-200"
    neutral_class = "w-3 h-3 rounded-full bg-slate-300"

    lr_status = neutral_class if LR_MODEL is None else success_class
    rf_status = neutral_class if RF_MODEL is None else success_class

    # Sidebar classes (inline panel: show/hide via width)
    sidebar_open_class = "w-80 bg-white border-l border-slate-200 flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden"
    sidebar_closed_class = "w-0 bg-white border-l border-slate-200 flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden"

    new_sidebar_class = current_sidebar_class_state if current_sidebar_class_state else sidebar_open_class
    upload_log = current_upload_log or ''

    # Handle Toggle/Close
    if triggered_id == 'close-sidebar':
        new_sidebar_class = sidebar_closed_class
        return lr_status, rf_status, upload_log, new_sidebar_class

    if triggered_id == 'toggle-debug-sidebar':
        if 'w-80' in new_sidebar_class:
            new_sidebar_class = sidebar_closed_class
        else:
            new_sidebar_class = sidebar_open_class
        return lr_status, rf_status, upload_log, new_sidebar_class

    log_updates = []

    # Handle LR Upload
    if triggered_id == 'upload-lr-model' and lr_contents:
        try:
            content_type, content_string = lr_contents.split(',')
            decoded = base64.b64decode(content_string)
            temp_path = "temp_lr_model.pkl"
            with open(temp_path, "wb") as f:
                f.write(decoded)
            model, info = model_load.load_lr_model(temp_path)
            if model:
                LR_MODEL = model
                LR_INFO = info
                lr_status = success_class
                log_updates.append(f"--- LR Model Loaded [{datetime.now().strftime('%H:%M:%S')}] ---\n{info}")
                new_sidebar_class = sidebar_open_class
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
            temp_path = "temp_rf_model.pkl"
            with open(temp_path, "wb") as f:
                f.write(decoded)
            model, info = model_load.load_rd_model(temp_path)
            if model:
                RF_MODEL = model
                RF_INFO = info
                rf_status = success_class
                log_updates.append(f"--- RF Model Loaded [{datetime.now().strftime('%H:%M:%S')}] ---\n{info}")
                new_sidebar_class = sidebar_open_class
            else:
                rf_status = error_class
                log_updates.append(f"--- RF Model Load Failed ---\n{info}")
        except Exception as e:
            rf_status = error_class
            log_updates.append(f"Error processing RF file: {str(e)}")

    if log_updates:
        new_entry = "\n\n".join(log_updates)
        upload_log = upload_log + ("\n\n" if upload_log else "") + new_entry

    return lr_status, rf_status, upload_log, new_sidebar_class


# --- Combine all logs into debug-log-content ---
@app.callback(
    Output('debug-log-content', 'children'),
    [Input('upload-log-store', 'data'),
     Input('prediction-log-store', 'data')]
)
def combine_debug_logs(upload_log, prediction_log):
    parts = [STARTUP_LOG]
    if upload_log:
        parts.append(upload_log)
    if prediction_log:
        parts.append(prediction_log)
    return "\n\n".join(parts)

# --- Auto-scroll debug log to bottom ---
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var el = document.getElementById('debug-log-content');
            if (el) { el.scrollTop = el.scrollHeight; }
        }, 100);
        return '';
    }
    """,
    Output('auto-scroll-trigger', 'children'),
    Input('debug-log-content', 'children')
)

# --- Onboarding Tour: Master Controller ---
app.clientside_callback(
    """
    function(helpClicks, tourSeen, tourStep, tourActive) {
        var ctx = dash_clientside.callback_context;
        var triggered = ctx.triggered.length > 0 ? ctx.triggered[0].prop_id : '';
        
        // Tour step configuration
        var steps = [
            {
                target: '.lg\\\\:col-span-4',
                title: '① Set Forecast Parameters',
                desc: 'Choose a target month and year for your RNFB price forecast. Fill in the economic indicators below:\\n\\n• CPI (Index) — Consumer Price Index\\n• CAD/USD Exchange Rate\\n• Diesel & Jet Fuel prices (logistics cost)\\n• CME Futures (commodity market data)\\n\\nCheck "Crisis" for pandemic/supply shock scenarios.',
                position: 'right'
            },
            {
                target: '#run-btn',
                title: '② Run Prediction',
                desc: 'Click this button to run the hybrid ML model (Linear Regression + Random Forest). The model combines a base trend from LR with residual corrections from RF to produce an accurate RNFB food basket price prediction.',
                position: 'right'
            },
            {
                target: '#debug-sidebar',
                title: '③ Review Debug Log',
                desc: 'The System Debug Log shows the internal computation details:\\n\\n• LR base prediction with input features\\n• RF residual correction with all 11 features\\n• Final hybrid calculation (LR + RF)\\n\\nLogs accumulate across runs so you can compare results.',
                position: 'left'
            },
            {
                target: '#price-chart',
                title: '④ View Forecast on Chart',
                desc: 'The main chart displays the historical RNFB price trend (2013–2021) along with hybrid model predictions. Your new forecast appears as a red dot at the target date, extending the prediction curve into the future.',
                position: 'top'
            }
        ];
        
        // Helper: position the tooltip and spotlight
        function positionTour(step) {
            var s = steps[step];
            var el = document.querySelector(s.target);
            if (!el) return;
            
            var rect = el.getBoundingClientRect();
            var pad = 8;
            
            // Spotlight
            var spot = document.getElementById('tour-spotlight');
            spot.style.top = (rect.top - pad) + 'px';
            spot.style.left = (rect.left - pad) + 'px';
            spot.style.width = (rect.width + pad * 2) + 'px';
            spot.style.height = (rect.height + pad * 2) + 'px';
            
            // Tooltip
            var tip = document.getElementById('tour-tooltip');
            var tipW = 380, tipH = tip.offsetHeight || 280;
            var tx, ty;
            
            if (s.position === 'right') {
                tx = rect.right + 20;
                ty = rect.top + rect.height / 2 - tipH / 2;
            } else if (s.position === 'left') {
                tx = rect.left - tipW - 20;
                ty = rect.top + rect.height / 2 - tipH / 2;
            } else if (s.position === 'top') {
                tx = rect.left + rect.width / 2 - tipW / 2;
                ty = rect.top - tipH - 20;
            } else {
                tx = rect.left + rect.width / 2 - tipW / 2;
                ty = rect.bottom + 20;
            }
            
            // Clamp to viewport
            tx = Math.max(10, Math.min(tx, window.innerWidth - tipW - 10));
            ty = Math.max(10, Math.min(ty, window.innerHeight - tipH - 10));
            
            tip.style.left = tx + 'px';
            tip.style.top = ty + 'px';
            
            // Update content
            document.getElementById('tour-title').textContent = s.title;
            document.getElementById('tour-desc').textContent = s.desc;
            
            // Step indicators
            var indicator = document.getElementById('tour-step-indicator');
            indicator.innerHTML = '';
            for (var i = 0; i < steps.length; i++) {
                var dot = document.createElement('div');
                dot.style.width = i === step ? '20px' : '8px';
                dot.style.height = '8px';
                dot.style.borderRadius = '4px';
                dot.style.transition = 'all 0.3s';
                dot.style.backgroundColor = i === step ? '#4F46E5' : (i < step ? '#A5B4FC' : '#E2E8F0');
                indicator.appendChild(dot);
            }
            
            // Button states
            document.getElementById('tour-prev-btn').style.display = step === 0 ? 'none' : '';
            var nextBtn = document.getElementById('tour-next-btn');
            nextBtn.textContent = step === steps.length - 1 ? 'Get Started! ✨' : 'Next →';
        }
        
        // Determine action
        var overlay = document.getElementById('tour-overlay');
        
        if (triggered === 'help-tour-btn.n_clicks') {
            // Help button clicked - show tour
            overlay.style.display = 'block';
            setTimeout(function() { positionTour(0); }, 100);
            return [true, 0, true];
        }
        
        if (!tourSeen && !tourActive) {
            // First visit - show tour after short delay
            setTimeout(function() {
                overlay.style.display = 'block';
                positionTour(0);
            }, 800);
            return [window.dash_clientside.no_update, 0, true];
        }
        
        return [window.dash_clientside.no_update, tourStep || 0, tourActive || false];
    }
    """,
    [Output('tour-seen', 'data'),
     Output('tour-step', 'data'),
     Output('tour-active', 'data')],
    [Input('help-tour-btn', 'n_clicks')],
    [State('tour-seen', 'data'),
     State('tour-step', 'data'),
     State('tour-active', 'data')]
)

# --- Tour Navigation (Next/Prev/Skip) ---
app.clientside_callback(
    """
    function(nextClicks, prevClicks, skipClicks, currentStep) {
        var ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
        
        var triggered = ctx.triggered[0].prop_id;
        var totalSteps = 4;
        var overlay = document.getElementById('tour-overlay');
        
        var steps = [
            { target: '.lg\\\\:col-span-4', title: '① Set Forecast Parameters', desc: 'Choose a target month and year for your RNFB price forecast. Fill in the economic indicators below:\\n\\n• CPI (Index) — Consumer Price Index\\n• CAD/USD Exchange Rate\\n• Diesel & Jet Fuel prices (logistics cost)\\n• CME Futures (commodity market data)\\n\\nCheck "Crisis" for pandemic/supply shock scenarios.', position: 'right' },
            { target: '#run-btn', title: '② Run Prediction', desc: 'Click this button to run the hybrid ML model (Linear Regression + Random Forest). The model combines a base trend from LR with residual corrections from RF to produce an accurate RNFB food basket price prediction.', position: 'right' },
            { target: '#debug-sidebar', title: '③ Review Debug Log', desc: 'The System Debug Log shows the internal computation details:\\n\\n• LR base prediction with input features\\n• RF residual correction with all 11 features\\n• Final hybrid calculation (LR + RF)\\n\\nLogs accumulate across runs so you can compare results.', position: 'left' },
            { target: '#price-chart', title: '④ View Forecast on Chart', desc: 'The main chart displays the historical RNFB price trend (2013–2021) along with hybrid model predictions. Your new forecast appears as a red dot at the target date, extending the prediction curve into the future.', position: 'top' }
        ];
        
        function positionTour(step) {
            var s = steps[step];
            var el = document.querySelector(s.target);
            if (!el) return;
            var rect = el.getBoundingClientRect();
            var pad = 8;
            var spot = document.getElementById('tour-spotlight');
            spot.style.top = (rect.top - pad) + 'px';
            spot.style.left = (rect.left - pad) + 'px';
            spot.style.width = (rect.width + pad * 2) + 'px';
            spot.style.height = (rect.height + pad * 2) + 'px';
            var tip = document.getElementById('tour-tooltip');
            var tipW = 380, tipH = tip.offsetHeight || 280;
            var tx, ty;
            if (s.position === 'right') { tx = rect.right + 20; ty = rect.top + rect.height / 2 - tipH / 2; }
            else if (s.position === 'left') { tx = rect.left - tipW - 20; ty = rect.top + rect.height / 2 - tipH / 2; }
            else if (s.position === 'top') { tx = rect.left + rect.width / 2 - tipW / 2; ty = rect.top - tipH - 20; }
            else { tx = rect.left + rect.width / 2 - tipW / 2; ty = rect.bottom + 20; }
            tx = Math.max(10, Math.min(tx, window.innerWidth - tipW - 10));
            ty = Math.max(10, Math.min(ty, window.innerHeight - tipH - 10));
            tip.style.left = tx + 'px';
            tip.style.top = ty + 'px';
            document.getElementById('tour-title').textContent = s.title;
            document.getElementById('tour-desc').textContent = s.desc;
            var indicator = document.getElementById('tour-step-indicator');
            indicator.innerHTML = '';
            for (var i = 0; i < totalSteps; i++) {
                var dot = document.createElement('div');
                dot.style.width = i === step ? '20px' : '8px';
                dot.style.height = '8px';
                dot.style.borderRadius = '4px';
                dot.style.transition = 'all 0.3s';
                dot.style.backgroundColor = i === step ? '#4F46E5' : (i < step ? '#A5B4FC' : '#E2E8F0');
                indicator.appendChild(dot);
            }
            document.getElementById('tour-prev-btn').style.display = step === 0 ? 'none' : '';
            document.getElementById('tour-next-btn').textContent = step === totalSteps - 1 ? 'Get Started! ✨' : 'Next →';
        }
        
        if (triggered === 'tour-skip-btn.n_clicks' || (triggered === 'tour-next-btn.n_clicks' && currentStep >= totalSteps - 1)) {
            // End tour
            overlay.style.display = 'none';
            return [true, 0, false];
        }
        
        var newStep = currentStep || 0;
        if (triggered === 'tour-next-btn.n_clicks') {
            newStep = Math.min(currentStep + 1, totalSteps - 1);
        } else if (triggered === 'tour-prev-btn.n_clicks') {
            newStep = Math.max(currentStep - 1, 0);
        }
        
        setTimeout(function() { positionTour(newStep); }, 50);
        return [window.dash_clientside.no_update, newStep, true];
    }
    """,
    [Output('tour-seen', 'data', allow_duplicate=True),
     Output('tour-step', 'data', allow_duplicate=True),
     Output('tour-active', 'data', allow_duplicate=True)],
    [Input('tour-next-btn', 'n_clicks'),
     Input('tour-prev-btn', 'n_clicks'),
     Input('tour-skip-btn', 'n_clicks')],
    [State('tour-step', 'data')],
    prevent_initial_call=True
)
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
     Output('status-text', 'children'),
     Output('prediction-log-store', 'data')],
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
     State('input-milk', 'value'),
     State('prediction-log-store', 'data')]
)
def update_chart(n_clicks, month, year, crisis_mode_val,
                 cpi, ex_rate, diesel, jet, temp, snow, cattle_l, cattle_f, wheat, milk, existing_pred_log):

    selected_date = f"{year}-{month}"
    is_crisis = 'crisis' in crisis_mode_val if crisis_mode_val else False
    prediction_log_lines = []

    # 1. Base chart data
    data = df_base.copy()

    # --- Real Model Prediction ---
    use_model = (LR_MODEL is not None and RF_MODEL is not None)

    if use_model:
        prediction_log_lines.append(f"━━━ Prediction Run [{datetime.now().strftime('%H:%M:%S')}] ━━━")
        prediction_log_lines.append(f"📅 Target Date: {selected_date}")
        prediction_log_lines.append(f"")

        # --- LR Prediction (base trend) ---
        lr_features = pd.DataFrame({
            'CPI_lag_1m': [float(cpi)],
            'currency_rate': [float(ex_rate)]
        })
        try:
            lr_pred = LR_MODEL.predict(lr_features)[0]
            prediction_log_lines.append(f"┌─ [LR] Linear Regression (Base Trend)")
            prediction_log_lines.append(f"│  Input:")
            prediction_log_lines.append(f"│    CPI_lag_1m     = {cpi}")
            prediction_log_lines.append(f"│    currency_rate  = {ex_rate}")
            prediction_log_lines.append(f"│  ✅ Result: {lr_pred:.4f}")
            prediction_log_lines.append(f"└─────────────────────────")
        except Exception as e:
            lr_pred = 420.0
            prediction_log_lines.append(f"[LR] ❌ Error: {str(e)}")
            prediction_log_lines.append(f"     Using fallback = {lr_pred}")

        prediction_log_lines.append(f"")

        # --- RF Prediction (residual/correction) ---
        rf_features = pd.DataFrame({
            'ZW_lag_12M':         [float(wheat)],
            'Diesel_Price_lag_1M': [float(diesel) * 100],
            'DC_lag_3M':          [float(milk)],
            'GF_lag_6M':          [float(cattle_f)],
            'WRSI_Anomaly':       [WRSI_ANOMALY_AVG],
            'Jet_Price_lag_1M':   [float(jet) * 100],
            'currency_rate':      [float(ex_rate)],
            'ZW_lag_8M':          [float(wheat)],
            'DC_lag_4M':          [float(milk)],
            'CPI_lag_1m':         [float(cpi)]
        })
        try:
            rf_pred = RF_MODEL.predict(rf_features)[0]
            prediction_log_lines.append(f"┌─ [RF] Random Forest (Residual)")
            prediction_log_lines.append(f"│  Input Features:")
            for col in rf_features.columns:
                prediction_log_lines.append(f"│    {col:.<25s} {rf_features[col].values[0]:.4f}")
            prediction_log_lines.append(f"│  ✅ Result: {rf_pred:.4f}")
            prediction_log_lines.append(f"└─────────────────────────")
        except Exception as e:
            rf_pred = 0.0
            prediction_log_lines.append(f"[RF] ❌ Error: {str(e)}")
            prediction_log_lines.append(f"     Using fallback = {rf_pred}")

        # --- Combine: Hybrid = LR + RF ---
        predicted_value = lr_pred + rf_pred
        regime_impact = 25.5 if is_crisis else 0
        predicted_value += regime_impact

        prediction_log_lines.append(f"")
        prediction_log_lines.append(f"┌─ [Hybrid] Final Calculation")
        prediction_log_lines.append(f"│  LR base     = {lr_pred:.2f}")
        prediction_log_lines.append(f"│  RF residual = {rf_pred:+.2f}")
        if is_crisis:
            prediction_log_lines.append(f"│  Crisis adj  = {regime_impact:+.1f}")
        prediction_log_lines.append(f"│  ✅ RNFB Predicted = ${predicted_value:.2f}")
        prediction_log_lines.append(f"└─────────────────────────")
        prediction_log_lines.append(f"━━━ Prediction Complete ━━━")
        status_text = "Crisis Impact Applied" if is_crisis else "Model Prediction"
    else:
        # --- Fallback: mock formula (original logic) ---
        prediction_log_lines.append(f"⚠ Models not loaded — using mock prediction [{datetime.now().strftime('%H:%M:%S')}]")

        existing_point = data[data['name'] == selected_date]
        if not existing_point.empty:
            base_value = existing_point['predicted'].values[0]
        else:
            last_point = data.iloc[-1]
            last_date = datetime.strptime(last_point['name'] + "-01", "%Y-%m-%d")
            sel_date_dt = datetime.strptime(selected_date + "-01", "%Y-%m-%d")
            diff_months = (sel_date_dt.year - last_date.year) * 12 + (sel_date_dt.month - last_date.month)
            trend_per_month = 0.3
            base_value = last_point['predicted'] + (diff_months * trend_per_month)

        core_impact = (cpi - 158.3) * 0.4 + (ex_rate - 1.36) * 15
        logistics_impact = (diesel - 1.85) * 10 + (jet - 2.10) * 15
        commodities_impact = (cattle_l - 185.5) * 0.02 + (wheat - 580) * 0.005
        regime_impact = 25.5 if is_crisis else 0
        total_delta = core_impact + logistics_impact + commodities_impact + regime_impact
        predicted_value = base_value + total_delta
        status_text = "Crisis Impact Applied" if is_crisis else "Mock Projection (no model)"

    # --- Build Chart ---
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

    new_prediction_log = "\n".join(prediction_log_lines)
    # Append to existing log
    if existing_pred_log:
        prediction_log = existing_pred_log + "\n\n" + new_prediction_log
    else:
        prediction_log = new_prediction_log

    return fig, f"${predicted_value:.2f}", f"{selected_date} Forecast", card_class, text_class, status_text, prediction_log


if __name__ == '__main__':
    app.run(debug=False)
