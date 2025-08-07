"""Dash dashboard for content performance prediction."""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Content Performance Predictor", className="text-center mb-4"),
            html.P("AI-powered social media content performance analysis and prediction", 
                   className="text-center text-muted")
        ])
    ]),
    
    # Navigation tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Performance Prediction", tab_id="prediction"),
                dbc.Tab(label="Caption Analyzer", tab_id="caption"),
                dbc.Tab(label="Best Posting Times", tab_id="times"),
                dbc.Tab(label="Platform Trends", tab_id="trends"),
                dbc.Tab(label="Cross-Platform Analysis", tab_id="cross-platform")
            ], id="tabs", active_tab="prediction")
        ])
    ], className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content")
    
], fluid=True)

# Prediction tab content
prediction_layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Content Performance Prediction"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Platform"),
                        dcc.Dropdown(
                            id="platform-dropdown",
                            options=[
                                {"label": "LinkedIn", "value": "linkedin"},
                                {"label": "Instagram", "value": "instagram"},
                                {"label": "Twitter", "value": "twitter"},
                                {"label": "Substack", "value": "substack"},
                                {"label": "Threads", "value": "threads"}
                            ],
                            value="linkedin"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Content Type"),
                        dcc.Dropdown(
                            id="content-type-dropdown",
                            options=[
                                {"label": "No Video", "value": "no_video"},
                                {"label": "Video", "value": "video"}
                            ],
                            value="no_video"
                        )
                    ], width=6)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Date"),
                        dcc.DatePickerSingle(
                            id="date-picker",
                            date=date.today(),
                            display_format="YYYY-MM-DD"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Button("Predict", id="predict-btn", color="primary", className="mt-4")
                    ], width=6)
                ]),
                html.Div(id="prediction-result")
            ])
        ])
    ], width=12)
])

# Caption analyzer tab content
caption_layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Caption Analyzer"),
            dbc.CardBody([
                dbc.Textarea(
                    id="caption-input",
                    placeholder="Enter your caption here...",
                    rows=4,
                    className="mb-3"
                ),
                dbc.Button("Analyze", id="analyze-btn", color="primary"),
                html.Div(id="caption-analysis-result")
            ])
        ])
    ], width=12)
])

# Best posting times tab content
times_layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Best Posting Times"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Platform"),
                        dcc.Dropdown(
                            id="times-platform-dropdown",
                            options=[
                                {"label": "LinkedIn", "value": "linkedin"},
                                {"label": "Instagram", "value": "instagram"},
                                {"label": "Twitter", "value": "twitter"},
                                {"label": "Substack", "value": "substack"},
                                {"label": "Threads", "value": "threads"}
                            ],
                            value="linkedin"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Content Type"),
                        dcc.Dropdown(
                            id="times-content-type-dropdown",
                            options=[
                                {"label": "All", "value": None},
                                {"label": "No Video", "value": "no_video"},
                                {"label": "Video", "value": "video"}
                            ],
                            value=None
                        )
                    ], width=6)
                ], className="mb-3"),
                dbc.Button("Analyze", id="times-analyze-btn", color="primary"),
                html.Div(id="times-analysis-result")
            ])
        ])
    ], width=12)
])

# Platform trends tab content
trends_layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Platform Trends"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Platform"),
                        dcc.Dropdown(
                            id="trends-platform-dropdown",
                            options=[
                                {"label": "LinkedIn", "value": "linkedin"},
                                {"label": "Instagram", "value": "instagram"},
                                {"label": "Twitter", "value": "twitter"},
                                {"label": "Substack", "value": "substack"},
                                {"label": "Threads", "value": "threads"}
                            ],
                            value="linkedin"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Days Back"),
                        dcc.Slider(
                            id="trends-days-slider",
                            min=7,
                            max=90,
                            step=7,
                            value=30,
                            marks={i: str(i) for i in range(7, 91, 7)}
                        )
                    ], width=6)
                ], className="mb-3"),
                dbc.Button("Analyze", id="trends-analyze-btn", color="primary"),
                html.Div(id="trends-analysis-result")
            ])
        ])
    ], width=12)
])

# Cross-platform analysis tab content
cross_platform_layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Cross-Platform Analysis"),
            dbc.CardBody([
                dbc.Button("Load Analysis", id="cross-platform-btn", color="primary"),
                html.Div(id="cross-platform-result")
            ])
        ])
    ], width=12)
])

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render tab content based on active tab."""
    if active_tab == "prediction":
        return prediction_layout
    elif active_tab == "caption":
        return caption_layout
    elif active_tab == "times":
        return times_layout
    elif active_tab == "trends":
        return trends_layout
    elif active_tab == "cross-platform":
        return cross_platform_layout
    else:
        return "Select a tab"

@app.callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    Input("platform-dropdown", "value"),
    Input("content-type-dropdown", "value"),
    Input("date-picker", "date"),
    prevent_initial_call=True
)
def predict_engagement(n_clicks, platform, content_type, selected_date):
    """Predict content engagement."""
    if not n_clicks:
        return ""
    
    try:
        # Prepare request data
        request_data = {
            "platform": platform,
            "content_type": content_type,
            "date": selected_date
        }
        
        # Make API request
        api_url = os.getenv("API_URL", "http://localhost:8000")
        response = requests.post(f"{api_url}/predict", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Create result card
            return dbc.Card([
                dbc.CardBody([
                    html.H4("Prediction Result", className="card-title"),
                    html.P(f"Platform: {result['platform'].title()}"),
                    html.P(f"Content Type: {result['content_type'].replace('_', ' ').title()}"),
                    html.P(f"Date: {result['date']}"),
                    html.H3(f"Predicted Engagement: {result['predicted_engagement']:.0f}", 
                           className="text-primary"),
                    html.P(f"Model Used: {result['model_used']}"),
                    html.P(f"Features Used: {', '.join(result['features_used'][:5])}...")
                ])
            ])
        else:
            return dbc.Alert(f"Error: {response.text}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output("caption-analysis-result", "children"),
    Input("analyze-btn", "n_clicks"),
    Input("caption-input", "value"),
    prevent_initial_call=True
)
def analyze_caption(n_clicks, caption_text):
    """Analyze caption text."""
    if not n_clicks or not caption_text:
        return ""
    
    try:
        # Prepare request data
        request_data = {
            "caption": caption_text
        }
        
        # Make API request
        api_url = os.getenv("API_URL", "http://localhost:8000")
        response = requests.post(f"{api_url}/analyze-caption", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Create analysis result
            return dbc.Card([
                dbc.CardBody([
                    html.H4("Caption Analysis", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Metrics"),
                            html.P(f"Sentiment Score: {result['sentiment_score']:.3f}"),
                            html.P(f"Word Count: {result['word_count']}"),
                            html.P(f"Character Count: {result['char_count']}"),
                            html.P(f"Hashtag Count: {result['hashtag_count']}"),
                            html.P(f"Mention Count: {result['mention_count']}"),
                            html.P(f"URL Count: {result['url_count']}"),
                            html.P(f"Complexity Score: {result['complexity_score']:.3f}")
                        ], width=6),
                        dbc.Col([
                            html.H5("Recommendations"),
                            html.Ul([html.Li(rec) for rec in result['recommendations']])
                        ], width=6)
                    ])
                ])
            ])
        else:
            return dbc.Alert(f"Error: {response.text}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output("times-analysis-result", "children"),
    Input("times-analyze-btn", "n_clicks"),
    Input("times-platform-dropdown", "value"),
    Input("times-content-type-dropdown", "value"),
    prevent_initial_call=True
)
def analyze_best_times(n_clicks, platform, content_type):
    """Analyze best posting times."""
    if not n_clicks:
        return ""
    
    try:
        # Prepare request data
        request_data = {
            "platform": platform,
            "content_type": content_type,
            "days_ahead": 7
        }
        
        # Make API request
        api_url = os.getenv("API_URL", "http://localhost:8000")
        response = requests.post(f"{api_url}/best-times", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Create visualization
            days = [time['day'] for time in result['best_times']]
            engagements = [time['avg_engagement'] for time in result['best_times']]
            
            fig = px.bar(
                x=days,
                y=engagements,
                title=f"Best Posting Times for {platform.title()}",
                labels={"x": "Day of Week", "y": "Average Engagement"}
            )
            
            return dbc.Card([
                dbc.CardBody([
                    html.H4("Best Posting Times", className="card-title"),
                    dcc.Graph(figure=fig),
                    html.H5("Recommendations:"),
                    html.Ul([html.Li(time['recommendation']) for time in result['best_times']])
                ])
            ])
        else:
            return dbc.Alert(f"Error: {response.text}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output("trends-analysis-result", "children"),
    Input("trends-analyze-btn", "n_clicks"),
    Input("trends-platform-dropdown", "value"),
    Input("trends-days-slider", "value"),
    prevent_initial_call=True
)
def analyze_trends(n_clicks, platform, days_back):
    """Analyze platform trends."""
    if not n_clicks:
        return ""
    
    try:
        # Prepare request data
        request_data = {
            "platform": platform,
            "days_back": days_back
        }
        
        # Make API request
        api_url = os.getenv("API_URL", "http://localhost:8000")
        response = requests.post(f"{api_url}/platform-trends", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Create visualization
            metrics = [trend['metric'] for trend in result['trend_data']]
            current_avgs = [trend['current_avg'] for trend in result['trend_data']]
            trends = [trend['trend'] for trend in result['trend_data']]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Current Average Engagement", "Trend Direction"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=metrics, y=current_avgs, name="Current Avg"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=metrics, y=trends, name="Trend"),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            
            return dbc.Card([
                dbc.CardBody([
                    html.H4(f"{platform.title()} Trends", className="card-title"),
                    dcc.Graph(figure=fig),
                    html.H5("Summary:"),
                    html.P(f"Average Engagement: {result['summary_stats']['avg_engagement']:.0f}"),
                    html.P(f"Total Posts: {result['summary_stats']['total_posts']}"),
                    html.P(f"Days Analyzed: {result['summary_stats']['days_analyzed']}"),
                    html.H5("Recommendations:"),
                    html.Ul([html.Li(rec) for rec in result['recommendations']])
                ])
            ])
        else:
            return dbc.Alert(f"Error: {response.text}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output("cross-platform-result", "children"),
    Input("cross-platform-btn", "n_clicks"),
    prevent_initial_call=True
)
def analyze_cross_platform(n_clicks):
    """Analyze cross-platform performance."""
    if not n_clicks:
        return ""
    
    try:
        # This would typically load data and create cross-platform analysis
        # For now, return a placeholder
        return dbc.Card([
            dbc.CardBody([
                html.H4("Cross-Platform Analysis", className="card-title"),
                html.P("Cross-platform analysis feature coming soon..."),
                html.P("This will show performance comparisons across all platforms.")
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

if __name__ == "__main__":
    app.run_server(
        debug=os.getenv("DASHBOARD_DEBUG", "true").lower() == "true",
        host=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.getenv("DASHBOARD_PORT", 8050))
    )