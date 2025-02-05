from flask import Flask
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from binance.client import Client
import datetime
import os
from sunucu.liveChartUI import create_chart_container, get_chart_styles

def create_crypto_chart(api_key, api_secret):
    """Flask ve Dash uygulamasını oluşturur"""
    # Flask uygulaması
    flask_app = Flask(__name__)
    
    # Dash uygulaması
    dash_app = Dash(__name__, server=flask_app, url_base_pathname='/')
    
    # Binance client
    client = Client(api_key, api_secret)
    
    # Layout
    dash_app.layout = html.Div([
        html.Div([
            html.H1("Canlı BTC/USDT Grafik", 
                   style={'color': 'white', 'textAlign': 'center', 'marginBottom': '20px'}),
            # UI bileşenlerini ekle
            create_chart_container(
                dcc.Graph(id='live-graph', style={'height': '100%'}),
                dcc.Interval(id='graph-update', interval=60*1000, n_intervals=0)
            ),
            html.Div(html.Style(get_chart_styles())),
            html.Div(id='debug-output', style={'display': 'none'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto'}),
        
        # Stilleri ekle
        html.Div(dcc.Markdown(get_chart_styles(), dangerously_allow_html=True))
    ])
    
    # Callback fonksiyonu
    @dash_app.callback(Output('live-graph', 'figure'),
                     Input('graph-update', 'n_intervals'))
    def update_graph(n):
        return {
            'data': [],
            'layout': {
                'title': 'TEST GRAFİĞİ',
                'paper_bgcolor': 'white'
            }
        }
    
    @dash_app.callback(
        Output('debug-output', 'children'),
        Input('graph-update', 'n_intervals')
    )
    def debug_callback(n):
        return f"Çerçeve CSS: {get_chart_styles()}"
    
    return flask_app, dash_app
