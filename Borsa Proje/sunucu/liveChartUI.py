from dash import html, dcc

def create_layout():
    return html.Div([
        html.Div([
            html.H1("BTC/USDT Canlı Fiyat Grafiği",
                   style={
                       'color': 'white',
                       'textAlign': 'center',
                       'marginBottom': '20px'
                   }),
            html.Div(
                id='live-graph',
                className='dash-graph',
                children=[
                    dcc.Graph(
                        id='main-chart',
                        config={'displayModeBar': False}
                    )
                ],
                style={
                    'height': '450px',
                    'border': '5px solid red',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'backgroundColor': '#1e222d',
                }
            )
        ]),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # 5 saniyede bir güncelle
            n_intervals=0
        )
    ],
    style={
        'width': '100%',
        'minHeight': '100vh',
        'backgroundColor': 'black',
        'padding': '20px'
    })

def create_styles():
    """CSS stillerini oluştur"""
    return {
        'control-panel': {
            'display': 'flex',
            'justify-content': 'space-between',
            'padding': '10px',
            'background-color': '#1e222d',
            'border-bottom': '1px solid #2a2e39'
        },
        'time-button': {
            'margin': '0 5px',
            'padding': '5px 15px',
            'background-color': '#2a2e39',
            'border': 'none',
            'color': '#fff',
            'cursor': 'pointer',
            'border-radius': '3px'
        },
        'time-button:hover': {
            'background-color': '#363c4e'
        },
        'time-button-selected': {
            'background-color': '#363c4e'
        },
        'chart-container': {
            'background-color': '#1e222d',
            'height': '100vh'
        }
    }

def create_chart_container(graph_component, interval_component):
    return html.Div(
        className="chart-container",
        style={
            'position': 'absolute',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'border': '5px solid red !important',
            'height': '400px',
            'width': '600px',
            'margin': '20px auto',
            'padding': '10px',
            'backgroundColor': '#2a2a2a',
            'zIndex': '10000'
        },
        children=[
            html.Div("TEST ÇERÇEVESİ", style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '24px'
            }),
            graph_component,
            interval_component
        ]
    )


def get_chart_styles():
    """Tüm özel stilleri içeren CSS'i döndürür"""
    return """
    <style>
        div.chart-container {
            all: unset !important;  /* Tüm varsayılan stilleri sıfırla */
            border: 20px solid cyan !important;
            display: block !important;
            width: 80vw !important;
            height: 70vh !important;
            margin: 50px auto !important;
            position: static !important;  /* Fixed/absolute sorun yaratıyor olabilir */
            z-index: 99999 !important;
            background: rgba(0,0,0,0.9) !important;
        }
    </style>
    """ 