from flask import Flask
from dash import Dash, dcc, html
from sunucu.liveChartUI import create_chart_container, get_chart_styles

# Flask ve Dash başlatıcı
flask_app = Flask(__name__)
dash_app = Dash(__name__, server=flask_app, url_base_pathname='/')

# Kritik layout ayarı
dash_app.layout = html.Div([
    html.Div(get_chart_styles()),  # CSS'i ekle
    create_chart_container(
        dcc.Graph(id='live-graph'),
        dcc.Interval(id='interval', interval=5000)
    )
])

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=8050) 