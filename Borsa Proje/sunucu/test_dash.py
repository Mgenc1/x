from dash import Dash, html
import dash_bootstrap_components as dbc
import flask

app = flask.Flask(__name__)
dash_app = Dash(
    __name__, 
    server=app,
    external_stylesheets=[dbc.themes.DARKLY]
)

dash_app.layout = html.Div(
    style={'backgroundColor': '#131722', 'minHeight': '100vh', 'padding': '20px'},
    children=[
        # Test için kırmızı çerçeveli bir div
        html.Div(
            "Test Grafik Alanı",
            style={
                'border': '2px solid red',
                'borderRadius': '5px',
                'padding': '20px',
                'margin': '20px',
                'backgroundColor': '#1e222d',
                'color': 'white',
                'height': '500px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'fontSize': '24px'
            }
        )
    ]
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True) 