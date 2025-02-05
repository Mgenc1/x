from dash import Dash, html
import flask

app = flask.Flask(__name__)
dash_app = Dash(__name__, server=app)

dash_app.layout = html.Div(
    style={
        'backgroundColor': 'black',  # Daha belirgin olması için siyah arka plan
        'minHeight': '100vh',
        'padding': '50px',  # Daha belirgin padding
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center'
    },
    children=[
        html.Div(
            "Test Grafik Alanı",
            style={
                'border': '5px solid red',  # Daha kalın çerçeve
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '20px',
                'backgroundColor': '#1e222d',
                'color': 'white',
                'height': '500px',
                'width': '80%',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'fontSize': '24px'
            }
        )
    ]
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8051, debug=True) 