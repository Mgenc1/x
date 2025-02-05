from flask import Flask
import os
from liveChart import create_app, get_secret  # sunucu. prefix'ini kaldırdık
from liveChartUI import create_layout  # sunucu. prefix'ini kaldırdık

# API anahtarlarını al
secrets = get_secret()
if not secrets:
    raise Exception("API anahtarları alınamadı!")

# Flask ve Dash uygulamasını oluştur
flask_app, dash_app = create_app(secrets['API_KEY'], secrets['API_SECRET'])

# Layout'u ayarla
dash_app.layout = create_layout()

# Health check endpoint
@flask_app.route('/')
def health_check():
    return "OK", 200

# Gunicorn için WSGI application
app = flask_app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port) 