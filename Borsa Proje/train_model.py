from binanceData import BinanceDataCollector
from binanceAIModel import VolumePredictor
from binance.client import Client
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import load_model

# Veri topla
collector = BinanceDataCollector()
df = collector.get_klines_data("BTCUSDT", Client.KLINE_INTERVAL_1HOUR)
df = df.iloc[-1000:]  # Son 1000 mum verisi

# Veri kontrolü ekleyin
if len(df) < 100:
    raise ValueError("Eğitim için yeterli veri yok! En az 100 mum gerekiyor.")

# Modeli eğit
predictor = VolumePredictor()
predictor.train(df)

# Modeli kaydetmeden önce optimizer'ı sıfırla
predictor.model.save_weights('volume_predictor.weights.h5')

# Yeni bir model oluşturup weights'leri yükle
new_model = VolumePredictor()
new_model.model.load_weights('volume_predictor.weights.h5')

# Modeli kaydetme kısmını güncelleyin
predictor.save('volume_predictor.h5')

# Modeli yükleme kısmı
model = load_model('volume_predictor.h5', compile=False)
model.compile(optimizer=Adam(), loss=Huber())

# Tahmin yap
prediction = predictor.predict_next_volume(df)
print(f"Tahmini Sonraki Hacim: {prediction:.2f} BTC") 