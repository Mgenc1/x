import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import joblib

class VolumePredictor:
    def __init__(self):
        self.time_steps = 60  # Zaman adımlarını tanımla
        self.model = self.build_model()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def build_model(self):
        model = Sequential([
            Input(shape=(self.time_steps, 5)),  # 5 özellik (supertrend, rvi, keltner_upper, keltner_lower, cdv)
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)  # Sadece volume tahmini için 1 çıktı
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                    loss=Huber())  # Huber sınıfını doğrudan kullan
        return model
    
    def prepare_data(self, df):
        # Özellikler ve hedef değişkeni hazırla
        features = df[['supertrend', 'rvi', 'keltner_upper', 'keltner_lower', 'cdv']].values
        target = df['volume'].values
        
        # Zaman serisi veriyi düzenle
        x, y = [], []
        for i in range(len(features)-self.time_steps-1):
            x.append(features[i:(i+self.time_steps)])
            y.append(target[i + self.time_steps])
        return np.array(x), np.array(y)

    def train(self, df):
        x_train, y_train = self.prepare_data(df)
        self.model.fit(x_train, y_train, batch_size=1, epochs=1)
        
    def predict_next_volume(self, df):
        # Scaler'ları mevcut veriyle fit et
        features = df[['supertrend', 'rvi', 'keltner_upper', 'keltner_lower', 'cdv']].values
        target = df['volume'].values.reshape(-1, 1)
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target)
        
        last_60 = df[-60:][['supertrend', 'rvi', 'keltner_upper', 'keltner_lower', 'cdv']].values
        last_60_scaled = self.feature_scaler.transform(last_60)
        prediction = self.model.predict(np.array([last_60_scaled]))
        return self.target_scaler.inverse_transform(prediction)[0][0] 

    def save_weights(self):
        self.model.save_weights('volume_model.weights.h5')
        joblib.dump(self.feature_scaler, 'feature_scaler.save')
        joblib.dump(self.target_scaler, 'target_scaler.save')

    def save(self, filename):
        """Modeli ve scaler'ları kaydeder"""
        self.model.save(filename, save_format='h5')
        joblib.dump(self.feature_scaler, filename.replace('.h5', '_feature_scaler.pkl'))
        joblib.dump(self.target_scaler, filename.replace('.h5', '_target_scaler.pkl')) 