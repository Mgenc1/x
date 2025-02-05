import pandas as pd
import numpy as np
import ta
import ast

class TechnicalAnalyzer:
    def __init__(self):
        self.interval_params = {
            '1m': {
                'supertrend_atr_period': 5,
                'supertrend_multiplier': 2.5,
                'rvi_window': 10,
                'keltner_ema_period': 20,
                'keltner_atr_period': 10,
                'keltner_multiplier': 1.5,
                'fib_lookback': 55
            },
            '5m': {
                'supertrend_atr_period': 7,
                'supertrend_multiplier': 3.0,
                'rvi_window': 14,
                'keltner_ema_period': 20,
                'keltner_atr_period': 14,
                'keltner_multiplier': 2.0,
                'fib_lookback': 89
            },
            '15m': {
                'supertrend_atr_period': 10,
                'supertrend_multiplier': 3.5,
                'rvi_window': 20,
                'keltner_ema_period': 50,
                'keltner_atr_period': 20,
                'keltner_multiplier': 2.5,
                'fib_lookback': 144
            }
        }
        self.params = self.interval_params['5m']  # Default değer

    def set_interval(self, interval):
        """Periyoda göre parametreleri güncelle"""
        self.params = self.interval_params.get(
            interval, 
            self.interval_params['5m']
        )

    def calculate_supertrend(self, df):
        """Supertrend hesapla"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR hesapla
        atr = ta.volatility.average_true_range(
            high, low, close, 
            window=self.params['supertrend_atr_period']
        )
        
        # Temel bantları hesapla
        final_upper = (high + low) / 2 + (self.params['supertrend_multiplier'] * atr)
        final_lower = (high + low) / 2 - (self.params['supertrend_multiplier'] * atr)
        
        # Supertrend değerlerini belirle
        supertrend = pd.Series(index=df.index)
        for i in range(1, len(df)):
            if close[i] > final_upper[i-1]:
                supertrend[i] = final_lower[i]
            elif close[i] < final_lower[i-1]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = supertrend[i-1]
        return supertrend

    def calculate_rvi(self, df):
        """Relative Vigor Index hesapla"""
        rvi = ta.momentum.rsi(
            df['close'], 
            window=self.params['rvi_window']
        )
        rvi_signal = rvi.rolling(4).mean()
        return rvi, rvi_signal

    def calculate_keltner(self, df):
        """Keltner Channel hesapla"""
        keltner = ta.volatility.KeltnerChannel(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.params['keltner_ema_period'],
            window_atr=self.params['keltner_atr_period'],
            multiplier=self.params['keltner_multiplier']
        )
        return keltner.keltner_channel_hband(), keltner.keltner_channel_lband()

    def calculate_fibonacci(self, df):
        """Tüm Fibonacci seviyelerini hesapla"""
        lookback = self.params['fib_lookback']
        max_price = df['high'].rolling(lookback).max()
        min_price = df['low'].rolling(lookback).min()
        return {
            '0.236': max_price - (max_price - min_price) * 0.236,
            '0.382': max_price - (max_price - min_price) * 0.382,
            '0.5': max_price - (max_price - min_price) * 0.5,
            '0.618': max_price - (max_price - min_price) * 0.618,
            '0.786': max_price - (max_price - min_price) * 0.786
        }

    def calculate_indicators(self, df):
        """Güvenli gösterge hesaplama"""
        try:
            # Gerekli sütunları kontrol et
            required_cols = ['high', 'low', 'close', 'volume', 'taker_buy_base']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Eksik sütunlar: Veri formatı değişmiş olabilir")
            
            if df.empty or len(df) < 50:
                raise ValueError("Hesaplama için yetersiz veri")
            
            # Orijinal hesaplama mantığı
            df['supertrend'] = self.calculate_supertrend(df)
            df['rvi'], df['rvi_signal'] = self.calculate_rvi(df)
            df['keltner_upper'], df['keltner_lower'] = self.calculate_keltner(df)
            df['vwap'] = (df['volume'] * (df['high']+df['low']+df['close'])/3).cumsum() / df['volume'].cumsum()
            df['cdv'] = (df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])).cumsum()
            
            fib = self.calculate_fibonacci(df)
            for level, values in fib.items():
                df[f'fib_{level}'] = values
            
            return df.dropna()
        
        except Exception as e:
            print(f"Gösterge hesaplama hatası: {str(e)}")
            return df

    def generate_signals(self, df):
        """Tüm sinyalleri birleştiren ana metod"""
        signals = []
        
        try:
            # 1. Temel Teknik Sinyaller
            latest = df.iloc[-1]
            if latest['close'] > latest['supertrend']:
                signals.append(('AL', 'Supertrend AL Sinyali'))
            else:
                signals.append(('SAT', 'Supertrend SAT Sinyali'))
            
            # 2. Likidasyon Sinyalleri
            from liquidationPoints import LiquidationAnalyzer
            analyzer = LiquidationAnalyzer()
            heatmap = analyzer.calculate_liquidation_levels(df)  # Heatmap oluşturuldu
            levels = analyzer.calculate_levels(df, heatmap)
            
            if latest['close'] > levels.get('resistance', 0):
                signals.append(('AL', f'Likidasyon Direnç: {levels["resistance"]:.2f}'))
            elif latest['close'] < levels.get('support', 0):
                signals.append(('SAT', f'Likidasyon Destek: {levels["support"]:.2f}'))
            
            # 3. Momentum Onayı
            if latest['rvi'] > latest['rvi_signal']:
                signals.append(('AL', 'RVI Momentum'))
            else:
                signals.append(('SAT', 'RVI Zayıf'))
            
            return signals
        
        except Exception as e:
            print(f"Sinyal oluşturma hatası: {str(e)}")
            return [('HATA', 'Sinyaller oluşturulamadı')]

    def print_technical_analysis(self, df, interval, symbol):
        try:
            # Veri boyutu kontrolü
            if len(df) < 2:
                raise ValueError("Analiz için yetersiz veri")
            
            print("\n📈 TEKNİK GÖSTERGELER:")
            print("-"*40)
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # 1. Trend Analizi
            print("\n1️⃣ TREND ANALİZİ:")
            trend_status = "YUKARI" if current_price > latest['supertrend'] else "AŞAĞI"
            print(f"• Supertrend: {trend_status} ({latest['supertrend']:.2f})")
            
            # 2. Momentum Analizi
            print("\n2️⃣ MOMENTUM ANALİZİ:")
            rvi_diff = latest['rvi'] - latest['rvi_signal']
            print(f"• RVI: {latest['rvi']:.2f} ({'YÜKSELİŞ' if rvi_diff > 0 else 'DÜŞÜŞ'})")
            print(f"• RVI Sinyal: {latest['rvi_signal']:.2f}")
            
            # 3. Hacim Analizi
            print("\n3️⃣ HACIM ANALİZİ:")
            vwap_diff = (current_price - latest['vwap'])/latest['vwap']*100
            print(f"• VWAP: {latest['vwap']:.2f} ({'Üzerinde' if vwap_diff > 0 else 'Altında'})")
            print(f"• VWAP Farkı: {vwap_diff:.2f}%")
            
            # 4. Volatilite
            print("\n4️⃣ VOLATİLİTE:")
            print(f"• Keltner Aralığı: {latest['keltner_upper'] - latest['keltner_lower']:.2f}")
            print(f"• Üst Bant: {latest['keltner_upper']:.2f}")
            print(f"• Alt Bant: {latest['keltner_lower']:.2f}")
            
            # 5. Fibonacci Seviyeleri
            print("\n5️⃣ FIBONACCI SEVİYELERİ:")
            for level in ['0.236', '0.382', '0.5', '0.618', '0.786']:
                fib_value = latest[f'fib_{level}']
                diff = (current_price - fib_value)/fib_value*100
                print(f"• {float(level)*100:.1f}%: {fib_value:.2f} ({'↑' if diff > 0 else '↓'} {abs(diff):.2f}%)")
            
            # 6. Likidasyon Analizi
            print("\n6️⃣ LİKİDİTE ANALİZİ:")
            print(f"• Kümülatif Delta: {latest['cdv']:.2f} ({'Pozitif' if latest['cdv'] > 0 else 'Negatif'})")

        except Exception as e:
            print(f"\n❌ Genel analiz hatası: {str(e)}") 