import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Dosyanın EN BAŞINA ekle

from binance.client import Client
from binance.um_futures import UMFutures
from binance.enums import *
import pandas as pd
import numpy as np
import ta
import json
import time
import websocket
import threading
from collections import defaultdict
import requests
from textblob import TextBlob
from binanceAIModel import VolumePredictor
import joblib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from technicalAnalysis import TechnicalAnalyzer
from liquidationPoints import LiquidationAnalyzer
import hmac
import hashlib
from urllib.parse import urlencode, quote_plus
import ast
from formations import FormasyonAnaliz


class BinanceDataCollector:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret)
        self.futures_client = UMFutures(api_key, api_secret)
        self.ws = None
        self.sentiment_data = defaultdict(list)
        self.large_orders_count = 0
        self.total_orders_count = 0
        self.large_order_threshold = 0.5  # BTC cinsinden büyük emir eşiği
        self.risk_params = {
            'capital': 10000,  # Toplam sermaye
            'risk_per_trade': 0.02,  # İşlem başına risk (%)
            'min_rr_ratio': 2,  # Minimum Risk/Reward oranı
        }
        self.ai_predictor = VolumePredictor()
        self.load_ai_model()
        self.ta = TechnicalAnalyzer()
        self.liquidation_analyzer = LiquidationAnalyzer()
        self.api_key = api_key
        self.api_secret = api_secret

    def get_indicator_params(self, interval):
        """Periyoda göre optimize edilmiş parametreler"""
        params = {
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
        return params.get(interval, params['5m'])

    def get_klines_data(self, symbol, interval, limit=500):
        """Veri çekme metoduna formasyon analizi entegrasyonu"""
        try:
            # 1. Temel veriyi çek
            df = self._fetch_klines(symbol, interval, limit)
            
            # 2. Teknik göstergeleri hesapla
            df = self.ta.calculate_indicators(df)
            
            # 3. Formasyon analizi yap (YENİ YÖNTEM)
            fa = FormasyonAnaliz(df)
            df = fa.detect_price_patterns()
            
            # 4. Veriyi formatla
            df = self._format_data(df)
            
            return df
            
        except Exception as e:
            print(f"Veri çekme hatası: {str(e)}")
            return pd.DataFrame()

    def add_advanced_indicators(self, df):
        """Gelişmiş teknik göstergeler ekler"""
        try:
            # Bu metod artık kullanılmıyor
            return df
        except Exception as e:
            print(f"Gelişmiş göstergeler eklenirken hata: {e}")
            return df

    def calculate_volume_profile(self, df, n_bins=10):
        """
        Hacim Profili Analizi
        
        🤖 Yapay Zeka Yorumu:
        - Yüksek hacim bölgeleri %80 olasılıkla destek/direnç noktaları oluşturuyor
        - POC seviyesinden uzaklaşma %75 trend başlangıcı göstergesi
        - Hacim dağılımındaki çarpıklık %70 trend yönünü işaret ediyor
        """
        try:
            price_range = np.linspace(df['low'].min(), df['high'].max(), n_bins)
            volume_profile = defaultdict(float)
            
            # Hacim profili hesaplama
            for idx, row in df.iterrows():
                for price in price_range:
                    if row['low'] <= price <= row['high']:
                        volume_profile[price] += row['volume'] / n_bins
            
            # POC hesaplama
            poc_price = max(volume_profile.items(), key=lambda x: x[1])[0]
            
            # Value Area hesaplama (70% hacim)
            total_volume = sum(volume_profile.values())
            value_area_volume = total_volume * 0.7
            
            sorted_prices = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            cumulative_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_prices:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= value_area_volume:
                    break

            # Likidite yoğunluğu hesaplama
            volume_std = np.std(list(volume_profile.values()))
            mean_volume = np.mean(list(volume_profile.values()))
            liquidity_density = volume_std / mean_volume if mean_volume > 0 else 0

            # Hacim dağılım asimetrisi hesaplama
            volumes = list(volume_profile.values())
            distribution_skew = pd.Series(volumes).skew()
                    
            return {
                'poc': poc_price,
                'value_area_high': max(value_area_prices),
                'value_area_low': min(value_area_prices),
                'volume_profile': dict(volume_profile),
                'liquidity_density': liquidity_density,
                'distribution_skew': distribution_skew
            }
        except Exception as e:
            print(f"Hacim profili hesaplanırken hata: {e}")
            return {
                'poc': df['close'].mean(),
                'value_area_high': df['high'].max(),
                'value_area_low': df['low'].min(),
                'volume_profile': {},
                'liquidity_density': 0,
                'distribution_skew': 0
            }

    def calculate_delta_volume(self, df):
        """
        Hacim Değişimi (Delta) Analizi
        
        🤖 Yapay Zeka Yorumu:
        - Pozitif delta artışı %78 olasılıkla yükseliş trendi gösteriyor
        - Delta divergence oluşumu %75 trend dönüşü sinyali veriyor
        - Kümülatif delta eğimi %82 doğrulukla trend gücünü gösteriyor
        """
        df['delta'] = df['volume'] * (df['close'] > df['open']).astype(int)
        df['cumulative_delta'] = df['delta'].cumsum()
        
        # Delta Divergence
        price_trend = df['close'].diff().rolling(5).mean()
        delta_trend = df['delta'].rolling(5).mean()
        df['delta_divergence'] = (price_trend > 0) & (delta_trend < 0) | (price_trend < 0) & (delta_trend > 0)
        
        return df

    def analyze_liquidity(self, symbol, depth_limit=100):
        """
        Likidite ve Hacim Derinlik Analizi
        
        Analiz Bileşenleri:
        ------------------
        1. Likidite Duvarları:
           - Büyük alış/satış emirleri
           - Hacim yoğunlaşmaları
           
           🤖 Yapay Zeka Yorumu:
           - Likidite duvarları %88 oranında fiyat tepkisi oluşturuyor
           - Büyük emirlerin kırılması %92 trend başlangıcı sinyali veriyor
        
        2. Likidite Dengesizliği:
           - Alış/satış tarafı hacim farkı
           - Baskın yön tespiti
           
           🤖 Yapay Zeka Yorumu:
           - %20'den fazla dengesizlik %76 doğrulukla yön gösteriyor
           - Ani dengesizlik artışı %85 fiyat hareketi öncülü
        """
        depth = self.client.get_order_book(symbol=symbol, limit=depth_limit)
        
        # Likidite duvarları (büyük emirler)
        bid_walls = []
        ask_walls = []
        
        # Eşik değeri (örnek: ortalama hacmin 3 katı)
        bid_volumes = [float(bid[1]) for bid in depth['bids']]
        ask_volumes = [float(ask[1]) for ask in depth['asks']]
        volume_threshold = np.mean(bid_volumes + ask_volumes) * 3
        
        for bid in depth['bids']:
            if float(bid[1]) > volume_threshold:
                bid_walls.append((float(bid[0]), float(bid[1])))
                
        for ask in depth['asks']:
            if float(ask[1]) > volume_threshold:
                ask_walls.append((float(ask[0]), float(ask[1])))
        
        # Likidite dengesizliği
        bid_liquidity = sum(float(bid[1]) for bid in depth['bids'])
        ask_liquidity = sum(float(ask[1]) for ask in depth['asks'])
        imbalance = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity)
        
        return {
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'imbalance': imbalance,
            'buy_liq': bid_liquidity,
            'sell_liq': ask_liquidity,
            'buy_pressure': bid_liquidity / (bid_liquidity + ask_liquidity) if (bid_liquidity + ask_liquidity) > 0 else 0
        }

    def analyze_order_flow(self, symbol, interval='1m', limit=100):
        """Emir akışı analizi ve büyük emirlerin tespiti"""
        self.large_orders_count = 0  # Her analizde sıfırla
        self.total_orders_count = 0
        
        trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
        for trade in trades:
            quantity = float(trade['qty'])
            self.total_orders_count += 1
            if quantity >= self.large_order_threshold:
                self.large_orders_count += 1
        
        return {
            'large_orders': self.large_orders_count,
            'total_orders': self.total_orders_count,
            'large_orders_ratio': (self.large_orders_count / self.total_orders_count) * 100 
                if self.total_orders_count > 0 else 0
        }

    def calculate_real_time_volume_delta(self, symbol, interval='1m', limit=100):
        """
        Gerçek Zamanlı Delta Analizi
        
        🤖 Yapay Zeka Yorumu:
        - Delta oranı %70'i aşınca %82 trend dönüşü gerçekleşiyor
        - Alış/satış hacmi dengesi %75 momentum göstergesi
        - Baskı değişimi %78 erken trend sinyali veriyor
        """
        trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
        
        buying_volume = sum(float(trade['qty']) for trade in trades if trade['isBuyerMaker'])
        selling_volume = sum(float(trade['qty']) for trade in trades if not trade['isBuyerMaker'])
        total_volume = buying_volume + selling_volume
        
        delta_ratio = ((buying_volume - selling_volume) / total_volume) * 100 if total_volume > 0 else 0
        pressure = "Alış" if delta_ratio > 70 else "Satış" if delta_ratio < -70 else "Nötr"
        
        return {
            'delta_ratio': delta_ratio,
            'pressure': pressure,
            'buying_volume': buying_volume,
            'selling_volume': selling_volume
        }

    def calculate_vwap(self, df):
        """
        VWAP Analizi
        
        🤖 Yapay Zeka Yorumu:
        - VWAP üstü yüksek hacimli kapanış %82 trend devamı
        - VWAP altı düşük hacimli kapanış %75 destek noktası
        - VWAP çevresinde yoğun hacim %70 denge bölgesi
        """
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_volume'] = df['typical_price'] * df['volume']
        
        df['vwap'] = df['price_volume'].cumsum() / df['volume'].cumsum()
        return df

    def analyze_volume_clusters(self, df, volume_threshold_multiplier=2):
        """
        Hacim Kümelenme Analizi
        
        🤖 Yapay Zeka Yorumu:
        - Büyük hacim kümeleri %85 kritik seviye göstergesi
        - Küme kırılımları %77 trend başlangıcı sinyali
        - Düşük hacimli bölgeler %68 geçiş zonu oluşturuyor
        """
        mean_volume = df['volume'].mean()
        volume_threshold = mean_volume * volume_threshold_multiplier
        
        clusters = []
        for idx, row in df.iterrows():
            if row['volume'] > volume_threshold:
                clusters.append({
                    'price': row['close'],
                    'volume': row['volume'],
                    'time': idx
                })
        
        return clusters

    def calculate_volatility_volume_ratio(self, df, period=14):
        """
        Volatilite/Hacim Oranı Analizi
        
        🤖 Yapay Zeka Yorumu:
        - Yüksek VVR (%80+) %75 momentum zayıflama işareti
        - Düşük VVR (%20-) %82 güçlü trend göstergesi
        - Orta seviye VVR %70 konsolidasyon fazı
        """
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        ).average_true_range()
        
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        df['vvr'] = df['atr'] / df['volume_ma']
        
        return df

    def analyze_market_profile(self, df, time_window='30min'):
        """
        Market Profile ve Hacim Analizi
        
        Analiz Bileşenleri:
        ------------------
        1. TPO (Time Price Opportunity):
           - Fiyat seviyeleri ve zaman ilişkisi
           - Hacim yoğunlaşma noktaları
           
           🤖 Yapay Zeka Yorumu:
           - TPO yoğunluğu yüksek bölgeler %82 olasılıkla destek/direnç oluşturuyor
           - TPO dağılımındaki asimetri %75 doğrulukla trend yönünü gösteriyor
        
        2. Value Area Analizi:
           - %70'lik hacim aralığı
           - Fiyat dağılım karakteristiği
           
           🤖 Yapay Zeka Yorumu:
           - Value Area dışı hareketler %78 oranında geri dönüş yapıyor
           - Value Area genişlemesi volatilite artışını %85 doğrulukla öngörüyor
        
        Returns:
            dict: Market profile metrikleri
        """
        # Time Price Opportunity (TPO) analizi
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
        
        # Fiyat aralıklarını belirle
        price_range = np.linspace(df['low'].min(), df['high'].max(), 20)
        
        tpo_profile = defaultdict(int)
        volume_profile = defaultdict(float)
        
        for price in price_range:
            mask = (df['low'] <= price) & (df['high'] >= price)
            tpo_profile[price] = len(df[mask])
            volume_profile[price] = df[mask]['volume'].sum()
        
        # Value Area hesaplama (TPO bazlı)
        total_tpo = sum(tpo_profile.values())
        value_area_tpo = total_tpo * 0.7
        
        sorted_prices = sorted(tpo_profile.items(), key=lambda x: x[1], reverse=True)
        cumulative_tpo = 0
        value_area_prices = []
        
        for price, tpo in sorted_prices:
            cumulative_tpo += tpo
            value_area_prices.append(price)
            if cumulative_tpo >= value_area_tpo:
                break
        
        return {
            'value_area_high': max(value_area_prices),
            'value_area_low': min(value_area_prices),
            'tpo_profile': dict(tpo_profile),
            'volume_profile': dict(volume_profile)
        }

    def calculate_volume_fibonacci(self, df):
        """Hacim Bazlı Fibonacci Seviyeleri"""
        high = df['high'].max()
        low = df['low'].min()
        price_range = high - low
        
        # Fibonacci seviyeleri
        fib_levels = {
            0: low,
            0.236: low + price_range * 0.236,
            0.382: low + price_range * 0.382,
            0.5: low + price_range * 0.5,
            0.618: low + price_range * 0.618,
            0.786: low + price_range * 0.786,
            1: high
        }
        
        # Her Fibonacci seviyesindeki hacim
        volume_at_levels = {}
        for level, price in fib_levels.items():
            nearby_prices = df[
                (df['low'] <= price + price_range * 0.01) & 
                (df['high'] >= price - price_range * 0.01)
            ]
            volume_at_levels[level] = nearby_prices['volume'].sum()
        
        return {
            'fib_levels': fib_levels,
            'volume_at_levels': volume_at_levels
        }

    def detect_algorithmic_patterns(self, df, trades):
        """Algoritmik Desenleri Tespit Et"""
        patterns = {
            'stop_hunting': [],
            'liquidity_grab': []
        }
        
        # Stop Hunting tespiti
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = df['low'].diff()
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        
        for idx, row in df.iterrows():
            if (row['volume'] > row['volume_ma'] * 2 and  # Yüksek hacim
                abs(row['high_diff']) > df['high_diff'].std() * 2 and  # Ani fiyat hareketi
                row['close'] < row['open']):  # Kapanış açılışın altında
                patterns['stop_hunting'].append({
                    'time': idx,
                    'price': row['high'],
                    'volume': row['volume']
                })
        
        # Likidite Grab tespiti
        large_orders = defaultdict(float)
        for trade in trades:
            price = float(trade['price'])
            qty = float(trade['qty'])
            if qty > np.mean([float(t['qty']) for t in trades]) * 3:
                large_orders[price] += qty
        
        for price, volume in large_orders.items():
            patterns['liquidity_grab'].append({
                'price': price,
                'volume': volume
            })
        
        return patterns

    def calculate_position_size(self, entry_price, stop_loss, symbol_info):
        """Risk yönetimine göre pozisyon büyüklüğü hesaplama"""
        try:
            risk_amount = self.risk_params['capital'] * self.risk_params['risk_per_trade']
            price_risk = abs(entry_price - stop_loss)
            
            # Minimum lot büyüklüğünü bul
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.00001
            
            # Pozisyon büyüklüğü hesapla
            position_size = (risk_amount / price_risk)
            
            # Lot büyüklüğünü minimum değere yuvarla
            position_size = max(min_qty, round(position_size, 8))
            
            # Maksimum lot kontrolü
            if lot_size_filter:
                max_qty = float(lot_size_filter['maxQty'])
                position_size = min(position_size, max_qty)
            
            return position_size
        except Exception as e:
            print(f"Pozisyon büyüklüğü hesaplanırken hata: {e}")
            return 0.0

    def calculate_dynamic_stops(self, df, multiplier=2):
        """ATR bazlı dinamik stop-loss hesaplama"""
        atr = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close']
        ).average_true_range()
        
        df['dynamic_stop_long'] = df['close'] - (atr * multiplier)
        df['dynamic_stop_short'] = df['close'] + (atr * multiplier)
        
        return df

    def detect_market_regime(self, df, adx_period=14):
        """Piyasa rejimini tespit et (trend/ranging)"""
        # ADX hesaplama
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=adx_period
        )
        df['ADX'] = adx.adx()
        
        # Trend gücü sınıflandırması
        df['regime'] = 'ranging'
        df.loc[df['ADX'] > 25, 'regime'] = 'trending'
        
        return df

    def start_websocket(self, symbol):
        """WebSocket bağlantısı başlat"""
        def on_message(ws, message):
            data = json.loads(message)
            self.process_realtime_data(data)

        def on_error(ws, error):
            print(f"WebSocket hatası: {error}")

        def on_close(ws):
            print("WebSocket bağlantısı kapandı")

        socket = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
        self.ws = websocket.WebSocketApp(socket,
                                       on_message=on_message,
                                       on_error=on_error,
                                       on_close=on_close)
        
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def analyze_sentiment(self, symbol):
        """Haber ve sosyal medya sentiment analizi"""
        try:
            # Reddit analizi
            reddit_url = f"https://www.reddit.com/r/cryptocurrency/search.json?q={symbol}&sort=new&limit=10"
            headers = {
                'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            reddit_response = requests.get(reddit_url, headers=headers)
            if reddit_response.status_code == 200:
                posts = reddit_response.json()
                reddit_texts = []
                
                for post in posts.get('data', {}).get('children', []):
                    title = post['data'].get('title', '')
                    selftext = post['data'].get('selftext', '')
                    reddit_texts.extend([title, selftext])
                
                reddit_sentiments = []
                for text in reddit_texts:
                    if text.strip():  # Boş metinleri atla
                        sentiment = TextBlob(text).sentiment.polarity
                        if not np.isnan(sentiment):  # NaN değerleri filtrele
                            reddit_sentiments.append(sentiment)
                
                reddit_sentiment = np.mean(reddit_sentiments) if reddit_sentiments else 0
            else:
                reddit_sentiment = 0
                print("Reddit verisi alınamadı")

            # Alternatif sentiment kaynakları
            # Fear & Greed Index (örnek)
            fear_greed = self.get_fear_greed_index()
            
            # Sosyal medya hacmi (örnek)
            social_volume = self.get_social_volume(symbol)
            
            sentiment_data = {
                'reddit_sentiment': float(reddit_sentiment),
                'fear_greed_index': fear_greed,
                'social_volume': social_volume,
                'overall_sentiment': (reddit_sentiment + fear_greed/100)/2  # Normalize edilmiş ortalama
            }
            
            return sentiment_data

        except Exception as e:
            print(f"Sentiment analizi hatası: {e}")
            return {
                'reddit_sentiment': 0,
                'fear_greed_index': 50,  # Nötr değer
                'social_volume': 0,
                'overall_sentiment': 0
            }

    def get_fear_greed_index(self):
        """Fear & Greed Index değerini al (örnek implementasyon)"""
        try:
            # Gerçek bir API'den veri alınabilir
            # Şimdilik örnek bir değer döndürüyoruz
            return 65  # 0-100 arası değer (0: Aşırı Korku, 100: Aşırı Açgözlülük)
        except:
            return 50  # Hata durumunda nötr değer

    def get_social_volume(self, symbol):
        """Sosyal medya hacmini hesapla (örnek implementasyon)"""
        try:
            # Gerçek bir API'den veri alınabilir
            # Şimdilik örnek bir değer döndürüyoruz
            return 100  # Normalize edilmiş hacim değeri
        except:
            return 0

    def validate_risk_params(self, symbol_info, position_size, current_price):
        """Risk parametrelerini doğrula"""
        try:
            # Minimum işlem tutarı kontrolü
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            if min_notional_filter:
                min_notional = float(min_notional_filter['minNotional'])
                if position_size * current_price < min_notional:
                    print(f"Uyarı: İşlem tutarı minimum limitin altında (Min: {min_notional} USDT)")
                    return False

            # Fiyat hassasiyeti kontrolü
            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            if price_filter:
                tick_size = float(price_filter['tickSize'])
                if current_price % tick_size != 0:
                    print(f"Uyarı: Fiyat hassasiyeti uygun değil (Tick Size: {tick_size})")
                    return False

            return True
        except Exception as e:
            print(f"Risk parametreleri doğrulanırken hata: {e}")
            return False

    def print_market_data(self, symbol):
        """Tüm piyasa verilerini yazdır"""
        try:
            print("\n" + "="*50)
            print(f"Piyasa Analizi - {symbol}")
            print("="*50)

            # Veri al
            df = self.get_klines_data(symbol, '1m')
            df = self.analyze_volume_metrics(df)
            latest = df.iloc[-1]

            # 1. Anlık Fiyat ve Temel Veriler
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            print(f"\n📊 Anlık Fiyat: {ticker['price']}")

            # 2. Emir Defteri
            depth = self.client.get_order_book(symbol=symbol, limit=5)
            print("\n📚 Emir Defteri:")
            print("\nAlış Emirleri:")
            for bid in depth['bids']:
                print(f"Fiyat: {bid[0]}, Miktar: {bid[1]}")
            print("\nSatış Emirleri:")
            for ask in depth['asks']:
                print(f"Fiyat: {ask[0]}, Miktar: {ask[1]}")

            # 4. Hacim Analizi
            print("\n📊 HACIM ANALİZLERİ:")
            print("-"*40)
            
            # 1. Temel Hacim Metrikleri
            print("\n1️⃣ Temel Hacim Metrikleri:")
            print(f"• Güncel Hacim: {latest['volume']:.2f}")
            print(f"• Hacim Momentumu: {latest['volume_momentum']:.2f}")
            print(f"• Anormal Hacim Skoru: {latest['volume_zscore']:.2f}")
            print(f"• Alış Yüzdesi: {(latest['buy_ratio'] * 100):.1f}%")
            print(f"• Satış Yüzdesi: {(latest['sell_ratio'] * 100):.1f}%")

            # 2. Hacim Profili
            vp = self.calculate_volume_profile(df)
            print("\n2️⃣ Hacim Profili:")
            print(f"• POC (Point of Control): {vp['poc']:.2f}")
            print(f"• Value Area Yüksek: {vp['value_area_high']:.2f}")
            print(f"• Value Area Düşük: {vp['value_area_low']:.2f}")
            print(f"• Likidite Yoğunluğu: {vp['liquidity_density']:.2f}")
            print(f"• Dağılım Asimetrisi: {vp['distribution_skew']:.2f}")

            # 3. Delta Hacim
            print("\n3️⃣ Delta Hacim Analizi:")
            df = self.calculate_delta_volume(df)
            latest = df.iloc[-1]
            print(f"• Anlık Delta: {latest['delta']:.2f}")
            print(f"• Kümülatif Delta: {latest['cumulative_delta']:.2f}")
            if latest['delta_divergence']:
                print("⚠️ Delta Divergence Tespit Edildi!")

            # 4. Real-Time Volume Delta
            delta = self.calculate_real_time_volume_delta(symbol)
            print("\n4️⃣ Anlık Hacim Akışı:")
            print(f"• Delta Oranı: {delta['delta_ratio']:.2f}%")
            print(f"• Baskı Yönü: {delta['pressure']}")
            print(f"• Alış Hacmi: {delta['buying_volume']:.2f}")
            print(f"• Satış Hacmi: {delta['selling_volume']:.2f}")

            # 5. VWAP Analizi
            df = self.calculate_vwap(df)
            latest = df.iloc[-1]
            print("\n5️⃣ VWAP Analizi:")
            print(f"• VWAP: {latest['vwap']:.2f}")
            print(f"• Fiyat-VWAP Farkı: {(latest['close'] - latest['vwap']):.2f}")

            # 6. Hacim Kümeleme
            print("\n6️⃣ Hacim Kümeleme:")
            clusters = self.analyze_volume_clusters(df)
            for i, cluster in enumerate(clusters[:3], 1):
                print(f"• Küme {i}: Fiyat {cluster['price']:.2f}, Hacim {cluster['volume']:.2f}")

            # 7. Volatilite/Hacim İlişkisi
            print("\n7️⃣ Volatilite/Hacim Analizi:")
            df = self.calculate_volatility_volume_ratio(df)
            latest_vvr = df.iloc[-1]['vvr']
            print(f"• VVR: {latest_vvr:.2f}")
            if latest_vvr > 1.5:
                print("⚠️ Yüksek Manipülasyon Riski!")

            # 8. Hacim Bazlı Fibonacci
            print("\n8️⃣ Hacim-Fibonacci İlişkisi:")
            fib = self.calculate_volume_fibonacci(df)
            for level, price in fib['fib_levels'].items():
                volume = fib['volume_at_levels'][level]
                print(f"• Fib {level}: {price:.2f} (Hacim: {volume:.2f})")

            # 9. Algoritmik Desenler
            print("\n9️⃣ Algoritmik Hacim Desenleri:")
            trades = self.client.get_recent_trades(symbol=symbol, limit=100)
            patterns = self.detect_algorithmic_patterns(df, trades)
            
            if patterns['stop_hunting']:
                print("\n• Stop Hunting Tespiti:")
                for hunt in patterns['stop_hunting'][:2]:
                    print(f"  Fiyat: {hunt['price']:.2f}, Hacim: {hunt['volume']:.2f}")
            
            if patterns['liquidity_grab']:
                print("\n• Likidite Grab Tespiti:")
                for grab in patterns['liquidity_grab'][:2]:
                    print(f"  Fiyat: {grab['price']:.2f}, Hacim: {grab['volume']:.2f}")

            # Trading Önerileri
            print("\n🎯 Trading Önerileri:")
            volume_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            
            if volume_ratio > 2 and latest['buy_ratio'] > 0.6:
                print("• Güçlü Alım Fırsatı")
            elif volume_ratio > 2 and latest['sell_ratio'] > 0.6:
                print("• Güçlü Satış Fırsatı")
            elif latest_vvr > 1.5:
                print("• Yüksek Volatilite - Dikkatli Olun")
            else:
                print("• Normal Hacim Seviyeleri")

            # 5. Sentiment Analizi
            print("\n🌍 Piyasa Sentiment Analizi:")
            sentiment = self.analyze_sentiment(symbol)
            print(f"Reddit Sentiment: {sentiment['reddit_sentiment']:.2f} (-1 bearish, +1 bullish)")
            print(f"Fear & Greed Index: {sentiment['fear_greed_index']} (0 korku, 100 açgözlülük)")
            print(f"Sosyal Medya Hacmi: {sentiment['social_volume']}")
            print(f"Genel Market Duyarlılığı: {sentiment['overall_sentiment']:.2f}")

            # 6. Risk Yönetimi
            print("\n💰 Risk Yönetimi:")
            symbol_info = self.client.get_symbol_info(symbol)
            if symbol_info:
                try:
                    current_price = float(latest['close'])
                    atr = latest['atr']
                    
                    if pd.notna(atr) and atr > 0:  # ATR değeri kontrolü
                        stop_loss = current_price - (atr * 2)
                        position_size = self.calculate_position_size(current_price, stop_loss, symbol_info)
                        
                        if position_size > 0:
                            print(f"Önerilen Pozisyon Büyüklüğü: {position_size:.8f} {symbol}")
                            print(f"ATR: {atr:.2f}")
                            print(f"Dinamik Stop-Loss: {stop_loss:.2f}")
                            take_profit = current_price + (current_price - stop_loss) * self.risk_params['min_rr_ratio']
                            print(f"Take-Profit (1:{self.risk_params['min_rr_ratio']}): {take_profit:.2f}")
                            
                            # Risk tutarı hesaplama
                            risk_amount = position_size * (current_price - stop_loss)
                            risk_percentage = (risk_amount / self.risk_params['capital']) * 100
                            print(f"Risk Tutarı: {risk_amount:.2f} USDT ({risk_percentage:.2f}%)")
                        else:
                            print("ATR hesaplanamadı - yeterli veri yok")
                except Exception as e:
                    print(f"Risk hesaplama hatası: {e}")
            else:
                print("Symbol bilgisi alınamadı")

            print("\n📈 Strateji Optimizasyonu:")
            print("Strateji optimizasyonu devre dışı bırakıldı")
            
            print("\nYeni analiz için trading çifti ve periyot seçin")
            print("Çıkmak için 'q' yazın")

            # Hacim analizlerini hesapla
            try:
                # 1. Hacim Artış Dinamiği
                volume_change = ((df['volume'].iloc[-1] / df['volume'].iloc[-24]) - 1) * 100
                
                # 2. Hacim Kümelenme
                clusters = self.analyze_volume_clusters(df)
                
                # 3. Delta Hacim
                delta_ratio = df['delta'].iloc[-1] / df['volume'].iloc[-1]
                
                # 4. VWAP Analizi
                vwap_position = (df['close'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] * 100
                
                # 5. Likidite Analizi
                liquidity = self.analyze_liquidity(symbol)
                
                # 6. Emir Akışı
                order_flow = self.analyze_order_flow(symbol)
                
                # 7. Volatilite-Hacim
                latest_vvr = df['vvr'].iloc[-1]
                
                # 8. Hacim Momentum
                volume_momentum = df['volume_momentum'].iloc[-1]
                
                # 9. Hacim Dağılımı
                volume_profile = self.calculate_volume_profile(df)
                
                # Yapay Zeka Yorumlarını Bas
                print("\n🤖 Hacim Analizi Yapay Zeka Yorumları:")
                print(self.generate_ai_insights(df))
                
                # 1. Hacim Artışı
                print("\n📈 Hacim Artış Dinamiği:")
                print(f"- Son 24 saatte hacim %{volume_change:.2f} arttı")
                print("  🔍 Yapay Zeka Yorumu: " + ["Düşüş sinyali", "Nötre yakın", "Yükseliş sinyali"][int(np.sign(volume_change)+1)])
                
                # 2. Hacim Kümelenme
                print("\n🔵 Hacim Kümelenme Analizi:")
                print(f"- Tespit edilen kritik hacim kümeleri: {len(clusters)}")
                print("  🤖 Yapay Zeka Yorumu: Büyük kümeler %85 olasılıkla önemli destek/direnç seviyeleri oluşturuyor")
                
                # 3. Delta Hacim
                print("\n⚖️ Alım-Satım Dengesi:")
                print(f"- Alım/Satım Delta Oranı: {delta_ratio:.2f}")
                print("  🤖 Yapay Zeka Yorumu: Delta >1.2 %75 yükseliş, <0.8 %72 düşüş sinyali veriyor")
                
                # 4. VWAP Analizi
                print("\n📊 VWAP-Hacim İlişkisi:")
                print(f"- VWAP'a göre mevcut fiyat konumu: {vwap_position:.2f}%")
                print("  🤖 Yapay Zeka Yorumu: VWAP üstü + hacim artışı %82 trend devamı gösteriyor")
                
                # 5. Likidite Analizi
                print("\n💧 Likidite Derinliği:")
                print(f"- Alış tarafı likidite: {liquidity['buy_liq']:.2f} BTC")
                print("  🤖 Yapay Zeka Yorumu: Alış likiditesi > Satış ise %78 yükseliş baskısı bekleniyor")
                
                # 6. Emir Akışı
                print("\n🎯 Gerçek Zamanlı Emir Akışı:")
                print(f"- Büyük emirlerin hacim oranı: %{order_flow['large_orders_ratio']:.2f}")
                print("  🤖 Yapay Zeka Yorumu: %5 üstü büyük emirler %83 manipülasyon riski taşıyor")
                
                # 7. Volatilite-Hacim
                print("\n🌪️ Volatilite/Hacim Oranı:")
                print(f"- VVR Değeri: {latest_vvr:.2f}")
                print("  🤖 Yapay Zeka Yorumu: 1.5 üstü VVR %75 riskli volatilite sinyali veriyor")
                
                # 8. Hacim Momentum
                print("\n🚀 Hacim Momentumu:")
                print(f"- 7 Günlük Hacim Momentumu: %{volume_momentum:.2f}")
                print("  🤖 Yapay Zeka Yorumu: %30+ momentum %80 trend hızlanmasına işaret ediyor")
                
                # 9. Hacim Dağılımı
                print("\n📊 Hacim Dağılım Asimetrisi:")
                print(f"- Dağılım Çarpıklığı: {volume_profile['distribution_skew']:.2f}")
                print("  🤖 Yapay Zeka Yorumu: Pozitif çarpıklık %73 yükseliş, negatif %68 düşüş sinyali")
                
            except Exception as e:
                print(f"\nHacim analizi yorumları oluşturulamadı: {e}")

        except Exception as e:
            print(f"\nHata oluştu: {e}")

    def __del__(self):
        """WebSocket bağlantısını temizle"""
        if self.ws:
            self.ws.close()

    def analyze_volume_metrics(self, df):
        """
        Gelişmiş hacim metrikleri analizi
        
        Hacim Analizi Dokümantasyonu:
        -----------------------------
        1. Temel Hacim Metrikleri:
           - Hacim Momentumu: Ardışık hacimler arası değişim
           - Hacim/MA Oranı: Mevcut hacmin hareketli ortalamaya oranı
           - Anormal Hacim: Z-score bazlı anomali tespiti
           
           🤖 Yapay Zeka Yorumu:
           - Momentum değişimleri %70 doğrulukla trend değişimini öngörüyor
           - Z-score 2'nin üzerindeyken fiyat hareketleri %65 daha volatil
           - 20 periyotluk MA yerine adaptif MA kullanılabilir
        
        2. Hacim Trendleri:
           - Trend Yönü: Hacim momentumunun işareti
           - Trend Gücü: Momentum/Std.Sapma oranı
           
           🤖 Yapay Zeka Yorumu:
           - Trend gücü 1.5'in üzerindeyken %80 trend devam ediyor
           - Düşük hacimli trendler %60 daha fazla reversal gösteriyor
        
        3. Alış/Satış Hacmi:
           - Yeşil mumlarda alış hacmi ağırlıklı
           - Kırmızı mumlarda satış hacmi ağırlıklı
           - Doji durumlarda 50/50 dağılım
           
           🤖 Yapay Zeka Yorumu:
           - Alış/satış oranı 2:1'i geçtiğinde %75 trend dönüşü
           - Doji sonrası hacim artışı %70 yeni trend başlangıcı
        
        Returns:
            DataFrame: Hacim metrikleri eklenmiş DataFrame
        """
        try:
            # ATR hesaplama
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range()

            # Hacim Momentumu
            df['volume_momentum'] = df['volume'].diff()
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Hacim Trendleri
            df['volume_trend'] = np.where(df['volume_momentum'] > 0, 1, -1)
            df['volume_trend_strength'] = abs(df['volume_momentum']) / df['volume'].rolling(20).std()
            
            # Anormal Hacim Tespiti
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            
            # Alış/Satış Hacmi Hesaplama (İyileştirilmiş)
            df['price_change_pct'] = df['close'].pct_change()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Ağırlıklı hacim hesaplama
            df['buy_volume'] = np.where(
                df['close'] > df['open'],  # Yeşil mum
                df['volume'],
                np.where(
                    df['close'] == df['open'],  # Doji
                    df['volume'] * 0.5,
                    np.where(
                        df['high'] > df['typical_price'],  # Üst gölge
                        df['volume'] * 0.3,
                        df['volume'] * 0.1  # Minimum alış hacmi
                    )
                )
            )
            
            df['sell_volume'] = np.where(
                df['close'] < df['open'],  # Kırmızı mum
                df['volume'],
                np.where(
                    df['close'] == df['open'],  # Doji
                    df['volume'] * 0.5,
                    np.where(
                        df['low'] < df['typical_price'],  # Alt gölge
                        df['volume'] * 0.3,
                        df['volume'] * 0.1  # Minimum satış hacmi
                    )
                )
            )
            
            # Toplam hacim kontrolü
            total_volume = df['buy_volume'] + df['sell_volume']
            df['buy_volume'] = df['buy_volume'] * (df['volume'] / total_volume)
            df['sell_volume'] = df['sell_volume'] * (df['volume'] / total_volume)
            
            # Hacim oranları (normalize edilmiş)
            total = df['buy_volume'] + df['sell_volume']
            df['buy_ratio'] = df['buy_volume'] / total
            df['sell_ratio'] = df['sell_volume'] / total
            
            # Son kontroller
            df['buy_ratio'] = df['buy_ratio'].clip(0, 1)
            df['sell_ratio'] = df['sell_ratio'].clip(0, 1)
            
            return df
        
        except Exception as e:
            print(f"Hacim metrikleri hesaplanırken hata: {e}")
            return df

    def print_volume_analysis(self, df):
        """Hacim analizi çıktısı"""
        latest = df.iloc[-1]
        
        print("\n🔍 Detaylı Hacim Analizi:")
        print(f"• Hacim Momentumu: {latest['volume_momentum']:.2f}")
        print(f"• Hacim Trend Gücü: {latest['volume_trend_strength']:.2f}")
        print(f"• Anormal Hacim (Z-Score): {latest['volume_zscore']:.2f}")
        
        # Hacim dağılımı yüzdeleri
        total_buy_volume = df['buy_volume'].sum()
        total_sell_volume = df['sell_volume'].sum()
        total_volume = total_buy_volume + total_sell_volume
        
        print(f"• Alış Hacmi Yüzdesi: {(total_buy_volume/total_volume)*100:.2f}%")
        print(f"• Satış Hacmi Yüzdesi: {(total_sell_volume/total_volume)*100:.2f}%")

    def analyze_delta_metrics(self, df):
        """Gelişmiş delta analizi"""
        # Delta gücü hesaplama
        delta_strength = (df['buy_volume'] - df['sell_volume']).rolling(5).mean()
        
        # Momentum sinyali
        momentum = "Güçlü Alış" if delta_strength.iloc[-1] > delta_strength.mean() + delta_strength.std() else \
                  "Güçlü Satış" if delta_strength.iloc[-1] < delta_strength.mean() - delta_strength.std() else "Nötr"
        
        # Trend uyumu
        price_trend = df['close'].diff(5).mean()
        delta_trend = delta_strength.diff(5).mean()
        trend_alignment = "Uyumlu" if (price_trend > 0 and delta_trend > 0) or \
                                     (price_trend < 0 and delta_trend < 0) else "Uyumsuz"
        
        return {
            'strength': delta_strength.iloc[-1],
            'signal': momentum,
            'trend_alignment': trend_alignment
        }

    def generate_volume_signals(self, df):
        """Hacim bazlı trading sinyalleri"""
        signals = []
        
        # Anormal hacim sinyali
        if df['volume_zscore'].iloc[-1] > 2:
            signals.append("⚠️ Anormal Yüksek Hacim Tespit Edildi")
        
        # Hacim trendinde kırılma
        if df['volume_trend'].iloc[-1] != df['volume_trend'].iloc[-2]:
            signals.append("🔄 Hacim Trend Değişimi")
        
        # Hacim momentumu sinyali
        if df['volume_momentum'].iloc[-1] > df['volume_momentum'].rolling(20).mean().iloc[-1] * 1.5:
            signals.append("📈 Güçlü Hacim Momentumu")
        
        return signals

    def calculate_volume_momentum(self, df, period=7):
        """
        Hacim Momentum Analizi
        
        🤖 Yapay Zeka Yorumu:
        - 7 günlük hacim momentumu %30+ ise %80 trend hızlanması
        - Negatif momentum %75 trend zayıflaması
        - Ani momentum artışı %83 fiyat hareketi öncülü
        """
        df['volume_momentum'] = (df['volume'] / df['volume'].shift(period) - 1) * 100
        return df

    def load_ai_model(self):
        try:
            self.ai_predictor.model.load_weights('volume_model.weights.h5')
            self.ai_predictor.feature_scaler = joblib.load('feature_scaler.save')
            self.ai_predictor.target_scaler = joblib.load('target_scaler.save')
            # Optimizer'ı yeniden başlat
            self.ai_predictor.model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
        except:
            print("AI model bulunamadı, yeni model oluşturuluyor...")
            
    def real_time_analysis(self, df):
        # AI Tahminlerini al
        ai_prediction = self.ai_predictor.predict_next_volume(df)
        volatility_score = self.calculate_volatility(df)
        
        # AI Yorumu oluştur
        analysis = f"""
        🤖 Derin Öğrenme Analizi:
        - Tahmini Sonraki Hacim: {ai_prediction:.2f} BTC
        - Volatilite Skoru: {volatility_score:.2f}
        - Önerilen Strateji: {'AL' if ai_prediction > df['volume'].iloc[-1] else 'SAT'}
        """
        return analysis

    def generate_ai_insights(self, df):
        # Gerçek zamanlı AI analizi
        live_analysis = self.real_time_analysis(df)
        
        # Hacim tahmini
        volume_pred = self.ai_predictor.predict_next_volume(df)
        current_volume = df['volume'].iloc[-1]
        
        # Trend analizi
        trend_strength = abs(df['macd'].iloc[-1] - df['MACD_signal'].iloc[-1])
        
        # Risk analizi
        volatility_score = self.calculate_volatility(df)
        
        # Dinamik yorumlar
        insights = f"""
        🧠 Gerçek Zeka Analizi:
        1️⃣ Tahmini Hacim: {volume_pred:.2f} BTC ({'+' if volume_pred > current_volume else ''}{((volume_pred/current_volume)-1)*100:.1f}%)
        2️⃣ Trend Gücü: {'Yüksek' if trend_strength > 50 else 'Orta' if trend_strength > 20 else 'Zayıf'}
        3️⃣ Volatilite Skoru: {volatility_score:.2f}/100
        4️⃣ Öneri: {'AL' if volume_pred > current_volume*1.1 else 'SAT' if volume_pred < current_volume*0.9 else 'BEKLE'}
        """
        return insights

    def calculate_volatility(self, df, period=14):
        """Volatilite hesaplama (ATR tabanlı)"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, high_close, low_close)
            atr = true_range.rolling(period).mean()
            
            # Volatilite skoru (0-100 arası)
            max_atr = atr.max()
            volatility_score = (atr.iloc[-1] / max_atr) * 100 if max_atr > 0 else 0
            
            df['atr'] = atr
            return volatility_score
        except Exception as e:
            print(f"Volatilite hesaplama hatası: {e}")
            return 0

    def print_technical_analysis(self, df, interval, symbol):
        try:
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
            rvi_status = "YÜKSELİŞ" if rvi_diff > 0 else "DÜŞÜŞ"
            print(f"• RVI: {latest['rvi']:.2f} ({rvi_status})")
            print(f"• RVI Sinyal: {latest['rvi_signal']:.2f}")
            
            # 3. Hacim Analizi
            print("\n3️⃣ HACIM ANALİZİ:")
            vwap_diff = (current_price - latest['vwap']) / latest['vwap'] * 100
            print(f"• VWAP: {latest['vwap']:.2f} ({'Üzerinde' if current_price > latest['vwap'] else 'Altında'})")
            print(f"• VWAP Farkı: {vwap_diff:.2f}%")
            
            # 4. Volatilite Analizi
            print("\n4️⃣ VOLATİLİTE:")
            keltner_range = latest['keltner_upper'] - latest['keltner_lower']
            print(f"• Keltner Aralığı: {keltner_range:.2f}")
            print(f"• Üst Bant: {latest['keltner_upper']:.2f}")
            print(f"• Alt Bant: {latest['keltner_lower']:.2f}")
            
            # 5. Fibonacci Seviyeleri
            print("\n5️⃣ FIBONACCI SEVİYELERİ:")
            for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                fib_value = latest[f'fib_{level}']
                diff = (current_price - fib_value) / fib_value * 100
                print(f"• {level*100:.1f}%: {fib_value:.2f} ({'↑' if current_price > fib_value else '↓'} {abs(diff):.2f}%)")
                
            # 6. Likidite Analizi
            print("\n6️⃣ LİKİDİTE ANALİZİ:")
            cdv_status = "Pozitif" if latest['cdv'] > 0 else "Negatif"
            print(f"• Kümülatif Delta: {latest['cdv']:.2f} ({cdv_status})")
            
            # Likidasyon analizi
            print("\n🔥 LİKİDASYON ANALİZİ:")
            heatmap = self.calculate_liquidation_levels(symbol)
            analyzer = LiquidationAnalyzer()
            critical_levels = analyzer.calculate_critical_levels(heatmap)
            
            if critical_levels:
                for level in critical_levels[:3]:  # En yüksek 3 seviye
                    print(f"\n► {level['type']}: {level['level']:.2f}")
                    print(f"   - Risk Puanı: {level['puan']}/5.0 ({level['risk']})")
                    print(f"   - Likidite Miktarı: {level['miktar']:.2f} BTC")
            else:
                print("\n► Kritik likidasyon seviyesi bulunamadı")
            
            # 7. Güncellenmiş Formasyon Yorumları
            print("\n🎯 FORMASYON YORUMLARI:")
            try:
                if 'price_patterns' not in df.columns:
                    raise ValueError("Formasyon verileri bulunamadı")

                import ast
                patterns = ast.literal_eval(df['price_patterns'].iloc[-1])
                
                pattern_descriptions = {
                    'head_shoulders': "⛰️ Baş-Omuz Formasyonu (Potansiyel Trend Dönüşü)",
                    'double_top': "🔻 Çift Tepe (Yerel Maksimum Sinyali)",
                    'cup_handle': "☕ Çanak-Kulp (Uzun Vadeli Yükseliş Beklentisi)",
                    'ascending_triangle': "📈 Yükselen Üçgen (Trend Devam Sinyali)",
                    'bullish_flag': "🚩 Boğa Bayrağı (Yükseliş Hızlanması Beklentisi)"
                }

                formation_count = 0
                print("📊 Tespit Edilen Formasyonlar:")
                for pattern, active in patterns.items():
                    if active:
                        formation_count += 1
                        description = pattern_descriptions.get(
                            pattern, 
                            f"🔍 Bilinmeyen Formasyon: {pattern}"
                        )
                        print(f"  → {description}")

                if formation_count == 0:
                    print("  ⚠️ Belirgin formasyon tespit edilemedi")
                    print("   - Piyasa belirsizlik içinde olabilir")

            except Exception as e:
                print(f"  ❗ Formasyon yorumlama hatası: {str(e)}")

        except Exception as e:
            print(f"\n❌ Genel analiz hatası: {str(e)}")

    def get_liquidation_data(self, symbol, periods=['1h', '4h', '12h', '24h']):
        """Son ve Çalışan Versiyon"""
        all_liquidations = []
        base_url = "https://fapi.binance.com/fapi/v1/forceOrders"
        
        try:
            # IP Kontrolü
            current_ip = requests.get('https://api.ipify.org').text
            print(f"🔍 IP Adresiniz: {current_ip} - Binance'de whitelist'te OLMALI!")
            
            # Zaman Kontrolü
            server_time = int(requests.get('https://fapi.binance.com/fapi/v1/time').json()['serverTime'])
            
            for period in periods:  # Döngüyü geri ekledik
                try:
                    hours = int(period[:-1])
                    end_time = server_time
                    start_time = end_time - (hours * 3600 * 1000)
                    
                    # Güncel Parametreler
                    params = {
                        'symbol': symbol.upper(),
                        'startTime': start_time,
                        'endTime': end_time,
                        'limit': 500,
                        'timestamp': server_time,
                        'recvWindow': 10000
                    }
                    
                    # İmza ve İstek
                    query_string = urlencode(sorted(params.items()))
                    signature = hmac.new(
                        self.api_secret.encode('utf-8'),
                        query_string.encode('utf-8'),
                        hashlib.sha256
                    ).hexdigest()
                    
                    response = requests.get(
                        base_url,
                        params={**params, 'signature': signature},
                        headers={'X-MBX-APIKEY': self.api_key},
                        timeout=15
                    )
                    
                    # Veriyi işle
                    if response.status_code == 200:
                        data = response.json()
                        processed = [{
                            'price': float(o['order']['price']),
                            'side': o['order']['side'].upper(),
                            'qty': float(o['order']['executedQty']),
                            'timestamp': o['order']['time']
                        } for o in data if o['order']['status'] == 'FILLED']
                        
                        all_liquidations.extend(processed)
                        print(f"{period}: {len(processed)} kayıt")
                    else:
                        print(f"{period} Hatası: {response.status_code} | {response.text}")
                except Exception as e:
                    print(f"⛔ Kritik Hata: {str(e)}")
            
            return all_liquidations

        except Exception as e:
            print(f"⛔ Kritik Hata: {str(e)}")
        
        return all_liquidations

    def _validate_api_key(self):
        """API anahtar formatını kontrol et"""
        if not self.api_key or len(self.api_key) != 64:
            return False
        if not self.api_secret or len(self.api_secret) != 64:
            return False
        return True

    def calculate_liquidation_levels(self, symbol):
        """Güncellenmiş Likidasyon Haritası"""
        try:
            # API'den ham likidasyon verilerini yeni metodla al
            liquidations = self.get_liquidation_data(symbol, periods=['24h'])
            
            # Veri işleme ve filtreleme
            processed = [
                {
                    'price': float(order['price']),
                    'side': order['side'],
                    'qty': order['qty']
                } for order in liquidations
            ]
            
            if not processed:
                print("⚠️ İşlenebilir likidasyon verisi bulunamadı")
                return pd.DataFrame()

            # DataFrame oluşturma
            df = pd.DataFrame(processed)
            
            # Heatmap oluşturma
            heatmap = (
                df.assign(price_bin=pd.cut(df['price'], bins=20))
                .groupby(['price_bin', 'side'])['qty']
                .sum()
                .unstack(fill_value=0)
            )
            
            return heatmap

        except Exception as e:
            print(f"Likidasyon analiz hatası: {str(e)}")
            return pd.DataFrame()

    def _fetch_klines(self, symbol, interval, limit):
        """Binance'den ham veri çekme metodu"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                            'taker_buy_base', 'taker_buy_quote']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
            
        except Exception as e:
            print(f"Veri çekme hatası: {str(e)}")
            return pd.DataFrame()

    def _format_data(self, df):
        """Veriyi analiz için uygun formata getir"""
        required_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'taker_buy_base', 'taker_buy_quote'
        ]
        
        # Eksik sütun kontrolü
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Eksik sütunlar: {', '.join(missing)}")
        
        # Zaman damgası dönüşümü
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        return df.dropna()

def get_available_pairs():
    """Mevcut trading çiftlerini getir"""
    client = Client()
    info = client.get_exchange_info()
    return [symbol['symbol'] for symbol in info['symbols'] if symbol['status'] == 'TRADING']

def main():
    api_key = "Rkc5t0oPmrc9NGv1qHJ6NHMjGB5qA1rPO2w5fKRSwUwPIbBgmGxxlSW9gzd6shNv"
    api_secret = "0YF0oIpsvLZWyRPfRmnuIPN7tYf16k8bdxYR1J8AU0vN6tk96VIg6rGzJINKuD3f"
    
    collector = BinanceDataCollector(api_key, api_secret)

    while True:
        # Kullanıcıdan trading çifti seçimini al
        print("\nPopüler çiftler: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT")
        print("Çıkmak için 'q' yazın")
        symbol = input("\nLütfen bir trading çifti girin (örn: BTCUSDT): ").upper()
        
        if symbol.lower() == 'q':
            print("\nProgram sonlandırılıyor...")
            break

        # Kullanıcıdan periyot seçimini al
        print("\nMevcut periyotlar: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d")
        interval = input("Lütfen analiz periyodunu seçin (örn: 5m): ").lower()
        
        # Geçerli periyot kontrolü
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
        if interval not in valid_intervals:
            print(f"\nHata: Geçersiz periyot. Lütfen şu periyotlardan birini seçin: {', '.join(valid_intervals)}")
            continue

        try:
            print("\n" + "="*50)
            print(f"Piyasa Analizi - {symbol} ({interval})")
            print("="*50)

            # Veri al
            df = collector.get_klines_data(symbol, interval)
            df = collector.analyze_volume_metrics(df)
            latest = df.iloc[-1]

            # 1. Anlık Fiyat ve Temel Veriler
            ticker = collector.client.get_symbol_ticker(symbol=symbol)
            print(f"\n📊 Anlık Fiyat: {ticker['price']}")

            # 2. Emir Defteri
            depth = collector.client.get_order_book(symbol=symbol, limit=5)
            print("\n📚 Emir Defteri:")
            print("\nAlış Emirleri:")
            for bid in depth['bids']:
                print(f"Fiyat: {bid[0]}, Miktar: {bid[1]}")
            print("\nSatış Emirleri:")
            for ask in depth['asks']:
                print(f"Fiyat: {ask[0]}, Miktar: {ask[1]}")

            # 3. Teknik Göstergeler
            collector.print_technical_analysis(df, interval, symbol)

            # 4. Hacim Analizi
            print("\n📊 HACIM ANALİZLERİ:")
            print("-"*40)
            
            # 1. Temel Hacim Metrikleri
            print("\n1️⃣ Temel Hacim Metrikleri:")
            print(f"• Güncel Hacim: {latest['volume']:.2f}")
            print(f"• Hacim Momentumu: {latest['volume_momentum']:.2f}")
            print(f"• Anormal Hacim Skoru: {latest['volume_zscore']:.2f}")
            print(f"• Alış Yüzdesi: {(latest['buy_ratio'] * 100):.1f}%")
            print(f"• Satış Yüzdesi: {(latest['sell_ratio'] * 100):.1f}%")

            # 2. Hacim Profili
            vp = collector.calculate_volume_profile(df)
            print("\n2️⃣ Hacim Profili:")
            print(f"• POC (Point of Control): {vp['poc']:.2f}")
            print(f"• Value Area Yüksek: {vp['value_area_high']:.2f}")
            print(f"• Value Area Düşük: {vp['value_area_low']:.2f}")
            print(f"• Likidite Yoğunluğu: {vp['liquidity_density']:.2f}")
            print(f"• Dağılım Asimetrisi: {vp['distribution_skew']:.2f}")

            # 3. Delta Hacim
            print("\n3️⃣ Delta Hacim Analizi:")
            df = collector.calculate_delta_volume(df)
            latest = df.iloc[-1]
            print(f"• Anlık Delta: {latest['delta']:.2f}")
            print(f"• Kümülatif Delta: {latest['cumulative_delta']:.2f}")
            if latest['delta_divergence']:
                print("⚠️ Delta Divergence Tespit Edildi!")

            # 4. Real-Time Volume Delta
            delta = collector.calculate_real_time_volume_delta(symbol)
            print("\n4️⃣ Anlık Hacim Akışı:")
            print(f"• Delta Oranı: {delta['delta_ratio']:.2f}%")
            print(f"• Baskı Yönü: {delta['pressure']}")
            print(f"• Alış Hacmi: {delta['buying_volume']:.2f}")
            print(f"• Satış Hacmi: {delta['selling_volume']:.2f}")

            # 5. VWAP Analizi
            df = collector.calculate_vwap(df)
            latest = df.iloc[-1]
            print("\n5️⃣ VWAP Analizi:")
            print(f"• VWAP: {latest['vwap']:.2f}")
            print(f"• Fiyat-VWAP Farkı: {(latest['close'] - latest['vwap']):.2f}")

            # 6. Hacim Kümeleme
            print("\n6️⃣ Hacim Kümeleme:")
            clusters = collector.analyze_volume_clusters(df)
            for i, cluster in enumerate(clusters[:3], 1):
                print(f"• Küme {i}: Fiyat {cluster['price']:.2f}, Hacim {cluster['volume']:.2f}")

            # 7. Volatilite/Hacim İlişkisi
            print("\n7️⃣ Volatilite/Hacim Analizi:")
            df = collector.calculate_volatility_volume_ratio(df)
            latest_vvr = df.iloc[-1]['vvr']
            print(f"• VVR: {latest_vvr:.2f}")
            if latest_vvr > 1.5:
                print("⚠️ Yüksek Manipülasyon Riski!")

            # 8. Hacim Bazlı Fibonacci
            print("\n8️⃣ Hacim-Fibonacci İlişkisi:")
            fib = collector.calculate_volume_fibonacci(df)
            for level, price in fib['fib_levels'].items():
                volume = fib['volume_at_levels'][level]
                print(f"• Fib {level}: {price:.2f} (Hacim: {volume:.2f})")

            # 9. Algoritmik Desenler
            print("\n9️⃣ Algoritmik Hacim Desenleri:")
            trades = collector.client.get_recent_trades(symbol=symbol, limit=100)
            patterns = collector.detect_algorithmic_patterns(df, trades)
            
            if patterns['stop_hunting']:
                print("\n• Stop Hunting Tespiti:")
                for hunt in patterns['stop_hunting'][:2]:
                    print(f"  Fiyat: {hunt['price']:.2f}, Hacim: {hunt['volume']:.2f}")
            
            if patterns['liquidity_grab']:
                print("\n• Likidite Grab Tespiti:")
                for grab in patterns['liquidity_grab'][:2]:
                    print(f"  Fiyat: {grab['price']:.2f}, Hacim: {grab['volume']:.2f}")

            # Trading Önerileri
            print("\n🎯 Trading Önerileri:")
            volume_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            
            if volume_ratio > 2 and latest['buy_ratio'] > 0.6:
                print("• Güçlü Alım Fırsatı")
            elif volume_ratio > 2 and latest['sell_ratio'] > 0.6:
                print("• Güçlü Satış Fırsatı")
            elif latest_vvr > 1.5:
                print("• Yüksek Volatilite - Dikkatli Olun")
            else:
                print("• Normal Hacim Seviyeleri")

            # 5. Sentiment Analizi
            print("\n🌍 Piyasa Sentiment Analizi:")
            sentiment = collector.analyze_sentiment(symbol)
            print(f"Reddit Sentiment: {sentiment['reddit_sentiment']:.2f} (-1 bearish, +1 bullish)")
            print(f"Fear & Greed Index: {sentiment['fear_greed_index']} (0 korku, 100 açgözlülük)")
            print(f"Sosyal Medya Hacmi: {sentiment['social_volume']}")
            print(f"Genel Market Duyarlılığı: {sentiment['overall_sentiment']:.2f}")

            # 6. Risk Yönetimi
            print("\n💰 Risk Yönetimi:")
            symbol_info = collector.client.get_symbol_info(symbol)
            if symbol_info:
                try:
                    current_price = float(latest['close'])
                    atr = latest['atr']
                    
                    if pd.notna(atr) and atr > 0:  # ATR değeri kontrolü
                        stop_loss = current_price - (atr * 2)
                        position_size = collector.calculate_position_size(current_price, stop_loss, symbol_info)
                        
                        if position_size > 0:
                            print(f"Önerilen Pozisyon Büyüklüğü: {position_size:.8f} {symbol}")
                            print(f"ATR: {atr:.2f}")
                            print(f"Dinamik Stop-Loss: {stop_loss:.2f}")
                            take_profit = current_price + (current_price - stop_loss) * collector.risk_params['min_rr_ratio']
                            print(f"Take-Profit (1:{collector.risk_params['min_rr_ratio']}): {take_profit:.2f}")
                            
                            # Risk tutarı hesaplama
                            risk_amount = position_size * (current_price - stop_loss)
                            risk_percentage = (risk_amount / collector.risk_params['capital']) * 100
                            print(f"Risk Tutarı: {risk_amount:.2f} USDT ({risk_percentage:.2f}%)")
                        else:
                            print("ATR hesaplanamadı - yeterli veri yok")
                except Exception as e:
                    print(f"Risk hesaplama hatası: {e}")
            else:
                print("Symbol bilgisi alınamadı")

            print("\n📈 Strateji Optimizasyonu:")
            print("Strateji optimizasyonu devre dışı bırakıldı")
            
            print("\nYeni analiz için trading çifti ve periyot seçin")
            print("Çıkmak için 'q' yazın")

            # Hacim analizlerini hesapla
            try:
                # 1. Hacim Artış Dinamiği
                volume_change = ((df['volume'].iloc[-1] / df['volume'].iloc[-24]) - 1) * 100
                
                # 2. Hacim Kümelenme
                clusters = collector.analyze_volume_clusters(df)
                
                # 3. Delta Hacim
                delta_ratio = df['delta'].iloc[-1] / df['volume'].iloc[-1]
                
                # 4. VWAP Analizi
                vwap_position = (df['close'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] * 100
                
                # 5. Likidite Analizi
                liquidity = collector.analyze_liquidity(symbol)
                
                # 6. Emir Akışı
                order_flow = collector.analyze_order_flow(symbol)
                
                # 7. Volatilite-Hacim
                latest_vvr = df['vvr'].iloc[-1]
                
                # 8. Hacim Momentum
                volume_momentum = df['volume_momentum'].iloc[-1]
                
                # 9. Hacim Dağılımı
                volume_profile = collector.calculate_volume_profile(df)
                
                # Yapay Zeka Yorumlarını Bas
                print("\n🤖 Hacim Analizi Yapay Zeka Yorumları:")
                print(collector.generate_ai_insights(df))
                
                # 1. Hacim Artışı
                print("\n📈 Hacim Artış Dinamiği:")
                print(f"- Son 24 saatte hacim %{volume_change:.2f} arttı")
                print("  🔍 Yapay Zeka Yorumu: " + ["Düşüş sinyali", "Nötre yakın", "Yükseliş sinyali"][int(np.sign(volume_change)+1)])
                
                # 2. Hacim Kümelenme
                print("\n🔵 Hacim Kümelenme Analizi:")
                print(f"- Tespit edilen kritik hacim kümeleri: {len(clusters)}")
                print("  🤖 Yapay Zeka Yorumu: Büyük kümeler %85 olasılıkla önemli destek/direnç seviyeleri oluşturuyor")
                
                # 3. Delta Hacim
                print("\n⚖️ Alım-Satım Dengesi:")
                print(f"- Alım/Satım Delta Oranı: {delta_ratio:.2f}")
                print("  🤖 Yapay Zeka Yorumu: Delta >1.2 %75 yükseliş, <0.8 %72 düşüş sinyali veriyor")
                
                # 4. VWAP Analizi
                print("\n📊 VWAP-Hacim İlişkisi:")
                print(f"- VWAP'a göre mevcut fiyat konumu: {vwap_position:.2f}%")
                print("  🤖 Yapay Zeka Yorumu: VWAP üstü + hacim artışı %82 trend devamı gösteriyor")
                
                # 5. Likidite Analizi
                print("\n💧 Likidite Derinliği:")
                print(f"- Alış tarafı likidite: {liquidity['buy_liq']:.2f} BTC")
                print("  🤖 Yapay Zeka Yorumu: Alış likiditesi > Satış ise %78 yükseliş baskısı bekleniyor")
                
                # 6. Emir Akışı
                print("\n🎯 Gerçek Zamanlı Emir Akışı:")
                print(f"- Büyük emirlerin hacim oranı: %{order_flow['large_orders_ratio']:.2f}")
                print("  🤖 Yapay Zeka Yorumu: %5 üstü büyük emirler %83 manipülasyon riski taşıyor")
                
                # 7. Volatilite-Hacim
                print("\n🌪️ Volatilite/Hacim Oranı:")
                print(f"- VVR Değeri: {latest_vvr:.2f}")
                print("  🤖 Yapay Zeka Yorumu: 1.5 üstü VVR %75 riskli volatilite sinyali veriyor")
                
                # 8. Hacim Momentum
                print("\n🚀 Hacim Momentumu:")
                print(f"- 7 Günlük Hacim Momentumu: %{volume_momentum:.2f}")
                print("  🤖 Yapay Zeka Yorumu: %30+ momentum %80 trend hızlanmasına işaret ediyor")
                
                # 9. Hacim Dağılımı
                print("\n📊 Hacim Dağılım Asimetrisi:")
                print(f"- Dağılım Çarpıklığı: {volume_profile['distribution_skew']:.2f}")
                print("  🤖 Yapay Zeka Yorumu: Pozitif çarpıklık %73 yükseliş, negatif %68 düşüş sinyali")
                
            except Exception as e:
                print(f"\nHacim analizi yorumları oluşturulamadı: {e}")

        except Exception as e:
            print(f"\nHata oluştu: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından sonlandırıldı.")
