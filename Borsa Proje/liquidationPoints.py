import pandas as pd
import numpy as np
from scipy.stats import zscore

class LiquidationAnalyzer:
    def __init__(self):
        self.risk_levels = {
            'low': 0.3,
            'moderate': 0.7,
            'high': 1.2,
            'extreme': 2.0
        }

    def calculate_liquidation_score(self, heatmap, current_price):
        """Likidasyon seviyelerini 5 üzerinden puanla"""
        try:
            # Boş veri ve geçersiz fiyat kontrolü
            if heatmap.empty or current_price <= 0 or not isinstance(heatmap.index, pd.IntervalIndex):
                return 0.0
                
            # 1. Yoğunluk Dağılım Puanı (0-1.5 arası)
            buy_density = heatmap['BUY'].sum() / (heatmap['BUY'].sum() + heatmap['SELL'].sum() + 1e-10)
            density_score = 1.5 * abs(buy_density - 0.5) * 2  # 0-1.5 arası
            
            # 2. Fiyat Yakınlık Puanı (0-1 arası)
            price_levels = heatmap.index.mid
            distances = np.abs(price_levels - current_price)
            proximity_score = 1 - (np.min(distances) / (current_price * 0.01 + 1e-10))  # %1'den fazla uzaklıkta 0
            
            # 3. Hacim Oranı (0-1 arası)
            max_buy = heatmap['BUY'].max()
            max_sell = heatmap['SELL'].max() 
            volume_ratio = min(max_buy/(max_sell+1e-10), max_sell/(max_buy+1e-10))
            
            # 4. Tarihsel Tutarlılık (0-0.5 arası)
            consistency = heatmap.rolling(3, min_periods=1).std().mean().mean()
            consistency_score = 0.5 * (1 - np.tanh(consistency*10)) 
            
            # 5. Risk Çarpanı (0-1 arası)
            risk_multiplier = np.log1p(heatmap.sum().sum()/100)  # Toplam likidasyon hacmi
            
            # Toplam Puan (0-5 arası)
            total_score = (density_score + proximity_score + volume_ratio + consistency_score) * risk_multiplier
            return min(5, max(0, total_score))
            
        except Exception as e:
            print(f"Puanlama hatası: {str(e)}")
            return 0.0

    def _calculate_level_strength(self, heatmap, side):
        """Seviye gücünü hesapla (Z-Score tabanlı)"""
        quantities = heatmap[side].fillna(0)
        if quantities.nunique() < 2:
            return 0.0
        z_scores = zscore(quantities)
        return np.mean(z_scores[-3:])

    def _calculate_distance_score(self, heatmap, current_price):
        """Fiyatın likidasyon seviyelerine yakınlık puanı"""
        price_levels = heatmap.index.mid
        distances = np.abs(price_levels - current_price)
        if len(distances) == 0:
            return 0.0
        normalized_dist = 1 - (distances / distances.max())
        return np.mean(normalized_dist[:3]) * 2

    def _calculate_volume_distribution(self, heatmap):
        """Hacim dağılım dengesi puanı"""
        buy_vol = heatmap['BUY'].sum()
        sell_vol = heatmap['SELL'].sum()
        if buy_vol + sell_vol == 0:
            return 0.0
        ratio = min(buy_vol, sell_vol) / max(buy_vol, sell_vol)
        return ratio * 2

    def _calculate_historical_consistency(self, heatmap):
        """Tarihsel tutarlılık puanı"""
        if len(heatmap) < 5:
            return 0.0
        rolling_corr = heatmap.rolling(5).corr().mean().mean()
        return max(0, rolling_corr * 2.5)

    def get_risk_assessment(self, score):
        """Puanı risk seviyesine çevir"""
        if score >= 4.5: return "ÇOK YÜKSEK RISK ⚠️⚡"
        if score >= 3.5: return "YÜKSEK RISK ⚠️"
        if score >= 2.5: return "ORTA RISK ◼️◼️◼️"
        if score >= 1.5: return "DÜŞÜK RISK ◼️◼️"
        return "ÇOK DÜŞÜK RISK ◼️"

    def calculate_levels(self, df, heatmap):
        """Ham veriyi işleyip analiz edilebilir seviyeler üret"""
        try:
            if heatmap.empty or df.empty:
                return {'support': 0.0, 'resistance': 0.0}

            current_price = df['close'].iloc[-1]
            buy_levels = heatmap['BUY'].nlargest(3)
            sell_levels = heatmap['SELL'].nlargest(3)

            return {
                'support': self.find_nearest_level(current_price, buy_levels),
                'resistance': self.find_nearest_level(current_price, sell_levels)
            }
        except Exception as e:
            print(f"Likitide analiz hatası: {str(e)}")
            return {'support': 0.0, 'resistance': 0.0}

    def find_nearest_level(self, price, levels):
        """Fiyata en yakın seviyeyi bul"""
        try:
            return levels.index[np.abs(levels.index.mid - price).argmin()]
        except:
            return 0.0 

    def calculate_risk_score(self, heatmap):
        """Likidasyon verilerini risk skoruna çevir"""
        if heatmap.empty:
            return 0.0
        buy_power = heatmap['BUY'].sum()
        sell_power = heatmap['SELL'].sum()
        return min(5, (abs(buy_power - sell_power) / (buy_power + sell_power + 1e-7)) * 5)

    def generate_recommendation(self, df, heatmap):
        """Fiyat ve likidasyona göre öneri üret (Güncellenmiş)"""
        try:
            current_price = df['close'].iloc[-1]
            if heatmap.empty or heatmap['SELL'].sum() == 0 or heatmap['BUY'].sum() == 0:
                return "Veri yetersiz"
            
            # Direnç ve desteği güvenli şekilde al
            resistance = heatmap['SELL'].idxmax().mid if not heatmap['SELL'].empty else current_price
            support = heatmap['BUY'].idxmax().mid if not heatmap['BUY'].empty else current_price
            
            # Fiyat kontrolü
            if current_price > resistance * 1.005:
                return "GÜÇLÜ AL ⬆️ (Direnç Aşıldı)"
            elif current_price < support * 0.995:
                return "ACİL SAT ⚠️ (Destek Kırıldı)"
            return "NÖTR ◻️ (Bölge içinde)"
        except Exception as e:
            print(f"Öneri oluşturma hatası: {str(e)}")
            return "Analiz hatası"

    def calculate_critical_levels(self, heatmap):
        """Likidasyon seviyelerini 5 üzerinden puanla (Son düzeltme)"""
        try:
            # Gelişmiş veri validasyonu
            if (not isinstance(heatmap.index, pd.IntervalIndex) or 
                heatmap.empty or 
                heatmap.shape[0] < 3):
                return []
            
            # Veri normalizasyonu
            heatmap = heatmap.apply(lambda x: pd.to_numeric(x, errors='coerce')
                                  .fillna(0).clip(lower=0))
            
            # Seviye belirleme optimizasyonu
            critical_levels = []
            for side in ['BUY', 'SELL']:
                if side not in heatmap: continue
                
                # Z-Score tabanlı anomali tespiti
                z_scores = zscore(heatmap[side])
                significant = heatmap[side][z_scores > 1.5]
                
                if not significant.empty:
                    max_val = significant.max()
                    for interval, val in significant.nlargest(3).items():
                        score = min(5, (val / (max_val / 5)) if max_val > 0 else 0)
                        critical_levels.append({
                            'type': 'DESTEK' if side == 'BUY' else 'DİRENÇ',
                            'level': interval.mid,
                            'miktar': val,
                            'puan': round(score, 1),
                            'risk': self._get_risk_label(score)
                        })
            
            return sorted(critical_levels, key=lambda x: (-x['puan'], x['level']))
            
        except Exception as e:
            print(f"Kritik seviye analiz hatası: {str(e)}")
            return []

    def _get_risk_label(self, score):
        """Puanı risk etiketine çevir"""
        if score >= 4.5: return "ÇOK YÜKSEK ⚠️🔥"
        if score >= 3.5: return "YÜKSEK ⚠️"
        if score >= 2.5: return "ORTA ◼️◼️◼️"
        if score >= 1.5: return "DÜŞÜK ◼️◼️"
        return "ÇOK DÜŞÜK ◼️" 