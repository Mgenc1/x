import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

class FormasyonAnaliz:
    def __init__(self, df, params=None):
        """
        Gelişmiş Formasyon Analiz Sınıfı
        
        Args:
            df (pd.DataFrame): OHLC verilerini içeren DataFrame
            params (dict): Formasyon parametreleri (opsiyonel)
        """
        self.df = df.copy()
        self.df['SMA_20'] = self.df['close'].rolling(20).mean()
        self.formasyonlar = []
        self.params = params or {
            'tepe_dip_pencere': 5,
            'formasyon_fark_limit': 0.03,
            'min_derinlik': 0.05,
            'ucgen_egim_limit': 0.005,
            'kama_egim_limit': 0.01,
            'hacim_artis_oran': 1.2,
            'formasyon_pencere': 30,
            'kulp_pencere_oran': 0.3,
            'max_derinlik': 0.2,
            'kama_penceresi': 20
        }

    def _tepe_dip_bul(self, pencere=5):
        """Gelişmiş tepe/dip tespiti"""
        tepeler, dipler = [], []
        for i in range(pencere, len(self.df)-pencere):
            if (self.df['high'].iloc[i] == self.df['high'].iloc[i-pencere:i+pencere+1].max()):
                tepeler.append(i)
            if (self.df['low'].iloc[i] == self.df['low'].iloc[i-pencere:i+pencere+1].min()):
                dipler.append(i)
        return tepeler, dipler

    def _egim_hesapla(self, seri):
        """Lineer regresyonla eğim hesaplama"""
        return linregress(np.arange(len(seri)), seri).slope

    def omuz_bas_omuz(self):
        """Omuz-Baş-Omuz formasyonu"""
        tepeler, _ = self._tepe_dip_bul()
        for i in range(1, len(tepeler)-1):
            left, head, right = tepeler[i-1], tepeler[i], tepeler[i+1]
            if (self.df['high'].iloc[left] < self.df['high'].iloc[head] and 
                self.df['high'].iloc[right] < self.df['high'].iloc[head]):
                self.formasyonlar.append(('Omuz Baş Omuz', head, None))

    def ters_omuz_bas_omuz(self):
        """Ters Omuz-Baş-Omuz formasyonu"""
        _, dipler = self._tepe_dip_bul()
        for i in range(1, len(dipler)-1):
            left, head, right = dipler[i-1], dipler[i], dipler[i+1]
            if (self.df['low'].iloc[left] > self.df['low'].iloc[head] and 
                self.df['low'].iloc[right] > self.df['low'].iloc[head]):
                self.formasyonlar.append(('Ters Omuz Baş Omuz', head, None))

    def ikili_tepe_dip(self):
        """İkili Tepe/Çift Dip formasyonları"""
        tepeler, dipler = self._tepe_dip_bul()
        # İkili Tepe
        for i in range(len(tepeler)-1):
            if abs(self.df['high'].iloc[tepeler[i]] - self.df['high'].iloc[tepeler[i+1]]) < self.params['formasyon_fark_limit']:
                self.formasyonlar.append(('İkili Tepe', tepeler[i+1], None))
        # Çift Dip
        for i in range(len(dipler)-1):
            if abs(self.df['low'].iloc[dipler[i]] - self.df['low'].iloc[dipler[i+1]]) < self.params['formasyon_fark_limit']:
                self.formasyonlar.append(('Çift Dip', dipler[i+1], None))

    def ucgen_formasyonlari(self):
        """Üçgen formasyonları tespiti"""
        for i in range(100, len(self.df)):
            bolge = self.df.iloc[i-30:i]
            ust_egim = self._egim_hesapla(bolge['high'])
            alt_egim = self._egim_hesapla(bolge['low'])
            
            # Yükselen Üçgen
            if ust_egim < self.params['ucgen_egim_limit'] and alt_egim > 0.01:
                self.formasyonlar.append(('Yükselen Üçgen', i, None))
            
            # Alçalan Üçgen
            elif ust_egim < -0.01 and alt_egim > -self.params['ucgen_egim_limit']:
                self.formasyonlar.append(('Alçalan Üçgen', i, None))
            
            # Simetrik Üçgen
            elif abs(ust_egim + alt_egim) < 0.005:
                self.formasyonlar.append(('Simetrik Üçgen', i, None))

    def kama_formasyonlari(self):
        """Kama formasyonları tespiti"""
        for i in range(100, len(self.df)):
            bolge = self.df.iloc[i-20:i]
            ust_egim = self._egim_hesapla(bolge['high'])
            alt_egim = self._egim_hesapla(bolge['low'])
            
            # Alçalan Kama
            if ust_egim < -self.params['kama_egim_limit'] and alt_egim < -self.params['kama_egim_limit']:
                self.formasyonlar.append(('Alçalan Kama', i, None))
            
            # Yükselen Kama
            elif ust_egim > self.params['kama_egim_limit'] and alt_egim > self.params['kama_egim_limit']:
                self.formasyonlar.append(('Yükselen Kama', i, None))

    def bayrak_flama(self):
        """Bayrak ve Flama formasyonları"""
        for i in range(50, len(self.df)):
            # Bayrak için hacim kontrolü
            hacim_artis = self.df['volume'].iloc[i-5:i].mean() > self.df['volume'].iloc[i-10:i-5].mean() * self.params['hacim_artis_oran']
            
            if hacim_artis:
                # Yükselen Bayrak
                if self.df['close'].iloc[i] > self.df['close'].iloc[i-5]:
                    self.formasyonlar.append(('Yükselen Bayrak', i, None))
                # Alçalan Bayrak
                else:
                    self.formasyonlar.append(('Alçalan Bayrak', i, None))
                
                # Flama (Daralan aralık)
                if (self.df['high'].iloc[i] - self.df['low'].iloc[i]) < (self.df['high'].iloc[i-5] - self.df['low'].iloc[i-5])*0.7:
                    self.formasyonlar.append(('Flama', i, None))

    def tum_formasyonlari_tara(self):
        """Tüm formasyon tarama işlemi"""
        self.omuz_bas_omuz()
        self.ters_omuz_bas_omuz()
        self.ikili_tepe_dip()
        self.ucgen_formasyonlari()
        self.kama_formasyonlari()
        self.bayrak_flama()
        return self

    def rapor_olustur(self):
        """Formasyon raporu oluşturma"""
        rapor = {
            'Trend Dönüş': [],
            'Trend Devam': [],
            'Uyarılar': []
        }
        
        formasyon_tipleri = {
            'Omuz Baş Omuz': 'Trend Dönüş',
            'Ters Omuz Baş Omuz': 'Trend Dönüş',
            'İkili Tepe': 'Trend Dönüş',
            'Çift Dip': 'Trend Dönüş',
            'Yükselen Üçgen': 'Trend Devam',
            'Alçalan Üçgen': 'Trend Devam',
            'Simetrik Üçgen': 'Trend Devam',
            'Alçalan Kama': 'Trend Devam',
            'Yükselen Kama': 'Trend Devam',
            'Yükselen Bayrak': 'Trend Devam',
            'Alçalan Bayrak': 'Trend Devam',
            'Flama': 'Trend Devam'
        }
        
        for formasyon, pozisyon, _ in self.formasyonlar:
            tip = formasyon_tipleri.get(formasyon, 'Bilinmeyen')
            detay = {
                'Formasyon': formasyon,
                'Pozisyon': self.df.index[pozisyon],
                'Fiyat': self.df['close'].iloc[pozisyon],
                'Güvenilirlik': self._guvenilirlik_hesapla(formasyon, pozisyon)
            }
            if tip in rapor:
                rapor[tip].append(detay)
            else:
                rapor['Uyarılar'].append(detay)
        
        return rapor

    def _guvenilirlik_hesapla(self, formasyon, pozisyon):
        """Formasyon güvenilirlik skoru"""
        skor = 0
        if formasyon in ['Omuz Baş Omuz', 'Ters Omuz Baş Omuz']:
            skor = min(90, len(self.formasyonlar)*10)
        elif 'Üçgen' in formasyon:
            skor = 80
        elif 'Kama' in formasyon:
            skor = 75
        elif 'Bayrak' in formasyon or 'Flama' in formasyon:
            skor = 70
        return f"%{skor}"

    def detect_all_patterns(self):
        """Tüm Formasyonları Tespit Et"""
        # Temel formasyonlar
        super().detect_all_patterns()
        
        # Yeni formasyonlar
        self.tum_formasyonlari_tara()
        
        return self.df

    def visualize_patterns(self):
        """Gelişmiş Görselleştirme"""
        plt.figure(figsize=(18,8))
        plt.plot(self.df['close'], label='Fiyat', alpha=0.5)
        plt.plot(self.df['SMA_20'], label='20 SMA', linestyle='--')
        
        renkler = {
            'Çanak': 'green',
            'Fincan-Kulp': 'darkgreen',
            'Yükselen Üçgen': 'purple',
            'Omuz Baş Omuz': 'red',
            'double_top': 'orange',
            'double_bottom': 'blue'
        }
        
        for f in self.formasyonlar:
            plt.scatter(
                self.df.index[f[1]],
                f[0],
                color=renkler.get(f[0], 'gray'),
                s=100,
                label=f[0]
            )
        
        plt.title('Gelişmiş Formasyon Analizi')
        plt.legend()
        plt.show()

    def get_detailed_analysis(self):
        """İstatistiksel Analiz Raporu"""
        if 'price_patterns' not in self.df.columns:
            self.detect_all_patterns()
            
        patterns = eval(self.df['price_patterns'].iloc[-1])
        active_patterns = [k for k,v in patterns.items() if v]
        
        return {
            'total_patterns': len(active_patterns),
            'active_patterns': active_patterns,
            'bearish_ratio': sum(1 for p in active_patterns if 'top' in p or 'head' in p)/len(active_patterns) if active_patterns else 0,
            'bullish_ratio': sum(1 for p in active_patterns if 'bottom' in p)/len(active_patterns) if active_patterns else 0
        }

    def detect_price_patterns(self):
        """Güncellenmiş Formasyon Tespit Metodu"""
        try:
            # Tüm formasyonları tarayalım
            patterns = {
                # Trend Dönüş Formasyonları
                'head_shoulders': self._detect_head_shoulders(),
                'inverse_head_shoulders': self._detect_inverse_head_shoulders(),
                'double_top': self._detect_double_top(),
                'double_bottom': self._detect_double_bottom(),
                
                # Trend Devam Formasyonları
                'cup_handle': self._detect_cup_handle(),
                'ascending_triangle': self._detect_ascending_triangle(),
                'descending_triangle': self._detect_descending_triangle(),
                'symmetrical_triangle': self._detect_symmetrical_triangle(),
                'bullish_flag': self._detect_bullish_flag(),
                'bearish_flag': self._detect_bearish_flag(),
                'pennant': self._detect_pennant(),
                
                # Diğer Formasyonlar
                'rising_wedge': self._detect_rising_wedge(),
                'falling_wedge': self._detect_falling_wedge(),
                'rounding_bottom': self._detect_rounding_bottom()
            }
            
            # DEBUG: Hangi formasyonlar tespit edildi?
            print("\n🔍 FORMATION DEBUG:")
            for pattern, status in patterns.items():
                print(f"{pattern.ljust(25)}: {'✅' if status else '❌'}")
            
            self.df['price_patterns'] = str({k:v for k,v in patterns.items() if v})
            return self.df
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.df

    def _detect_head_shoulders(self):
        """Omuz-Baş-Omuz tespiti"""
        try:
            self.omuz_bas_omuz()
            return len([f for f, _, _ in self.formasyonlar if f == 'Omuz Baş Omuz']) > 0
        except Exception as e:
            print(f"OBO Hatası: {str(e)}")
            return False

    def _detect_inverse_head_shoulders(self):
        """Ters Omuz-Baş-Omuz tespiti"""
        try:
            self.ters_omuz_bas_omuz()
            return len([f for f, _, _ in self.formasyonlar if f == 'Ters Omuz Baş Omuz']) > 0
        except Exception as e:
            print(f"Ters OBO Hatası: {str(e)}")
            return False

    def _detect_double_top(self):
        """İkili Tepe tespiti"""
        try:
            self.ikili_tepe_dip()
            return len([f for f, _, _ in self.formasyonlar if f == 'İkili Tepe']) > 0
        except Exception as e:
            print(f"İkili Tepe Hatası: {str(e)}")
            return False

    def _detect_double_bottom(self):
        """Çift Dip tespiti"""
        try:
            self.ikili_tepe_dip()
            return len([f for f, _, _ in self.formasyonlar if f == 'Çift Dip']) > 0
        except Exception as e:
            print(f"Çift Dip Hatası: {str(e)}")
            return False

    def _detect_cup_handle(self):
        """Çanak-Kulp tespiti"""
        try:
            self._fincan_kulp()
            return len([f for f, _, _ in self.formasyonlar if f == 'Fincan-Kulp']) > 0
        except Exception as e:
            print(f"Çanak-Kulp Hatası: {str(e)}")
            return False

    def _detect_triangle_patterns(self, direction):
        # Bu metodun mevcut kodda olup olmadığını kontrol etmek için
        # burada boş bir metod olarak bırakılıyor.
        return False

    def _detect_flag_pennant(self):
        # Bu metodun mevcut kodda olup olmadığını kontrol etmek için
        # burada boş bir metod olarak bırakılıyor.
        return False

    def _detect_bullish_flag(self):
        """Yükselen Bayrak tespiti"""
        try:
            self.bayrak_flama()
            return len([f for f, _, _ in self.formasyonlar if f == 'Yükselen Bayrak']) > 0
        except Exception as e:
            print(f"Yükselen Bayrak Hatası: {str(e)}")
            return False

    def _detect_bearish_flag(self):
        """Alçalan Bayrak tespiti"""
        try:
            self.bayrak_flama()
            return len([f for f, _, _ in self.formasyonlar if f == 'Alçalan Bayrak']) > 0
        except Exception as e:
            print(f"Alçalan Bayrak Hatası: {str(e)}")
            return False

    def _detect_pennant(self):
        """Flama tespiti"""
        try:
            self.bayrak_flama()
            return len([f for f, _, _ in self.formasyonlar if f == 'Flama']) > 0
        except Exception as e:
            print(f"Flama Hatası: {str(e)}")
            return False

    def _egim_analizi(self, seri):
        """Fiyat serisinin eğimini hesaplar"""
        try:
            return linregress(np.arange(len(seri)), seri.values).slope
        except:
            return 0

    def _canak_formasyonu(self):
        """Gelişmiş Çanak Formasyonu Tespiti"""
        pencere = self.params['formasyon_pencere']
        for i in range(pencere, len(self.df)-pencere):
            bolge = self.df.iloc[i-pencere:i]
            min_fiyat = bolge['low'].min()
            max_fiyat = bolge['high'].max()
            
            derinlik = (max_fiyat - min_fiyat) / min_fiyat
            egim = self._egim_analizi(bolge['close'])
            
            if (self.params['min_derinlik'] < derinlik < self.params['max_derinlik'] and
                egim > 0 and
                self.df['close'].iloc[i] > self.df['SMA_20'].iloc[i]):
                
                self.formasyonlar.append({
                    'tip': 'Çanak',
                    'pozisyon': i,
                    'seviye': self.df['close'].iloc[i],
                    'derinlik': derinlik,
                    'sinif': 'bullish'
                })

    def _fincan_kulp(self):
        """Fincan-Kulp Formasyonu Tespiti (Düzeltilmiş)"""
        try:
            pencere = self.params['formasyon_pencere']
            kulp_pencere = int(pencere * self.params['kulp_pencere_oran'])
            
            for i in range(pencere+kulp_pencere, len(self.df)):
                # Fincan kısmı
                fincan = self.df.iloc[i-pencere-kulp_pencere:i-kulp_pencere]
                fincan_min = fincan['low'].min()
                fincan_max = fincan['high'].max()
                
                # Kulp kısmı
                kulp = self.df.iloc[i-kulp_pencere:i]
                kulp_min = kulp['low'].min()
                kulp_max = kulp['high'].max()
                
                # Formasyon kriterleri
                fincan_derinlik = (fincan_max - fincan_min) / fincan_min
                kulp_derinlik = (kulp_max - kulp_min) / kulp_min
                
                if (self.params['min_derinlik'] < fincan_derinlik < self.params['max_derinlik'] and
                    0.01 < kulp_derinlik < 0.1 and
                    self.df['close'].iloc[i] > self.df['SMA_20'].iloc[i]):
                    
                    # Tuple formatında ekleme yap
                    self.formasyonlar.append(('Fincan-Kulp', i, None))
        except Exception as e:
            print(f"Fincan-Kulp Hatası: {str(e)}")

    def _yukselen_ucgen(self):
        """Yükselen Üçgen Formasyonu Tespiti"""
        pencere = self.params['formasyon_pencere']
        for i in range(pencere, len(self.df)):
            bolge = self.df.iloc[i-pencere:i]
            
            # Üst direnç çizgisi (yatay/yavaş düşen)
            ust_egim = self._egim_analizi(bolge['high'])
            
            # Alt destek çizgisi (yükselen)
            alt_egim = self._egim_analizi(bolge['low'])
            
            if (ust_egim < 0.005 and  # Yatay veya hafif düşüş
                alt_egim > 0.01 and   # Belirgin yükseliş
                (bolge['high'].max() - bolge['low'].min())/bolge['low'].min() < 0.2):
                
                self.formasyonlar.append({
                    'tip': 'Yükselen Üçgen',
                    'pozisyon': i,
                    'seviye': self.df['close'].iloc[i],
                    'ust_egim': ust_egim,
                    'alt_egim': alt_egim,
                    'sinif': 'bullish'
                })

    def get_formasyon_yorumlari(self):
        """Güncellenmiş Formasyon Yorumları"""
        formasyon_tanımları = {
            'Çanak': "🍯 Çanak Formasyonu (Orta Vadeli Yükseliş Beklentisi)",
            'Fincan-Kulp': "☕ Fincan-Kulp (Uzun Vadeli Güçlü Yükseliş)",
            'Yükselen Üçgen': "📈 Yükselen Üçgen (Kısa Vadeli Breakout Beklentisi)",
            'Omuz Baş Omuz': "⛰️ Omuz-Baş-Omuz (Trend Dönüş Sinyali)",
            'double_top': "🔻 Çift Tepe (Direnç Seviyesi)",
            'double_bottom': "🔼 Çift Dip (Destek Seviyesi)"
        }
        
        try:
            patterns = eval(self.df['price_patterns'].iloc[-1])
            return [formasyon_tanımları[p] for p in patterns if p in formasyon_tanımları]
        except:
            return ["Formasyon yorumlama hatası"]

    def _detect_ascending_triangle(self):
        """Yükselen Üçgen tespiti"""
        try:
            self.ucgen_formasyonlari()
            return len([f for f, _, _ in self.formasyonlar if f == 'Yükselen Üçgen']) > 0
        except Exception as e:
            print(f"Yükselen Üçgen Hatası: {str(e)}")
            return False

    def _detect_descending_triangle(self):
        """Alçalan Üçgen tespiti"""
        try:
            self.ucgen_formasyonlari()
            return len([f for f, _, _ in self.formasyonlar if f == 'Alçalan Üçgen']) > 0
        except Exception as e:
            print(f"Alçalan Üçgen Hatası: {str(e)}")
            return False

    def _detect_symmetrical_triangle(self):
        """Simetrik Üçgen tespiti"""
        try:
            self.ucgen_formasyonlari()
            return len([f for f, _, _ in self.formasyonlar if f == 'Simetrik Üçgen']) > 0
        except Exception as e:
            print(f"Simetrik Üçgen Hatası: {str(e)}")
            return False

    def _detect_rising_wedge(self):
        """Yükselen Kama tespiti"""
        try:
            pencere = self.params['kama_penceresi']
            for i in range(pencere, len(self.df)):
                high_slope = self._egim_hesapla(self.df['high'].iloc[i-pencere:i])
                low_slope = self._egim_hesapla(self.df['low'].iloc[i-pencere:i])
                
                if high_slope > 0.01 and low_slope > 0.008 and high_slope > low_slope:
                    self.formasyonlar.append(('Yükselen Kama', i, None))
            return len([f for f, _, _ in self.formasyonlar if f == 'Yükselen Kama']) > 0
        except Exception as e:
            print(f"Yükselen Kama Hatası: {str(e)}")
            return False

    def _detect_falling_wedge(self):
        """Alçalan Kama tespiti"""
        try:
            pencere = self.params['kama_penceresi']
            for i in range(pencere, len(self.df)):
                high_slope = self._egim_hesapla(self.df['high'].iloc[i-pencere:i])
                low_slope = self._egim_hesapla(self.df['low'].iloc[i-pencere:i])
                
                if high_slope < -0.01 and low_slope < -0.008 and high_slope < low_slope:
                    self.formasyonlar.append(('Alçalan Kama', i, None))
            return len([f for f, _, _ in self.formasyonlar if f == 'Alçalan Kama']) > 0
        except Exception as e:
            print(f"Alçalan Kama Hatası: {str(e)}")
            return False

    def _detect_rounding_bottom(self):
        """Yuvarlak Dip tespiti"""
        try:
            pencere = self.params['formasyon_pencere']
            for i in range(pencere, len(self.df)):
                bolge = self.df.iloc[i-pencere:i]
                price_change = (bolge['close'].iloc[-1] - bolge['close'].iloc[0])/bolge['close'].iloc[0]
                vol_increase = bolge['volume'].iloc[-5:].mean() > bolge['volume'].iloc[:5].mean()*1.5
                
                if abs(price_change) < 0.1 and vol_increase:
                    self.formasyonlar.append(('Yuvarlak Dip', i, None))
            return len([f for f, _, _ in self.formasyonlar if f == 'Yuvarlak Dip']) > 0
        except Exception as e:
            print(f"Yuvarlak Dip Hatası: {str(e)}")
            return False

# Kullanım Örneği
if __name__ == "__main__":
    # Örnek veri yükleme
    df = pd.read_csv('veri.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    # Analiz yap
    fa = FormasyonAnaliz(df)
    df = fa.detect_all_patterns()
    
    # Sonuçları göster
    print("Aktif Formasyonlar:", fa.get_detailed_analysis())
    fa.visualize_patterns() 