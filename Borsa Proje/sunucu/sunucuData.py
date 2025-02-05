from datetime import datetime
import pandas as pd
import boto3
from botocore.exceptions import ClientError

class SunucuData:
    def __init__(self):
        self.data = None
        self.last_update = None
        
    def load_data(self, file_path):
        """Veri dosyasını yükler"""
        try:
            self.data = pd.read_csv(file_path)
            self.last_update = datetime.now()
            return True
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return False
            
    def get_data(self):
        """Mevcut veriyi döndürür"""
        return self.data
        
    def get_last_update(self):
        """Son güncelleme zamanını döndürür"""
        return self.last_update
        
    def update_data(self, new_data):
        """Veriyi günceller"""
        try:
            self.data = new_data
            self.last_update = datetime.now()
            return True
        except Exception as e:
            print(f"Veri güncelleme hatası: {e}")
            return False 

def get_api_credentials():
    # Gerçek API anahtarlarınızı buraya ekleyin
    return {
        'API_KEY': 'your_api_key_here',
        'API_SECRET': 'your_api_secret_here'
    }

def get_api_credentials():
    """API anahtarlarını AWS Secrets Manager'dan al"""
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name='eu-north-1'
        )
        
        response = client.get_secret_value(
            SecretId='binance-api-keys'
        )
        
        if 'SecretString' in response:
            return eval(response['SecretString'])
        else:
            raise Exception("SecretString bulunamadı")
            
    except ClientError as e:
        print(f"AWS Hatası: {e.response['Error']['Code']}")
        return None
    except Exception as e:
        print(f"Genel Hata: {str(e)}")
        return None 