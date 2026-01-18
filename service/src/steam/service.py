"""
Serviço de predição de sucesso de jogos da Steam.
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime


class SteamGameService:
    """
    Um serviço de predição de sucesso de jogos da Steam.
    
    Prediz se um jogo terá >= 70% de reviews positivas baseado em suas características.
    """

    def __init__(self, file_model_path: str = None):
        """
        Inicializa o serviço carregando o modelo treinado.
        
        Args:
            file_model_path: Caminho para o arquivo .pkl do modelo
        """
        model_data = pickle.load(open(file_model_path, 'rb'))
        self.model = model_data["model"]
        self.features = model_data["features"]

    def predict(self, data_tuple: list = []) -> list:
        """
        Realiza a predição para um jogo.
        
        Args:
            data_tuple: Lista com [price, discount, median_forever, release_date, 
                       num_languages, owners, publisher_experience, ccu]
        
        Returns:
            Lista com [predição (0 ou 1), confiança]
        """
        # Features esperadas pelo modelo (já processadas)
        # ["price", "discount", "log_median_forever", "days_since_release",
        #  "num_languages", "log_popularity_density", "log_owners", "publisher_experience"]
        
        # Input: [price, discount, median_forever, release_date_str, num_languages, owners, publisher_exp, ccu]
        price = float(data_tuple[0])
        discount = float(data_tuple[1])
        median_forever = float(data_tuple[2])
        release_date_str = str(data_tuple[3])
        num_languages = int(data_tuple[4])
        owners = float(data_tuple[5])
        publisher_experience = float(data_tuple[6]) if len(data_tuple) > 6 else 1.0
        ccu = float(data_tuple[7]) if len(data_tuple) > 7 else 0.0
        
        # Processar features
        release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
        now = datetime(2026, 1, 18)
        days_since_release = (now - release_date).days
        
        # Calcular popularity_density
        popularity_density = owners / max(days_since_release, 1)
        
        # Aplicar transformações logarítmicas
        log_median_forever = np.log1p(median_forever)
        log_popularity_density = np.log1p(popularity_density)
        log_owners = np.log1p(owners)
        
        # Criar DataFrame com features processadas
        features_dict = {
            "price": price,
            "discount": discount,
            "log_median_forever": log_median_forever,
            "days_since_release": days_since_release,
            "num_languages": num_languages,
            "log_popularity_density": log_popularity_density,
            "log_owners": log_owners,
            "publisher_experience": publisher_experience
        }
        
        dataset = pd.DataFrame([features_dict])
        X = dataset[self.features]
        
        # Predição
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return [int(prediction), float(confidence)]
