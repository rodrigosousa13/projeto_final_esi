import pandas as pd
import numpy as np
import pickle
import logging
import sys
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import metrics

# =====================================================
# 1. Configuração de Log (Formato exato solicitado)
# =====================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        # Grava tudo no arquivo de log de forma detalhada
        logging.FileHandler("pipeline.log", mode='w')
    ]
)
logger = logging.getLogger("__main__")

# =====================================================
# 2. Engenharia de Atributos (Mantendo suas alterações)
# =====================================================
def clean_and_prepare(df):
    df = df.copy()
    
    # Target (Sucesso do Jogo)
    total_reviews = df["positive"] + df["negative"]
    df["target"] = (df["positive"] / total_reviews.replace(0, 1) >= 0.7).astype(int)
    
    # Atributos Temporais e de Engajamento
    df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')
    now = datetime(2026, 1, 16) 
    df["days_since_release"] = (now - df["release_date"]).dt.days.fillna(0)
    df["num_languages"] = df["languages"].fillna("").apply(lambda x: len(x.split(",")) if x else 1)
    
    # Cálculos das métricas base
    df["engagement_ratio"] = df["ccu"] / df["owners"].replace(0, 1)
    df["popularity_density"] = df["owners"] / df["days_since_release"].replace(0, 1)
    
    pub_count = df.groupby("publisher").size()
    df["publisher_experience"] = df["publisher"].map(pub_count).fillna(1)

    # Transformação de Escala (Log)
    for col in ["owners", "ccu", "median_forever", "popularity_density"]:
        df[f"log_{col}"] = np.log1p(df[col])
        
    return df

# =====================================================
# 3. Execução do Pipeline
# =====================================================
def run_pipeline(dataset_path):
    df_raw = pd.read_csv(dataset_path)
    df = clean_and_prepare(df_raw)

    # Features: Mantendo exatamente a sua seleção (não mexi aqui)
    features = [
        "price", "discount", "log_median_forever", "days_since_release",
        "num_languages", "log_popularity_density", 
        "log_owners", "publisher_experience"
    ]
    
    X = df[features].dropna()
    y = df.loc[X.index, "target"]

    # Modelos: Mantendo apenas LRC e RFC conforme seu código
    models = {
        "LRC": Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "RFC": RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42)
    }

    # [Step-1] Benchmark
    logger.debug("[Step-1] Realizando Benchmark")
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    model_scores = {name: [] for name in models.keys()}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        for name, model in models.items():
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            roc = metrics.roc_auc_score(y_test, y_proba)
            f1 = metrics.f1_score(y_test, y_pred)
            acc = metrics.accuracy_score(y_test, y_pred)
            pre = metrics.precision_score(y_test, y_pred)
            rec = metrics.recall_score(y_test, y_pred)

            model_scores[name].append(roc)
            
            # Formato de Log exato que você pediu
            logger.debug(f"Fold {fold} - Model {name}: ROC-AUC={roc}, F1={f1}, Acc={acc}, Pre={pre}, Rec={rec}")

    # [Step-2] Seleção
    logger.debug("")
    logger.debug("[Step-2] Selecionando Melhor Modelo")
    best_model_name = max(model_scores, key=lambda k: np.mean(model_scores[k]))
    logger.debug("")
    logger.debug(f"Best Model {best_model_name}")

    # [Step-3] Modelo Final (Champion)
    logger.debug("")
    logger.debug("[Step-3] Criando o Modelo Final")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_f, y_train_f)
    
    champion_model = models[best_model_name]
    champion_model.fit(X_res, y_res)
    
    y_pred_f = champion_model.predict(X_test_f)
    y_proba_f = champion_model.predict_proba(X_test_f)[:, 1]
    
    c_roc = metrics.roc_auc_score(y_test_f, y_proba_f)
    c_f1 = metrics.f1_score(y_test_f, y_pred_f)
    c_acc = metrics.accuracy_score(y_test_f, y_pred_f)
    c_pre = metrics.precision_score(y_test_f, y_pred_f)
    c_rec = metrics.recall_score(y_test_f, y_pred_f)

    logger.debug(f"Champion {best_model_name}: ROC-AUC={c_roc}, F1={c_f1}, Acc={c_acc}, Pre={c_pre}, Rec={c_rec}")

    # Salva o arquivo pickle
    with open("best_game_model.pkl", "wb") as f:
        pickle.dump({"model": champion_model, "features": features}, f)

# =====================================================
# 4. Inicialização e Mensagens de Terminal
# =====================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Ajuste automático do caminho baseado na estrutura de pastas vista na sua imagem
    DEFAULT_PATH = os.path.join(BASE_DIR, "..", "..", "data", "src", "data", "datasets", "steam_games_dataset_tratado.csv")
    
    target_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    
    if os.path.exists(target_path):
        print("Iniciando treinamento... Aguarde.")
        run_pipeline(target_path)
        print("Terminado.")
    else:
        print(f"Erro: Arquivo não encontrado em {target_path}")