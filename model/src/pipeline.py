import pandas as pd
import numpy as np
import pickle
import logging
import sys
import os
from datetime import datetime

from rich.console import Console
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import metrics


## Setando o Log 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", mode='w')
    ]
)
logger = logging.getLogger("__main__")


## Limpeza e Engenharia de Atributos
def preprocess_dataset(df: pd.DataFrame, max_error=0.10) -> pd.DataFrame:
    """
    Realiza a limpeza estatística e a engenharia de atributos do dataset.

    Executa o pipeline inicial de dados, dividindo-se em duas fases críticas:
    a filtragem de observações com alto erro amostral e a construção de variáveis 
    sintéticas (features) para o modelo de aprendizado de máquina.

    Args:
        df (pd.DataFrame): DataFrame bruto contendo dados extraídos da API da Steam.
        max_error (float): Margem de erro máxima tolerada para a nota do jogo (padrão 10%).

    Returns:
        pd.DataFrame: Conjunto de dados processado, filtrado e com as colunas 
            necessárias para o treinamento do modelo.
    """
    df = df.copy()
    initial_shape = df.shape[0]

    # --- PARTE 1: LIMPEZA (FILTRO DE LINHAS) ---
    # Cálculo do n e da Margem de Erro para garantir estabilidade do target
    df["total_reviews"] = df["positive"] + df["negative"]
    df = df[df["total_reviews"] > 0].copy()
    
    # E = 1.96 * sqrt(0.25 / n)
    df["margin_of_error"] = 1.96 * np.sqrt(0.25 / df["total_reviews"])
    
    # Removemos jogos com erro maior que 10% (aprox. < 96 reviews)
    df = df[df["margin_of_error"] <= max_error].copy()
    logger.debug(f"Sanity Check: {initial_shape - df.shape[0]} registros removidos (Erro > {max_error*100}%).")

    # --- PARTE 2: ENGENHARIA (CRIAÇÃO DE COLUNAS) ---
    # Definição do Target
    df["target"] = (df["positive"] / df["total_reviews"] >= 0.7).astype(int)
    
    # Atributos Temporais
    df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')
    now = datetime(2026, 1, 16) 
    df["days_since_release"] = (now - df["release_date"]).dt.days.fillna(0)
    
    # Atributos de Engajamento e Autoridade
    df["num_languages"] = df["languages"].fillna("").apply(lambda x: len(x.split(",")) if x else 1)
    df["popularity_density"] = df["owners"] / df["days_since_release"].replace(0, 1)
    
    pub_count = df.groupby("publisher").size()
    df["publisher_experience"] = df["publisher"].map(pub_count).fillna(1)

    # Transformações Logarítmicas para normalizar a escala
    for col in ["owners", "ccu", "median_forever", "popularity_density"]:
        df[f"log_{col}"] = np.log1p(df[col])
        
    return df


    
def run_pipeline(dataset_path) -> None:
    """
        Executa o pipeline completo de Machine Learning para predição de sucesso de jogos.

        Args:
            dataset_path (str): Caminho local para o arquivo CSV contendo o dataset 
                bruto da Steam. (opcional)

        Returns:
            None: A função não retorna valores em memória, mas gera artefatos de saída:
                - 'best_game_model.pkl': Modelo persistido e lista de features.
                - 'pipeline.log': Logs detalhados de cada fold e métricas finais.

        Raises:
            FileNotFoundError: Caso o caminho do dataset fornecido seja inválido.
            ValueError: Se o dataset não contiver as colunas mínimas necessárias 
                para o cálculo do target.
        """
    df_raw = pd.read_csv(dataset_path)
    df = preprocess_dataset(df_raw)

    features = [
        "price", "discount", "log_median_forever", "days_since_release",
        "num_languages", "log_popularity_density",
        "log_owners", "publisher_experience"

    ]

    X = df[features].dropna()
    y = df.loc[X.index, "target"]

    # Definição dos Modelos Individuais
    lr = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression(max_iter=1000))])
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42)
    svc = Pipeline([("scaler", RobustScaler()), ("clf", SVC(kernel='linear', probability=True, random_state=42))])
    ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')

    models = {
        "LRC": lr,
        "RFC": rf,
        "SVC": svc,
        "Ensemble_Voting": ensemble
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
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            roc = metrics.roc_auc_score(y_test, y_proba)
            f1 = metrics.f1_score(y_test, y_pred)
            acc = metrics.accuracy_score(y_test, y_pred)
            pre = metrics.precision_score(y_test, y_pred)
            rec = metrics.recall_score(y_test, y_pred)

            model_scores[name].append(roc)
            logger.debug(f"Fold {fold} - Model {name}: ROC-AUC={roc}, F1={f1}, Acc={acc}, Pre={pre}, Rec={rec}")



    # [Step-2] Seleção
    logger.debug("")
    logger.debug("[Step-2] Selecionando Melhor Modelo")
    best_model_name = max(model_scores, key=lambda k: np.mean(model_scores[k]))
    logger.debug("")
    logger.debug(f"Best Model {best_model_name}")

    # [Step-3] Modelo Final (best)
    logger.debug("")
    logger.debug("[Step-3] Criando o Modelo Final")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_f, y_train_f)

    best_model = models[best_model_name]
    best_model.fit(X_res, y_res)

    y_pred_f = best_model.predict(X_test_f)
    y_proba_f = best_model.predict_proba(X_test_f)[:, 1]

    logger.debug(f"best {best_model_name}: ROC-AUC={metrics.roc_auc_score(y_test_f, y_proba_f)}, F1={metrics.f1_score(y_test_f, y_pred_f)}, Acc={metrics.accuracy_score(y_test_f, y_pred_f)}, Pre={metrics.precision_score(y_test_f, y_pred_f)}, Rec={metrics.recall_score(y_test_f, y_pred_f)}")


    # Salva o arquivo pickle
    with open("best_game_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": features}, f)

## Inicialização
if __name__ == "__main__":
    console = Console()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_PATH = os.path.join(BASE_DIR, "..", "..", "data", "src", "data", "datasets", "steam_games_dataset_tratado.csv")
    
    LOG_FILE = "pipeline.log"
    log_uri = Path(os.path.abspath(LOG_FILE)).as_uri()

    # Caso não seja passado dataset como argumento, utilizará o caminho padrão
    target_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    
    if os.path.exists(target_path):
    
        with console.status("Iniciando treinamento... Aguarde", spinner="dots"):
            run_pipeline(target_path)
        console.print("Terminado, confira [i][link=file:///{log_uri}]pipeline.log[/link][i]")
    
    else:
        console.print(f"[bold red]Erro:[/bold red] Arquivo não encontrado em: [yellow]{target_path}[/yellow]")