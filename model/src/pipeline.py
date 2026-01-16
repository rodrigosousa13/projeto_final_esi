import pandas
import pickle
import logging
import sys

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn import metrics

# =====================================================
# Utils
# =====================================================

def load_dataset(dataset_path) -> pandas.DataFrame:
    """ Carrega um arquivo CSV para treinamento 
    
    Parameters: dataset_path (str): caminho para o arquivo

    Returns: (pandas.DataFrame): Dataframe com os dados carregados
    
    """
    return pandas.read_csv(dataset_path,index_col=None,header=0)


def create_target(dataset, threshold=0.7):
    dataset = dataset.copy()
    dataset["review_ratio"] = (
        dataset["positive"] / (dataset["positive"] + dataset["negative"])
    )
    dataset["GOOD_GAME"] = (dataset["review_ratio"] >= threshold).astype(int)
    return dataset


def save_model(model, model_name, data_balance, cv_criteria):
    """ Salva o modelo fornecido em um arquivo pickle """      
    with open(f"model_{model_name}_{cv_criteria.upper()}_{data_balance}.pkl", "wb") as f:
        pickle.dump(model, f)



def extract_model_metrics_scores(y_test, y_pred) -> dict:
    return {
        "accuracy_score": metrics.accuracy_score(y_test, y_pred),
        "roc_auc_score": metrics.roc_auc_score(y_test, y_pred),
        "f1_score": metrics.f1_score(y_test, y_pred),
        "precision_score": metrics.precision_score(y_test, y_pred),
        "recall_score": metrics.recall_score(y_test, y_pred),
        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
        "classification_report": metrics.classification_report(y_test, y_pred)
    }

# =====================================================
# Experiment
# =====================================================

def run_experiment(dataset, x_features, y_label, data_balance, models, grid_params_list, cv_criteria):
    X = dataset[x_features]
    y = dataset[y_label]

    skf = StratifiedKFold(n_splits=10)
    models_info_per_fold = {}

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if data_balance == "SMOTE":
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)

        models_info = {}
        for model_name in models:
            grid = GridSearchCV(
                models[model_name],
                grid_params_list[model_name],
                cv=5,
                scoring=cv_criteria
            )
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            models_info[model_name] = {
                "score": extract_model_metrics_scores(y_test, y_pred),
                "best_estimator": grid.best_estimator_
            }

        models_info_per_fold[i] = models_info

    return models_info_per_fold

# =====================================================
# Main orchestration
# =====================================================

def start(dataset_path):
    
    logging.basicConfig(filename="pipeline.log", level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    dataset = load_dataset(dataset_path)
    dataset = create_target(dataset)


    x_features = [c for c in dataset.columns if c != "GOOD_GAME"]
    y_label = "GOOD_GAME"

    models = {
        "LRC": LogisticRegression(max_iter=10**6),
        "RFC": RandomForestClassifier(),
        "SVC": LinearSVC()
    }

    grid_params = {
        "LRC": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear"]
        },
        "RFC": {
            "n_estimators": [100, 200],
            "max_depth": [6, 10, None]
        },
        "SVC": {
            "C": [0.01, 0.1, 1, 10]
        }
    }

    logger.debug("[Step-1] Benchmark")
    fold_results = run_experiment(
        dataset,
        x_features,
        y_label,
        data_balance="SMOTE",
        models=models,
        grid_params_list=grid_params,
        cv_criteria="roc_auc"
    )

    logger.debug("[Step-2] Resultados por fold")
    for fold, info in fold_results.items():
        for model_name, result in info.items():
            sc = result["score"]
            logger.debug(
                f"Fold {fold} | {model_name} | "
                f"ROC={sc['roc_auc_score']:.3f} | "
                f"F1={sc['f1_score']:.3f}"
            )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        start(sys.argv[1])
    else:
        print("Informe o caminho do dataset.")
