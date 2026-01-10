# Análise exploratória inicial do dataset da Steam

# =========================
# Imports
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Carregamento do dataset
# =========================
DATA_PATH = "data/src/data/datasets/steam_games_dataset.csv"

df = pd.read_csv(DATA_PATH)

# =========================
# Verificação inicial
# =========================
print("Dimensão do dataset:")
print(df.shape)

print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

# =========================
# Análise de valores ausentes
# =========================

# Cálculo de valores ausentes por coluna
missing_count = df.isnull().sum()
missing_percentage = df.isnull().mean() * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percentage
})

# Filtragem para mostrar apenas colunas com valores ausentes
missing_df = missing_df[missing_df['Missing Count'] > 0]
missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)

# Exibição dos resultados
print("\nValores ausentes por coluna:")
print(missing_df)

df_sem_score = df.drop(columns=['score_rank'], errors='ignore')

linhas_com_ausentes = df_sem_score[df_sem_score.isnull().any(axis=1)]
print(f"\nNúmero de linhas com pelo menos um valor ausente (excluindo 'score_rank'): {linhas_com_ausentes.shape[0]}")
print(linhas_com_ausentes)

