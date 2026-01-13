# Análise exploratória inicial do dataset da Steam

# =========================
# Imports
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
# Tratamento de valores ausentes
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

# Coluna com ~99% de valores ausentes e sem relevância aparente
df_tratado = df.drop(columns=['score_rank'], errors='ignore')

# Dataset possui ~1% de linhascom valores ausentes.
# Optou-se por removê-las para simplificar a preparação dos dados.
df_tratado = df_tratado.dropna()

# =========================
# Tratamento de valores duplicados
# =========================

# Verificando a existência de linhas duplicadas
duplicated_count = df_tratado.duplicated(subset= df_tratado.columns.difference(['appid'])).sum()
print(f"\nNúmero de linhas duplicadas: {duplicated_count}")

#Como não há colunas-chave para identificar duplicatas, não há necessidade de removê-las.

# =========================
# Tratando Valores Inconsistentes
# =========================

# Transformando release_date datetime
df_tratado_dt = df_tratado.copy()
df_tratado_dt["release_date"] = pd.to_datetime(df_tratado_dt["release_date"], errors='coerce')

print(df_tratado[df_tratado_dt['release_date'].isnull()])

# Verificando se há datas inválidas
invalid_dates = df_tratado_dt['release_date'].isnull().sum()
print(f"\nNúmero de datas de lançamento inválidas: {invalid_dates}")

# Removendo linhas com datas inválidas
df_tratado["release_date"] = pd.to_datetime(df_tratado["release_date"], errors='coerce') 
df_tratado = df_tratado.dropna(subset=['release_date'])
print(f"Números de linhas com valores ausentes após remoção: {df_tratado.isnull().sum().sum()}")
    
# Transformando owners em int

# Remover vírgulas e garantir string
owners_str = df_tratado['owners'].astype(str).str.replace(',', '', regex=False)

# Extrair intervalos do tipo '10000 .. 20000' e usar apenas o limite inferior (min)
ranges = owners_str.str.extract(r'(?P<min>\d+)\s*\.\.\s*(?P<max>\d+)')
has_range = ranges['min'].notnull()
if has_range.any():
    # Usar apenas o valor mínimo como estimativa
    df_tratado.loc[has_range, 'owners'] = ranges.loc[has_range, 'min'].astype(int)

# Valores que já são números simples
is_number = owners_str.str.match(r'^\d+$')
if is_number.any():
    df_tratado.loc[is_number, 'owners'] = owners_str[is_number].astype(int)

# Forçar conversão numérica final (valores não parseáveis viram NaN)
df_tratado['owners'] = pd.to_numeric(df_tratado['owners'], errors='coerce')

print(df_tratado['owners'].head())
print(f"Números de linhas com valores ausentes após conversão de owners: {df_tratado['owners'].isnull().sum()}")   

#========================
# Análise Estatística Descritiva
#========================

print("\nEstatísticas descritivas:")
print(df_tratado.describe(include='all'))
# =========================
# Visualizações
# =========================

# Diretório para salvar gráficos
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Heatmap de correlação entre variáveis numéricas

corr_cols = df_tratado.select_dtypes(include=['number']).columns
plt.figure(figsize=(12, 8))
correlation_matrix = df_tratado[corr_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), bbox_inches='tight')
plt.close()

# Histograma da coluna 'price'
plt.figure(figsize=(10, 6))
sns.histplot(df_tratado['price'], bins=50, kde=True)
plt.title('Distribuição de Preços dos Jogos na Steam')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.xlim(0, df_tratado['price'].quantile(0.95))  # Limitar eixo x para melhor visualização
plt.savefig(os.path.join(OUTPUT_DIR, 'price_distribution.png'), bbox_inches='tight')
plt.close()

# Gráfico de barras das gêneros mais comuns
plt.figure(figsize=(12, 8))
top_genres = df_tratado['genre'].value_counts().nlargest(10)
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title('Top 10 Gêneros de Jogos na Steam')
plt.xlabel('Número de Jogos')
plt.ylabel('Gênero')
plt.savefig(os.path.join(OUTPUT_DIR, 'top_genres.png'), bbox_inches='tight')
plt.close()

# Gráfico de dispersão entre 'price' e 'owners'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_tratado, x='price', y='owners', alpha=0.5)
plt.title('Relação entre Preço e Número de Donos')
plt.xlabel('Preço')
plt.ylabel('Número de Donos')
plt.xlim(0, df_tratado['price'].quantile(0.95))  # Limitar eixo x para melhor visualização
plt.ylim(0, df_tratado['owners'].quantile(0.95))  # Limitar eixo y para melhor visualização
plt.savefig(os.path.join(OUTPUT_DIR, 'price_vs_owners.png'), bbox_inches='tight')
plt.close()

# =========================
# Salvando o dataset tratado
# =========================

OUTPUT_PATH = "data/src/data/datasets/steam_games_dataset_tratado.csv"
df_tratado.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset tratado salvo em: {OUTPUT_PATH}")
