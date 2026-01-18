# Serviço de Predição de Sucesso de Jogos da Steam

> **Parte 3 - Engenharia de Sistemas Inteligentes**\
> Universidade Federal do Ceará (UFC)\
> Autores: Rodrigo Sousa Barbosa, Clara Lima Silva, Ryan dos Santos Oliveira, Arthur Thomé Costa

API Flask que encapsula o modelo de ML treinado na Parte 2 para predição de sucesso de jogos da Steam.

## Instalação e Execução

### 1. Instalar dependências

```bash
cd service
poetry install
```

### 2. Iniciar o serviço

```bash
poetry run python run.py
```

Serviço disponível em `http://127.0.0.1:5000`

### 3. Testar o serviço

**Health check:**
```bash
curl http://127.0.0.1:5000/health
```

**Demo completa:**
```bash
poetry run python service_client.py
```

**Testes unitários:**
```bash
poetry run python -u -m unittest discover tests
```

## Endpoints

### `GET /health`
Verifica status do serviço.

```bash
curl http://127.0.0.1:5000/health
```

### `POST /predict`
Realiza predição de sucesso para um jogo.

**Entrada:** Lista com 8 valores `[price, discount, median_forever, release_date, num_languages, owners, publisher_exp, ccu]`

**Exemplo:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data_tuple": [59.99, 0, 720, "2024-03-15", 25, 500000, 5, 50000]}'
```

**Resposta:**
```json
{
  "prediction": 1,
  "confidence": 0.85,
  "success": true,
  "message": "Jogo previsto como SUCESSO (>=70% reviews positivas)"
}
```

## Formato de Entrada

| Posição | Campo | Tipo | Descrição |
|---------|-------|------|-----------|
| 0 | price | float | Preço em USD |
| 1 | discount | float | Desconto em % (0-100) |
| 2 | median_forever | float | Tempo médio jogado (minutos) |
| 3 | release_date | string | Data lançamento "YYYY-MM-DD" |
| 4 | num_languages | int | Número de idiomas |
| 5 | owners | float | Número estimado de proprietários |
| 6 | publisher_experience | float | Experiência da publisher |
| 7 | ccu | float | Concurrent users (peak) |

## Exemplo Python

```python
import requests

# AAA game
data = [59.99, 0, 720, "2024-03-15", 25, 500000, 5, 50000]

response = requests.post(
    'http://127.0.0.1:5000/predict',
    json={'data_tuple': data}
)

print(response.json())
# {'prediction': 1, 'confidence': 0.85, 'success': True, ...}
```

## Estrutura

```
service/
├── src/steam/
│   ├── app.py          # Flask API
│   └── service.py      # Lógica de predição
├── tests/
│   └── test_app.py     # Testes unitários
├── run.py              # Script de execução
├── service_client.py   # Cliente demo
└── pyproject.toml      # Dependências Poetry
```

## Como Funciona

1. **Carregamento**: Modelo carregado de `../model/best_game_model.pkl` na inicialização
2. **Processamento**: Features transformadas automaticamente (log, days_since_release, etc.)
3. **Predição**: Retorna 0 (fracasso) ou 1 (sucesso ≥70% reviews positivas)

## Cenários de Teste (service_client.py)

- **AAA RPG**: Alto orçamento, muitos owners → SUCESSO
- **Indie Metroidvania**: Bom engagement → SUCESSO
- **City Builder**: Nicho engajado → SUCESSO
- **Early Access**: Poucos dados → INCERTO
- **Indie Baixo Engagement**: Poucos owners/tempo → FRACASSO

## Tecnologias

- Flask 3.1.0+
- scikit-learn 1.5.0+
- pandas 2.2.0+
- Poetry (gerenciamento de dependências)

## Requisitos

- Python ≥ 3.10
- Modelo treinado em `../model/best_game_model.pkl`