from flask import Flask, request, jsonify
from steam.service import SteamGameService
from pathlib import Path

app = Flask(__name__)
# Caminho relativo para o modelo na pasta model/
MODEL_PATH = Path(__file__).parent.parent.parent.parent / "model" / "best_game_model.pkl"
steam_service = SteamGameService(str(MODEL_PATH))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de predição.
    
    Recebe um JSON com 'data_tuple' contendo:
    [price, discount, median_forever, release_date, num_languages, owners, publisher_experience, ccu]
    
    Retorna:
    {
        "prediction": 0 ou 1 (0=fracasso, 1=sucesso),
        "confidence": probabilidade (0.0 a 1.0)
    }
    """
    try:
        data = request.get_json()
        data_tuple = data.get('data_tuple')
        
        if not data_tuple:
            return jsonify({'error': 'Missing data_tuple parameter'}), 400
        
        result = steam_service.predict(data_tuple)
        
        return jsonify({
            "prediction": result[0],
            "confidence": result[1],
            "success": bool(result[0]),
            "message": "Jogo previsto como SUCESSO (>=70% reviews positivas)" if result[0] == 1 
                      else "Jogo previsto como FRACASSO (<70% reviews positivas)"
        })
    
    except (TypeError, ValueError) as e:
        print(f"Erro: {e}")
        return jsonify({'error': 'Invalid parameters', 'details': str(e)}), 400
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Steam Game Success Predictor",
        "version": "1.0.0"
    })

@app.route('/', methods=['GET'])
def home():
    """Endpoint raiz com informações do serviço."""
    return jsonify({
        "service": "Steam Game Success Predictor",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Prediz sucesso de um jogo",
            "/health": "GET - Verifica status do serviço"
        },
        "input_format": {
            "data_tuple": [
                "price (float)",
                "discount (float, 0-100)",
                "median_forever (float, minutos)",
                "release_date (string, YYYY-MM-DD)",
                "num_languages (int)",
                "owners (float)",
                "publisher_experience (float, opcional)",
                "ccu (float, opcional)"
            ]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
