import unittest
from steam.app import app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        """Testa o endpoint de health check."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')

    def test_predict_success_aaa_game(self):
        """Testa predição para jogo AAA (deve ter sucesso)."""
        # Jogo AAA: alto preço, muitos owners, alto engagement
        data_tuple = [59.99, 0, 720, "2024-03-15", 25, 500000, 5, 50000]
        data_tuple = [float(element) if i != 3 else element for i, element in enumerate(data_tuple)]
        
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        
        result = response.get_json()
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        # Espera-se sucesso (1) para jogo AAA bem estabelecido
        self.assertEqual(result['prediction'], 1)

    def test_predict_indie_game(self):
        """Testa predição para jogo indie."""
        # Jogo indie: preço médio, poucos owners, engagement moderado
        data_tuple = [14.99, 15, 240, "2025-11-01", 5, 15000, 1, 200]
        data_tuple = [float(element) if i != 3 else element for i, element in enumerate(data_tuple)]
        
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        
        result = response.get_json()
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        # Resultado pode variar para indie game
        self.assertIn(result['prediction'], [0, 1])

    def test_predict_f2p_game(self):
        """Testa predição para jogo Free-to-Play."""
        # F2P: preço zero, muitos owners, alto engagement
        data_tuple = [0.0, 0, 1200, "2022-01-15", 30, 2000000, 10, 100000]
        data_tuple = [float(element) if i != 3 else element for i, element in enumerate(data_tuple)]
        
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        
        result = response.get_json()
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        # F2P popular tende a ter sucesso
        self.assertEqual(result['prediction'], 1)

    def test_predict_failed_game(self):
        """Testa predição para jogo com características de fracasso."""
        # Jogo com baixo engagement, poucos owners, recém-lançado
        data_tuple = [4.99, 0, 30, "2026-01-10", 2, 100, 1, 5]
        data_tuple = [float(element) if i != 3 else element for i, element in enumerate(data_tuple)]
        
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        
        result = response.get_json()
        self.assertIn('prediction', result)
        # Jogo com muito poucos owners tende a fracassar
        self.assertEqual(result['prediction'], 0)

    def test_predict_missing_data(self):
        """Testa erro quando falta data_tuple."""
        response = self.client.post('/predict', json={})
        self.assertEqual(response.status_code, 400)
        
        result = response.get_json()
        self.assertIn('error', result)

    def test_predict_invalid_data(self):
        """Testa erro com dados inválidos."""
        # Data inválida
        data_tuple = [29.99, 20, 360, "invalid-date", 12, 50000, 1, 1500]
        
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
