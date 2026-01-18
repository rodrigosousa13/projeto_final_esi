"""
Cliente de demonstração para o serviço de predição de sucesso de jogos da Steam.

Testa o modelo com jogos reais populares da Steam.
"""

import requests
import json


def test_real_game(game_name: str, data_tuple: list, real_success: str):
    """
    Testa predição para um jogo real.
    
    Args:
        game_name: Nome do jogo
        data_tuple: Dados do jogo
        real_success: Sucesso real do jogo (para comparação)
    """
    print(f"\n{'~'*80}")
    print(f"JOGO: {game_name}")
    print(f"Sucesso Real: {real_success}")
    print(f"{'~'*80}")
    
    # Converter para float (exceto release_date)
    data_tuple_formatted = [
        float(element) if i != 3 else element 
        for i, element in enumerate(data_tuple)
    ]
    
    print(f"\nDados enviados:")
    labels = ["Preço", "Desconto%", "Tempo médio (min)", "Data lançamento", 
              "Idiomas", "Owners", "Pub. Exp.", "CCU"]
    for label, value in zip(labels, data_tuple_formatted):
        print(f"  {label:20s}: {value}")
    
    try:
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            json={'data_tuple': data_tuple_formatted}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[RESULTADO DA PREDIÇÃO]")
            print(f"  Predição: {'SUCESSO' if result['success'] else 'FRACASSO'}")
            print(f"  Confiança: {result['confidence']:.1%}")
            print(f"  {result['message']}")
            
            # Comparar com expectativa
            if "SUCESSO" in real_success and result['success']:
                print(f"\n  ✓ Predição CORRETA! Modelo identificou padrão de sucesso")
            elif "FRACASSO" in real_success and not result['success']:
                print(f"\n  ✓ Predição CORRETA! Modelo identificou padrão de fracasso")
            elif "INCERTO" in real_success:
                print(f"\n  ℹ️  Cenário incerto - predição exploratória")
            else:
                print(f"\n  ✗ Predição não corresponde à expectativa")
        else:
            print(f"\n[ERRO] Status: {response.status_code}")
            print(response.json())
    except requests.exceptions.ConnectionError:
        print("\n[ERRO] Serviço não está rodando!")
        print("Inicie com: poetry run python run.py")


def main():
    """Função principal de demonstração."""
    
    print("~"*80)
    print("DEMONSTRAÇÃO DO SERVIÇO DE PREDIÇÃO DE SUCESSO DE JOGOS DA STEAM")
    print("~"*80)
    print("\nTestes com cenários hipotéticos de jogos")
    print("(Dados baseados em padrões típicos da indústria)")
    
    # Formato: [price, discount, median_forever, release_date, num_languages, owners, publisher_exp, ccu]
    
    # 1. Cenário: AAA Premium de grande estúdio
    # Exemplo: novo RPG de estúdio AAA
    aaa_rpg = [69.99, 0, 1200, "2025-03-15", 22, 8000000, 10, 350000]
    test_real_game(
        "Cenário AAA: Initial Reality XXI",
        aaa_rpg,
        "SUCESSO esperado (padrão típico de AAA bem recebido)"
    )
    
    # 2. Cenário: Indie metroidvania de qualidade
    # Exemplo: jogo estilo Hollow Knight
    indie_metroid = [19.99, 0, 950, "2025-06-12", 12, 2500000, 1, 45000]
    test_real_game(
        "Full Knight: CottonMusic",
        indie_metroid,
        "SUCESSO esperado (indie com boa recepção)"
    )
    
    # 3. Cenário: Shooter AAA de franquia estabelecida
    # Exemplo: novo jogo de ação/aventura
    aaa_action = [59.99, 0, 850, "2025-02-20", 20, 6000000, 9, 280000]
    test_real_game(
        "Cenário AAA: The Legend of Lonk - Wheezer of the Woods",
        aaa_action,
        "SUCESSO esperado (franquia com base fiel)"
    )
    
    # 4. Cenário: Jogo de estratégia turn-based
    # Exemplo: novo jogo de estratégia 4X
    strategy_4x = [59.99, 0, 1400, "2025-02-11", 18, 4500000, 10, 95000]
    test_real_game(
        "Cenário Estratégia: The Last Birthday: Party 33",
        strategy_4x,
        "SUCESSO esperado (nicho mas engajamento alto)"
    )
    
    # 5. Cenário: Indie roguelike de sucesso
    # Exemplo: jogo estilo Hades/Vampire Survivors
    indie_roguelike = [14.99, 0, 720, "2025-04-20", 10, 3200000, 2, 65000]
    test_real_game(
        "Cenário Indie: Pizza Survivors",
        indie_roguelike,
        "SUCESSO esperado (fórmula comprovada)"
    )
    
    # 6. Cenário: Simulação/City Builder
    # Exemplo: jogo de construção de cidades
    city_builder = [29.99, 0, 680, "2025-04-26", 14, 2800000, 3, 58000]
    test_real_game(
        "Cenário Simulação: City Sim",
        city_builder,
        "SUCESSO esperado (nicho engajado)"
    )
    
    # 7. Cenário: RPG de estúdio médio
    # Exemplo: RPG de estúdio respeitado
    mid_rpg = [49.99, 0, 890, "2025-08-15", 15, 2200000, 6, 52000]
    test_real_game(
        "Cenário Médio: RPG de Estúdio Estabelecido",
        mid_rpg,
        "SUCESSO esperado (público fiel ao estúdio)"
    )
    
    # 8. Cenário: Jogo de mundo aberto ambicioso
    # Exemplo: novo open world com produção alta
    open_world = [69.99, 0, 1350, "2025-09-20", 24, 5500000, 8, 185000]
    test_real_game(
        "Cenário AAA: Mundo Aberto Ambicioso",
        open_world,
        "SUCESSO esperado (alto orçamento, boa execução)"
    )
    
    # 9. Cenário: Indie com baixo engagement
    # Exemplo: jogo pequeno com pouco tempo jogado
    indie_fail = [9.99, 0, 95, "2025-12-05", 4, 3500, 1, 65]
    test_real_game(
        "Cenário Indie: Baixo Engagement",
        indie_fail,
        "FRACASSO esperado (pouco tempo jogado, poucos owners)"
    )
    
    # 10. Cenário: Early Access muito recente
    # Exemplo: jogo lançado há poucos dias
    early_access = [24.99, 10, 180, "2026-01-15", 8, 8500, 1, 320]
    test_real_game(
        "Cenário Early Access: Lançamento Recente",
        early_access,
        "INCERTO (dados ainda muito limitados)"
    )
    
    print("\n" + "~"*80)
    print("RESUMO")
    print("~"*80)
    print("\nTestes realizados com cenários hipotéticos.")
    print("O modelo foi treinado com dados da Steam até ~2024.")
    print("\nEste é um teste de GENERALIZAÇÃO do modelo:")
    print("  ✓ Testa padrões típicos da indústria de jogos")
    print("  ✓ Predições baseadas em características aprendidas")
    print("\nO que o modelo avalia:")
    print("  - Engajamento (tempo jogado)")
    print("  - Alcance (número de idiomas, owners)")
    print("  - Popularidade (CCU - concurrent users)")
    print("  - Faixa de preço vs valor percebido")
    print("  - Experiência da publisher")
    print("\nJogos com alto engajamento + muitos owners + boa distribuição = SUCESSO")
    print("~"*80)


if __name__ == "__main__":
    main()
