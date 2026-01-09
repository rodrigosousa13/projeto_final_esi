import requests
from requests.exceptions import SSLError
from collections import OrderedDict
import logging
import pandas as pd
import time
import json
import os


def execution_timer(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        print(f"Starting execution of {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Finished execution of {func.__name__}. Time taken: {end_time - start_time:.2f} seconds.")
        return result
    return wrapper


def get_request(url, parameters=None):
    """Return json-formatted response of a get request using optional parameters.
    
    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request
    
    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)
        
        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' '*10)
        
        # recursively try again
        return get_request(url, parameters)
    
    if response:
        return response.json()
    else:
        # response is none usually means too many requests. Wait and try again 
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying.')
        return get_request(url, parameters)
    

def get_steamspy_all_data(path, last_page=86):
    url = "https://steamspy.com/api.php"
    all_data = {}

    for i in range(last_page + 1):
        parameters = {'request': 'all', 'page': i}
        data = get_request(url, parameters)
        all_data.update(data)

        if i < last_page:
            print(f'Página [{i}/{last_page}] carregada')
            print('Sleep de 60s...')
            time.sleep(60)
   
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


def select_even_game_ids(games_json: dict, k: int, return_as_str: bool = True):
    keys = list(games_json.keys())
    n = len(keys)
    if k <= 0:
        return []
    if k >= n:
        return keys if return_as_str else [int(x) for x in keys]
    if k == 1:
        mid = keys[n // 2]
        return [mid] if return_as_str else [int(mid)]
    indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    picked = [keys[i] for i in indices]
    return picked if return_as_str else [int(x) for x in picked]

@execution_timer
def get_game_id_list(path: str, amount: int = 5_000, return_as_str: bool = True):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    return select_even_game_ids(data, amount, return_as_str)


def get_detailed_game_data(appid: str):

    # SteamSpy
    steamspy_url = "https://steamspy.com/api.php"
    parameters = {'request': 'appdetails', 'appid': int(appid)}
    base = get_request(steamspy_url, parameters)
    if not isinstance(base, dict):
        return None
    
    # Steam Store
    steamapi_url = "http://store.steampowered.com/api/appdetails/"
    store_params = {'appids': appid}
    store = get_request(steamapi_url, store_params)

    if not isinstance(store, dict):
        return None

    entry = store.get(appid)
    if not entry or not entry.get('success'):
        return None
    
    data = entry.get('data', {})
    base.update({
        'release_date': (data.get('release_date') or {}).get('date'),
        'name': data.get('name')
    })

    return base


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='scraping.log', encoding='utf-8', level=logging.DEBUG)

    folder_path = 'datasets'
    json_name = 'steamspy_data.json'
    json_path = os.path.join(os.getcwd(), folder_path, json_name)
    if not os.path.exists(json_path):
        logger.debug('JSON do SteamSpy inexistente; Importando dados')
        logger.debug('\tETC: 1:26:00')
        get_steamspy_all_data()

    logger.debug('Criando Lista de appid\'s')
    amount = 5_000
    appid_list = get_game_id_list(amount)
    logger.debug(f'\tNúmero de appids: {len(appid_list)}')
    rows = []
    logger.debug('\nAcessando APIs para obtenção dos dados')
    etc = 1.5 * amount
    logger.debug(f'\tETC: {time.strftime("%H:%M:%S", time.gmtime(etc))}')
    for i, appid in enumerate(appid_list):
        app_data = get_detailed_game_data(appid)
        if app_data:
            rows.append(app_data)
        else:
            logger.warning(f'Dados não disponíveis para appid {appid}')
        print(f'[{i+1}/{amount}]')
        time.sleep(1)
    logger.debug(f'\tTotal de linhas do dataset: {len(rows)}')

    df = pd.DataFrame(rows)
    csv_name = 'steam_games_dataset.csv'
    csv_path = os.path.join(os.getcwd(), folder_path, csv_name)
    df.to_csv(csv_path, index=False)
    logger.debug(f'Dataset salvo em {csv_path}')


