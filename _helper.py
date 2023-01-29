import json

def get_config():
    with open('config.json') as f:
        CONFIG = json.load(f)
    return CONFIG