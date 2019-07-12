import json
from run import run

config = json.load(open('config.json', 'r'))

run(config, classification=True, training=True)
