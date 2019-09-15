import json
from run import run

config = json.load(open('config.json', 'r'))

if "load_model" in config.keys():
    training = config["load_model"]["training"]
else:
    training = True

run(config, classification=True, training=training)
