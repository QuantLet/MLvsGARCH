import json
from run import run

config = json.load(open('config.json', 'r'))

if "model" in config.keys():
    training = config["model"]["training"]
else:
    training = True

run(config, classification=True, training=training)
