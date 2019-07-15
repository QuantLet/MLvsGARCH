[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **MLvsGARCHml** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'MLvsGARCHml'

Published in: 'Crypto volatility forecasting: ML vs GARCH'

Description: 'Train a LSTM neural network in a cross-validation manner based on architecture defined in `core.py` and `config.json` file.'

Keywords: 'deep learning, recurrent neural network, LSTM, time series cross-validation, realised volatility, cryptocurrency, btc'

Author: 'Bruno Spilak'

See also:
- MLvsGARCHml
- MLvsGARCHcomp

Submitted:  '12.07.2019'

Datafile: '`../data/btc_1H_20160101_20190101.csv`: candle price of btc with 1 hour frequency'

Input:  'The reader can modify the parameters of the model in each code file and config file.'

Output:
- saved_models

```

### PYTHON Code
```python

import json
from run import run

config = json.load(open('config.json', 'r'))

run(config, classification=True, training=True)

```

automatically created on 2019-07-15