<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: MLvsGARCHecon

Published in: Crypto volatility forecasting: ML vs GARCH

Description: Do a first econometrics analysis of btc log returns following Chen et al (2017) in `MLvsGARCHecon_1.R`, then build a rolling forecast with selected model `MLvsGARCHecon_2.R`

Keywords: econometrics, GARCH, EGARCH, volatility forecasting, realized volatility, cryptocurrency, btc

Author: Bruno Spilak

See also: 
- MLvsGARCHml
- MLvsGARCHcomp
- econ-tgarch
- econ_arch
- econ_arima
- econ_ccgar
- econ_crix
- econ_garch
- econ_vola

Submitted: 12.07.2019

Datafile: `../data/btc_1H_20160101_20190101.csv`: candle price of btc with 1 hour frequency

Input: The reader can modify the parameters of the model in each code file.

Output: 
- btc_log_returns.png
- btc_qqplot.png
- btc_qqplot_estgarch.png
- saved_models

```
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/MLvsGARCH/master/MLvsGARCHecon/btc_log_returns.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/MLvsGARCH/master/MLvsGARCHecon/btc_qqplot.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/MLvsGARCH/master/MLvsGARCHecon/btc_qqplot_estgarch.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/MLvsGARCH/master/MLvsGARCHecon/res_squared_p_acf.png" alt="Image" />
</div>

