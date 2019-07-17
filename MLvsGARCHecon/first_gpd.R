rm(list = ls(all = TRUE))
graphics.off()

set.seed(10)

library(POT)
library(ismev)
library(fGarch)
library(rugarch)
library(MLmetrics)

# Load variables and helper functions
dset = "../data/btc_1H_20160101_20190101"


dset.name = paste("./", dset, ".csv", sep = "")
dataset <- read.csv(dset.name, header = TRUE)
dates = dataset$X
dataset = data.frame("close" = dataset$close)
rownames(dataset) = dates
dataset <- timeSeries::as.timeSeries(dataset, FinCenter = "GMT")

window_size = 24*30

# If testing on subset of data
#dataset = dataset[(length(dataset) - 2000):length(dataset), 1]


# Convert price series to loss series
dataset = - na.omit(diff(log(dataset)))
length.dataset = length(dataset[, 1])
dates = rownames(dataset)

# VaR for a Generalized Pareto Distribution (GDP)
gdp.quantile = function(probs, threshold, scale, shape, n, Nu){
  quant = threshold + (scale / shape) * (( ( n/Nu ) * ( 1 - probs ) )^(-shape) - 1)
  return(quant)
}

var.normal = function(mean, sd, probs)
{
  var = mean + sd * qnorm(p=probs)
  return(var)
}

# VaR for a Generalized Pareto Distribution (GDP)
var.gpd = function(threshold, scale, shape, probs, n, Nu)
{
  var = threshold + (scale / shape) * (( ( n/Nu ) * ( 1 - probs ) )^(-shape) - 1)
  return(var)
}

# Split data
refit = 24

# Now fit GARCH(1,1) to the series
test_size = 10 #(length(dataset) - window_size)
qs = c(0.90, 0.95, 0.975, 0.99)
prediction = matrix(nrow = test_size, ncol = ( 3 + ( 2 * length(qs) ) ) )

count = 1
for (i in (window_size + 1):(window_size + test_size)){
  print(i)
  if (i%%1000 == 0){
    print(( length(dataset) - i))
    print(prediction[(i- window_size- 1),])
    save(prediction, file = 'prediction_online_first.RData')
  }
  n = window_size
  
  data.series = dataset[(i - window_size): (i - 1),1]
  
  # Normalize entire dataset to have variance one
  return.sd = apply(data.series, 2, sd)
  data.series = data.series$close / return.sd
  next.return = dataset[i] / return.sd
  date = dates[i]
  # Fit model
  if (FALSE){#(count%%24 == 0 || count == 1){
    #print("Refit")
    fitted.model = garchFit(formula = ~arma(3,1) + garch(1,2),
                            data = data.series,
                            cond.dist = "QMLE", 
                            trace = FALSE)
    
    fitted.model = garchFit(formula = ~arma(3,1) + garch(1,2),
                            data = data.series,
                            cond.dist = "QMLE", 
                            trace = FALSE)
    # Predict next value
    model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
    
    fcast = c(fcast, tmp)
    for (j in 1:10){
      tmp = ugarchforecast(specf, dataset$close[1:(i+j - 1)], n.ahead = 1)
      fcast = c(fcast, tmp)
    }
    
    #print(model.forecast)
    model.mean = model.forecast$meanForecast #serie
    model.sd = model.forecast$standardDeviation #volatility
    
    # get standardized residuals
    est_sigma = fitted.model@sigma.t
    stdres = data.series / est_sigma
    #Fit gpd to residuals over threshold
    # Determine threshold
    
    
    prediction[count, 1] = date
    prediction[count, 2] = next.return
    prediction[count, 3] = return.sd
    
    for (j in 1:length(qs)){
      q = qs[j]
      # EVTmodel.threshold = quantile(data.series, q)
      # k = sum(data.series >= EVTmodel.threshold)
      k = length(stdres) * (1-q)
      EVTmodel.threshold = (sort(stdres, decreasing = TRUE))[(k+1)]
      # Fit GPD to residuals
      EVTmodel.fit = gpd.fit(xdat = stdres,
                             threshold = EVTmodel.threshold, 
                             npy = NULL, 
                             show = FALSE)
      # Extract scale and shape parameter estimates
      EVTmodel.scale = EVTmodel.fit$mle[1]
      EVTmodel.shape = EVTmodel.fit$mle[2]
      # Estimate quantiles
      Nu = EVTmodel.fit$nexc
      EVTmodel.zq = gdp.quantile(q, EVTmodel.threshold, EVTmodel.scale, EVTmodel.shape, n, Nu)
      # Predict return value
      predicted_value_mean = model.mean + model.sd * EVTmodel.zq
      # predicted_value = model.sd * EVTmodel.zq
      
      predicted_norm = var.normal(mean=model.mean, sd=model.sd, probs=q)
      prediction[count, ((j-1) * length(qs) + 4)] = predicted_value_mean
      prediction[count, ((j-1) * length(qs) + 5)] = predicted_norm
    }
  }
  fitted.model = garchFit(formula = ~arma(3,1) + garch(1,2),
                          data = data.series,
                          cond.dist = "QMLE", 
                          trace = FALSE)
  # Predict next value
  model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
  model.mean = model.forecast$meanForecast #serie
  model.sd = model.forecast$standardDeviation #volatility
  
  # get standardized residuals
  est_sigma = fitted.model@sigma.t
  stdres = data.series / est_sigma
  #Fit gpd to residuals over threshold
  # Determine threshold
  
  
  prediction[count, 1] = date
  prediction[count, 2] = next.return
  prediction[count, 3] = return.sd
  
  for (j in 1:length(qs)){
    q = qs[j]
    # EVTmodel.threshold = quantile(data.series, q)
    # k = sum(data.series >= EVTmodel.threshold)
    k = length(stdres) * (1-q)
    EVTmodel.threshold = (sort(stdres, decreasing = TRUE))[(k+1)]
    # Fit GPD to residuals
    EVTmodel.fit = gpd.fit(xdat = stdres,
                           threshold = EVTmodel.threshold, 
                           npy = NULL, 
                           show = FALSE)
    # Extract scale and shape parameter estimates
    EVTmodel.scale = EVTmodel.fit$mle[1]
    EVTmodel.shape = EVTmodel.fit$mle[2]
    # Estimate quantiles
    Nu = EVTmodel.fit$nexc
    EVTmodel.zq = gdp.quantile(q, EVTmodel.threshold, EVTmodel.scale, EVTmodel.shape, n, Nu)
    # Predict return value
    predicted_value_mean = model.mean + model.sd * EVTmodel.zq
    # predicted_value = model.sd * EVTmodel.zq
    
    predicted_norm = var.normal(mean=model.mean, sd=model.sd, probs=q)
    prediction[count, ((j-1) * 2 + 4)] = predicted_value_mean
    prediction[count, ((j-1) * 2 + 5)] = predicted_norm
  }
  count = count + 1
}


data_pred = prediction[,-1]
mode(data_pred) = "double"
df = data.frame(data_pred)
dates = prediction[,1]
colnames(df) = c("returns", "sd", "evt_var_10%", "var_10%", "evt_var_5%", "var_5%", "evt_var_2.5%", "var_2.5%", "evt_var_1%", "var_1%" )
rownames(df) = dates
time <- Sys.time()
path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
write.csv(df, paste0(path, "_prediction.csv"), row.names = TRUE)


save = TRUE
time <- Sys.time()
path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
write.csv(prediction, paste0(path, "_prediction.csv"), row.names = TRUE)

if (save){
  
  data_pred = prediction[,c(-1, -6,-9)]
  mode(data_pred) = "double"
  df = data.frame(data_pred)
  dates = prediction[,1]
  colnames(df) = c("returns", "sd", "evt_var_5%", "var_5%", "evt_var_2.5%", "var_2.5%", "evt_var_1%", "var_1%" )
  rownames(df) = dates
  time <- Sys.time()
  path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
  write.csv(df, paste0(path, "_prediction.csv"), row.names = TRUE)
}

# Some plots
max_ = max(df)
min_ = min(df)
plot(df[, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[, "evt_var_5%"], col = 'red')
lines(df[, "var_5%"], col = 'green')

plot(df[, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[, "evt_var_2.5%"], col = 'red')
lines(df[, "var_2.5%"], col = 'green')


plot(df[, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[, "evt_var_1%"], col = 'red')
lines(df[, "var_1%"], col = 'green')

# plot losses only
mask = df[,"returns"] >= 0
max_ = max(df[mask,])
min_ = 0
plot(df[mask, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[mask, "evt_var_5%"], col = 'red')
lines(df[mask, "var_5%"], col = 'green')

plot(df[mask, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[mask, "evt_var_2.5%"], col = 'red')
lines(df[mask, "var_2.5%"], col = 'green')


plot(df[mask, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[mask, "evt_var_1%"], col = 'red')
lines(df[mask, "var_1%"], col = 'green')

# Some signals, classification metrics
vol_long = apply(df, 2, diff) > 0
vol_long = as.matrix(vol_long)
mode(vol_long) =  "integer"

Precision(vol_long[,1], vol_long[,3], positive = 1)
Recall(vol_long[,1], vol_long[,3], positive = 1)

q = 0.95
dataset.sd = sd(dataset)
t = quantile(dataset/dataset.sd, q)

vol_jump = as.matrix(df > 0)
mode(vol_jump) <- "integer"

max_ = max(df)
min_ = min(df)
plot(df[, "returns"], type = 'l', ylim = c(min_,max_))
lines(df[, "evt_var_5%"], col = 'red')
lines(df[, "var_5%"], col = 'green')
lines(rep(t, nrow(prediction)), col = "blue")

Precision(vol_jump[,1], vol_jump[,3], positive = 1)
Recall(vol_jump[,1], vol_jump[,3], positive = 1)

Precision(vol_jump[,1], vol_jump[,4], positive = 1)
Recall(vol_jump[,1], vol_jump[,4], positive = 1)

signal_pred_jump =  prediction[, 2] <= t
mode(signal_pred_jump) <- "integer"
losses_evt = (-prediction[, 1] * signal_pred_jump) * dataset.sd

signal_pred_jump =  prediction[, 3] <= t
mode(signal_pred_jump) <- "integer"
losses = (-prediction[, 1] * signal_pred_jump) * dataset.sd

bh = cumprod(-prediction[, 1]* dataset.sd + 1)
plot(bh, type = 'l', ylim = c(min(bh), max(bh)))
lines(cumprod(losses_evt + 1), col = 'red')
lines(cumprod(losses + 1), col = 'green')

# VAR-short strategy:
mask_trade =  prediction[, 2] <= t
signal_pred_jump = mask_trade
mode(signal_pred_jump) <- "integer"
gain_evt = prediction[mask_trade, 1] * dataset.sd
plot(gain_evt, type = 'l')

plot(cumprod( -prediction[, 1] * dataset.sd + 1))
plot(cumprod(prediction[, 1] * dataset.sd * vol_jump[,2] + 1), type = 'l')

Precision(head(vol_jump[,1], 100), head(vol_jump[,3], 100), positive = 1)

