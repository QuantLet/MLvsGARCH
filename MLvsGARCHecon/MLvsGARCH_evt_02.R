rm(list = ls(all = TRUE))
graphics.off()

set.seed(10)

library(POT)
library(ismev)
library(fGarch)
library(MLmetrics)
source("./definition.R")

# Constants
day = 24
month = day*30

TEST = TRUE

threshold = 

fit_pred = function() {
  fitted.model = garchFit(
    formula = ~ arma(3, 1) + garch(1, 2),
    data = data.series,
    cond.dist = "QMLE",
    trace = FALSE
  )
  # Predict next value
  model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
  model.mean = model.forecast$meanForecast #serie
  model.sd = model.forecast$standardDeviation #volatility
  
  # get standardized residuals
  est_sigma = fitted.model@sigma.t
  stdres = data.series / est_sigma
  #Fit gpd to residuals over threshold
  # Determine threshold
  
  #prediction[count, 1] = date
  #prediction[count, 2] = next.return
  #prediction[count, 3] = return.sd
  
  prediction_i = c(date, next.return[1], return.sd[1])
  
  
  k = length(stdres) * (1 - q_fit)
  EVTmodel.threshold = (sort(stdres, decreasing = TRUE))[(k + 1)]
  # Fit GPD to residuals
  EVTmodel.fit = gpd.fit(
    xdat = stdres,
    threshold = EVTmodel.threshold,
    npy = NULL,
    show = FALSE
  )
  # Extract scale and shape parameter estimates
  EVTmodel.scale = EVTmodel.fit$mle[1]
  EVTmodel.shape = EVTmodel.fit$mle[2]
  # Estimate quantiles
  Nu = EVTmodel.fit$nexc
  q = 0.90
  # Extract scale and shape parameter estimates
  EVTmodel.scale = EVTmodel.fit$mle[1]
  EVTmodel.shape = EVTmodel.fit$mle[2]
  # Estimate quantiles
  Nu = EVTmodel.fit$nexc
  EVTmodel.zq = var.gpd(q, EVTmodel.threshold, EVTmodel.scale, EVTmodel.shape, n, Nu)
  
  # Calculate the Value-At-Risk
  EVTmodel.var = model.mean + model.sd * EVTmodel.zq
  model.var = var.normal(probs = q,
                         mean = model.mean,
                         sd = model.sd)
  
  # Calculate the Expected Shortfall
  model.es = es.normal(probs=q, mean=model.mean, sd=model.sd)
  EVTmodel.es = model.mean + model.sd * es.gpd(var=EVTmodel.zq,
                                               threshold=EVTmodel.threshold,
                                               scale=EVTmodel.scale,
                                               shape=EVTmodel.shape)
  
  # Calculate proba
  model.proba = pnorm(( lower[1] - model.mean ) / model.sd)
  EVTmodel.proba = pgpd(( lower[1] - model.mean ) / model.sd,
                        loc = EVTmodel.threshold,
                        scale = EVTmodel.scale,
                        shape = EVTmodel.shape)
  
  
  print(c(model.proba, EVTmodel.proba))
  # predicted_value = model.sd * EVTmodel.zq
  #prediction[count, ((j-1) * 2 + 4)] = predicted_value_mean
  #prediction[count, ((j-1) * 2 + 5)] = predicted_norm
  
  q_data = c(EVTmodel.threshold, EVTmodel.var, EVTmodel.es,  EVTmodel.proba, 
             model.var, model.es, model.proba, model.mean , model.sd, EVTmodel.zq)
  
  prediction_i = c(prediction_i, q_data)
  
  
  return (prediction_i)
}

# Load variables and helper functions
dset = "../data/btc_1H_lower_20160101_20190101"
dset.name = paste("./", dset, ".csv", sep = "")
dataset <- read.csv(dset.name, header = TRUE)
dates = dataset$X
dataset = data.frame("close" = dataset$close, "lower" = dataset$lower)
rownames(dataset) = dates
dataset <- timeSeries::as.timeSeries(dataset, FinCenter = "GMT")

# Fit model on one month history
window_size = 4 * month
q_fit = 0.1  # fit to 10% of worst outcomes

# Convert price series to returns
dataset = cbind(dataset, c(NaN, diff(log(dataset$close))))
colnames(dataset) = c("close","lower", "returns")

# Forget about 2016
dataset = dataset[rownames(dataset) >= '2017-05-01 00:00:00', c("returns", "lower")]

length.dataset = length(dataset)
dates = rownames(dataset)

# Split data
test_size = (length(dataset) - window_size)
qs = c(0.90)
prediction = matrix(nrow = test_size, ncol = (3 + (10 * length(qs))))


if (TEST){
  test_size = 20
}

count = 1
time <- Sys.time()
save_path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
for (i in (window_size + 1):(window_size + test_size)) {
  if (i %% 1000 == 0) {
    print((length(dataset) - i))
    print(prediction[(i - window_size - 1), ])
    write.csv(prediction, paste0(save_path, "prediction_online_first.csv"), row.names = FALSE)
    print(head(prediction))
  }
  n = window_size
  
  data.series = dataset[(i - window_size):(i - 1), "returns"]
  # Normalize entire dataset to have variance one
  return.sd = apply(data.series, 2, sd)
  data.series = data.series$returns / return.sd
  next.return = dataset[i] / return.sd
  lower = dataset[i - 1, "lower"]/ return.sd
  date = dates[i]
  # Fit model and get prediction
  if (!TEST){
    prediction_i =  tryCatch(
      
      fit_pred(),
      
      error = function(e)
        base::rep(NA, ncol(prediction)), #function(e) base::print('ERROR'),
      
      silent = FALSE)
  }
  if (TEST){
    prediction_i = fit_pred()
  }
  prediction[count,] = as.vector(prediction_i)
  count = count + 1
}

data_pred = prediction[!is.na(prediction[, 1]),]
dates = data_pred[, 1]
data_pred = data_pred[,-1]
mode(data_pred) = "double"
df = data.frame(data_pred)
cn = c()
for (q in qs){
  cn = c(cn, 
         paste0("threshold_", q),
         paste0("evt_var_", q),
         paste0("evt_es_", q),
         paste0("var_", q),
         paste0("es_", q),
         paste0("mean_", q),
         paste0("sd_", q),
         paste0("zq_", q)
  )
}
colnames(df) = c(
  "std_losses",
  "norm_sd",
  cn
)

rownames(df) = dates


######### SAVE
if (!TEST) {
  write.csv(df, paste0(save_path, "_prediction_10per_proba.csv"), row.names = TRUE)
}