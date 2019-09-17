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

TEST = FALSE

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
  
  for (j in 1:length(qs)) {
    q = qs[j]
    # EVTmodel.threshold = quantile(data.series, q)
    # k = sum(data.series >= EVTmodel.threshold)
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
    
    #EVTmodel.zq = qgpd(q, loc = EVTmodel.threshold, scale = EVTmodel.scale, shape = EVTmodel.shape)
    EVTmodel.zq = var.gpd(q, EVTmodel.threshold, EVTmodel.scale, EVTmodel.shape, n, Nu)
    
    # Predict return value
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

    # predicted_value = model.sd * EVTmodel.zq
    #prediction[count, ((j-1) * 2 + 4)] = predicted_value_mean
    #prediction[count, ((j-1) * 2 + 5)] = predicted_norm
    
    q_data = c(EVTmodel.threshold, EVTmodel.var, EVTmodel.es, model.var, model.es, model.mean , model.sd, EVTmodel.zq)
    
    prediction_i = c(prediction_i, q_data)
  }
  
  return (prediction_i)
}

# Load variables and helper functions
dset = "../data/btc_1H_20160101_20190101"


dset.name = paste("./", dset, ".csv", sep = "")
dataset <- read.csv(dset.name, header = TRUE)
dates = dataset$X
dataset = data.frame("close" = dataset$close)
rownames(dataset) = dates
dataset <- timeSeries::as.timeSeries(dataset, FinCenter = "GMT")


# Fit model on one month history
window_size = 4 * month
q_fit = 0.1  # fit to 10% of worst outcomes

# Convert price series to loss series
dataset = na.omit(diff(log(dataset)))

# Forget about 2016
dataset = dataset[rownames(dataset) >= '2017-05-01 00:00:00', 1]


length.dataset = length(dataset[, 1])
dates = rownames(dataset)



# Split data
test_size = (length(dataset) - window_size)
qs = c(0.90, 0.95, 0.975, 0.99)
prediction = matrix(nrow = test_size, ncol = (3 + (8 * length(qs))))


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
  
  data.series = dataset[(i - window_size):(i - 1), 1]
  
  # Normalize entire dataset to have variance one
  return.sd = apply(data.series, 2, sd)
  data.series = data.series$close / return.sd
  next.return = dataset[i] / return.sd
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
for (q in c('10%', '5%', '2.5%', '1%')){
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
  write.csv(df, paste0(save_path, "_prediction_10per.csv"), row.names = TRUE)
}