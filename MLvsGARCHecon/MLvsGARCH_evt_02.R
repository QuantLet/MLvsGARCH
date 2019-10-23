rm(list = ls(all = TRUE))
graphics.off()

set.seed(10)

library(POT)
library(ismev)
library(fGarch)
library(MLmetrics)
source("./definition.R")

TEST = FALSE

# Constants
day = 24
month = day*30


fit_pred = function() {
  prediction_i = c(date, next.return[1], return.sd[1])
  fitted.model = garchFit(
    formula = ~ arma(3, 1) + garch(1, 2),
    data = data.series,
    cond.dist = "QMLE",
    trace = FALSE
  )
  # Get standardized residuals
  model.residuals  = fGarch::residuals(fitted.model , standardize = TRUE)
  model.coef = coef(fitted.model)
  # Predict next value
  model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
  model.mean = model.forecast$meanForecast # conditional mean
  model.sd = model.forecast$standardDeviation # conditional volatility

  # Fit gpd to residuals over threshold
  # Determine threshold
  
  EVTmodel.threshold = quantile(model.residuals, (1 - q_fit))
  
  # Fit GPD to residuals
  EVTmodel.fit = gpd.fit(
    xdat = model.residuals,
    threshold = EVTmodel.threshold,
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
  model.proba = pnorm(( lower[1] - model.mean) / model.sd)
  model.proba = 1 - model.proba
  EVTmodel.proba = pgpd(( lower[1] - model.mean) / model.sd,
                       loc = EVTmodel.threshold,
                        scale = EVTmodel.scale,
                        shape = EVTmodel.shape)
  EVTmodel.proba = 1 - EVTmodel.proba
  EVTmodel.proba.est = tail.gpd((lower[1] - model.mean) / model.sd,
                                EVTmodel.threshold, 
                                EVTmodel.scale, 
                                EVTmodel.shape,
                                n, 
                                Nu)
  q_data = c(EVTmodel.threshold, EVTmodel.var, EVTmodel.es,  EVTmodel.proba, EVTmodel.proba.est,
             model.var, model.es, model.proba, model.mean , model.sd, EVTmodel.zq)
  
  prediction_i = c(prediction_i, q_data)
  
  
  return (prediction_i)
}

# Load variables and helper functions
dset = "../data/new_btc_1H_lower_20160101_20190101"
dset.name = paste("./", dset, ".csv", sep = "")
dataset <- read.csv(dset.name, header = TRUE)
dates = dataset$X
dataset = data.frame("close" = dataset$close, "lower" = dataset$lower)
rownames(dataset) = dates
dataset <- timeSeries::as.timeSeries(dataset, FinCenter = "GMT")

# Fit model on 4 months history
window_size = 4 * month
q_fit = 0.2  # fit to 10% of worst outcomes

# Convert price series to losses and lower (negative) threshold to a positive threshold
dataset = cbind(dataset, c(NaN, - diff(log(dataset$close))))
colnames(dataset) = c("close", "lower", "returns")
dataset$lower = - dataset$lower

# Forget about 2016
dataset = dataset[rownames(dataset) >= '2017-01-01 00:00:00', c("returns", "lower")]
# dataset = dataset[rownames(dataset) <= '2018-12-04 00:00:00', c("returns", "lower")]

length.dataset = nrow(dataset)
dates = rownames(dataset)

# Split data
test_size = length.dataset - window_size

qs = c(0.90)
prediction = matrix(nrow = test_size, ncol = (3 + (11 * length(qs))))


if (TEST){
  test_size = 20
}

count = 1
time <- Sys.time()
save_path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
for (i in (window_size + 1):(window_size + test_size)) {
  if ((test_size + window_size - i) %% 250 == 0){
    print(paste0("Steps to go: ", test_size + window_size - i))
  }
  if (i %% 1000 == 0) {
    print(paste("Saving prediction at step", i))
    write.csv(prediction, paste0(save_path, "prediction_online_first.csv"), row.names = FALSE)
  }
  n = window_size
  
  data.series = dataset[(i - window_size):(i - 1), "returns"]
  # Normalize entire dataset to have variance one
  return.sd = apply(data.series, 2, sd)
  data.series = data.series$returns / return.sd
  next.return = dataset[i, "returns"]
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
         paste0("evt_proba_", q),
         paste0("evt_proba_est_", q),
         paste0("var_", q),
         paste0("es_", q),
         paste0("proba_", q),
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
if (TEST){
  write.csv(df, paste0(
    paste0('./',
           save_path),
           "_prediction_10per_proba_TEST.csv"
    ),
  row.names = TRUE)
} else {
  write.csv(df, paste0(
    paste0('./saved_models/',
           save_path), 
           "_prediction_qfit_02.csv"
    ),
  row.names = TRUE)
}
