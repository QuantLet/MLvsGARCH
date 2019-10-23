# install and load packages
libraries = c( "timeSeries", "forecast", "fGarch", "bsts", "rugarch", "caret") #"tseries2
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# load dataset

get_returns = function(path){
  data = read.csv(file=path, header=TRUE, sep=",", dec=".")
  close = data[1:nrow(data),"close"]
  dates = c(levels(data[1:nrow(data),"X"]))
  dates = as.POSIXct(dates, format="%Y-%m-%d %H:%M:%S", tz = "GMT")
  ret = data.frame("dates" = dates[2:length(dates)], "close" = diff(log(close)))
  ret = na.omit(ret)
  
  ret = timeSeries::as.timeSeries(data.frame("close"=ret$close,
                                             row.names=ret$dates),
                                  FinCenter = "GMT")
  
  ret = na.omit(ret)
  
  return (ret)
}


rolling_forecast = function(train_ret, test_ret, test_dates, armaOrder, every = 24, 
                            distribution = 'sstd', save_path=NULL){
  # Initialization
  predsigma = c()
  predfitted = c()
  n_train = length(train_ret)
  close = c(train_ret, test_ret)
  length_test = length(test_ret)
  # Specify model: EGARCH(1,2) 
  spec = ugarchspec(mean.model = list(armaOrder = armaOrder),
                    variance.model = list(model = 'eGARCH',
                                          garchOrder = c(1,2)), 
                    distribution = distribution)
  print('Fit model')
  fit = ugarchfit(spec, close[1:n_train])
  list_coefs = as.list(coef(fit))
  # create a specification with fixed parameters:
  specf = ugarchspec(mean.model = list(armaOrder = armaOrder),
                     variance.model = list(model = 'eGARCH',
                                           garchOrder = c(1,2)), 
                     distribution = distribution, fixed.pars =  list_coefs)
  
  # we will also create the closed form forecast
  afor = matrix(NA, ncol = length_test, nrow = 2)
  rownames(afor) = c('Mu', 'Sigma')
  colnames(afor) = test_dates #paste('T+', 1:length_test, sep = '')
  # T+1 we can use ugarchsim:
  tmp = ugarchforecast(fit, n.ahead = 1)
  afor[, 1] = c(fitted(tmp), sigma(tmp))
  # for T+(i>1):
  for (i in 2:length_test) {
    if (i%%100 == 0){
      print(length_test - i)
    }
    tmp = ugarchforecast(specf, close[1:(n_train + i - 1)], n.ahead = 1)
    afor[, i] = c(fitted(tmp), sigma(tmp))
  }
  #plot(abs(close[(n_train + 1):(n_train + length_test)]), type = 'l')
  #lines(afor[2,1:ncol(afor)], type='l', col = 'red')
  print('saving')
  write.csv(t(afor), paste0(save_path, '.csv'), row.names = TRUE)
  save(afor, file=paste0(save_path, '.RData'))
  write.csv(list_coefs, paste0(save_path, '_coef.csv'), row.names = FALSE)
  
  return_ = list(afor, fit, list_coefs)
  names(return_) <- c("afor", "fit", "coefs")
  return (return_)
  
}


cv_prediction = function(ret, cv, armaOrder, distribution, every, n_train = NULL, save = FALSE, comments = NULL){
  dates = as.POSIXct(row.names(ret), format="%Y-%m-%d %H:%M:%S", tz = "GMT")
  results = c()
  time <- Sys.time()
  filepath = file.path("./saved_models", gsub(' ', '', gsub('-', '', gsub(':', '', time))))
  filepath = paste0(paste0(filepath, '_'), comments)
  print(filepath)
  ifelse(!dir.exists("./saved_models/"), dir.create("./saved_models/"), FALSE)
  ifelse(!dir.exists(filepath), dir.create(filepath), FALSE)

  for (i in seq(0, (length(cv) - 1))) {
    print(i)
    start_time <- Sys.time()
    print(
      paste('FIT TO GO', as.integer(( length(cv) - (i+1) )) ) 
    )
    cv_i = paste0('cv_', i)
    train_dates_i = as.POSIXct(cv[[cv_i]]$train, format="%Y-%m-%d %H:%M:%S", tz = "GMT")
    train_ret = ret[dates %in% train_dates_i]
    
    if (!is.null(n_train)){
      train_ret = train_ret[(length(train_ret) - n_train + 1): length(train_ret)] 
    }
    print(length(train_ret))
    test_dates_i = as.POSIXct(cv[[cv_i]]$date_test, format="%Y-%m-%d %H:%M:%S", tz = "GMT")
    test_ret = ret[dates %in% test_dates_i]

    savepath = paste0(paste0(filepath, '/cv'), i)
    print(savepath)
    returns_ = rolling_forecast(train_ret,
                                test_ret,
                                test_dates_i,
                                armaOrder = armaOrder, 
                                every = every, 
                                distribution = distribution, 
                                save_path=savepath)
    end_time <- Sys.time()
    print(end_time - start_time)
  }
  
  if (save){
    results = c(results, returns_)
    save(results,
         file =  paste0(filepath, '/results.RData')
    )
  }
  return (results)
}