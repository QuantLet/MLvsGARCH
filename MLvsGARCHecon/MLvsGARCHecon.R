rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c( "rjson", "timeSeries")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

source('./core.R')

hour = 1
day = hour*24
week = day*7
month = week*4


####### data
ret = get_returns("../data/btc_1H_20160101_20190101.csv")

####### model parameters
distribution = 'sstd'
armaOrder = c(3, 1)
n_train = 6*month #length(ret$close[(ret$dates <=cv[1])])
comments = paste0('FINAL')

####### cross validation folds
cv <- fromJSON(file = "../MLvsGARCHml/saved_models/12072019-143851/global_dates.json")
nb_cv = length(cv)
####### Refit frequency of GARCH model
every = nb_cv

start_time_total <- Sys.time()
results = cv_prediction(ret, cv, armaOrder, distribution, every,
                        n_train = n_train, save = FALSE, comments = comments)
end_time_total <- Sys.time()
print('Total time:')
print(end_time_total - start_time_total)

