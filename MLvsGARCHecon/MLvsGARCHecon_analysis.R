rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c("rugarch", "FinTS")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# load dataset
path = "../data/btc_1H_train_0.csv"
data = read.csv(file=path, header=TRUE, sep=",", dec=".")
close = data[1:nrow(data),"close"]
date1 = as.Date(c(levels(data[1:nrow(data),"X"])))
# take log returns
ret = diff(log(close))
ret = na.omit(ret)
# plot of btc return
plot(date1[2:length(date1)], ret, type = 'l', xlab='time', ylab='btc log returns')

par(mfrow = c(1, 2))
# histogram of returns
hist(ret, col = "grey", breaks = 40, freq = FALSE, xlab = NA)
lines(density(ret), lwd = 2)
mu = mean(ret)
sigma = sd(ret)
x = seq(-4, 4, length = 100)
curve(dnorm(x, mean = mu, sd = sigma), add = TRUE, col = "darkblue", 
      lwd = 2)
# qq-plot
par(pty="s") 
qqnorm((ret - mu)/ sigma, xlim = c(-15,15), ylim = c(-15,15), main = NULL)
qqline((ret - mu)/ sigma)

# Fit ARIMA model
order = c(3, 0, 1) # arimaorder(fit)
ARIMAfit <- arima(ret, order = order)
summary(ARIMAfit)

# vola cluster
par(mfrow = c(1, 1))
res = ARIMAfit$residuals
res2 = ARIMAfit$residuals^2
plot(res, ylab = NA, type = 'l')
plot(res2, ylab='Squared residuals', main=NA)

par(mfrow = c(1, 2))
acfres2 = acf(res2, main = NA, lag.max = 20, ylab = "Sample Autocorrelation", 
              lwd = 2)
pacfres2 = pacf(res2, lag.max = 20, ylab = "Sample Partial Autocorrelation", 
                lwd = 2, main = NA)

# arch effect
res = ARIMAfit$residuals
ArchTest(res)  #library FinTS
Box.test(res2, type = "Ljung-Box")

# We reject null hypothesis of both Archtest and Ljung-Box => autocorrelation in the squared residuals

# EtGarch

#fit the rugarch eGarch model with skew student distribution
spec = ugarchspec(mean.model = list(armaOrder = c(3,1)),
variance.model = list(model = 'eGARCH',
garchOrder = c(1,2)),
distribution = 'sstd')
essgarch12 <- ugarchfit(spec, ret, solver = 'hybrid')

# qq plot
par(pty="s")
plot(essgarch12, which = 9)#, xlim = c(-15,15))


# To control plot param need to call qdist and .qqLine
zseries = as.numeric(residuals(essgarch12, standardize=TRUE))
distribution = essgarch12@model$modeldesc$distribution
idx = essgarch12@model$pidx
pars  = essgarch12@fit$ipars[,1]
skew  = pars[idx["skew",1]]
shape = pars[idx["shape",1]]
if(distribution == "ghst") ghlambda = -shape/2 else ghlambda = pars[idx["ghlambda",1]]

par(mfrow = c(1, 1), pty="s") 
n = length(zseries)
x = qdist(distribution = distribution, lambda = ghlambda, 
      skew = skew, shape = shape, p = ppoints(n))[order(order(zseries))]
plot(x, zseries,  ylim = c(-15,15), ylab="Sample Quantiles", xlab="Theoretical Quantiles")
rugarch:::.qqLine(y = zseries, dist = distribution, datax = TRUE,  lambda = ghlambda, 
                  skew = skew, shape = shape)



