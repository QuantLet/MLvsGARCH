# VaR for a Generalized Pareto Distribution (GDP)
var.normal = function(probs, mean, sd)
{
  var = mean + sd * qnorm(p = probs)
  return(var)
}

# VaR for a Generalized Pareto Distribution (GDP)
var.gpd = function(probs, threshold, scale, shape, n, Nu)
{
  var = threshold + (scale / shape) * (((n / Nu) * (1 - probs)) ^ (-shape) - 1)
  return(var)
}

fit_predict = function(data.series,
                       next.return.std,
                       return.sd,
                       model="garch",
                       distribution="QMLE", 
                       arma_order=c(0,0), 
                       garch_order=c(1,0),
                       q_fit = 0.1, 
                       qs = c(0.1)) {
  
  #######################################################################
  ### Check that model and distribution are allowed
  valid.models = c("garch", "gjr", "fam", "cs")
  valid.distributions = c("norm", "std", "QMLE")
  if (!(model %in% valid.models))
  {
    stop("Invalid model chosen")
  }
  if (!(distribution %in% valid.distributions))
  {
    stop("Invalid conditional distribution chosen")
  }
  #######################################################################
  
  
  
  #######################################################################
  # Fit models and extract needed parameters (and residuals)
  
  ##########################################################
  ####### Use package fGarch for plain garch model
  
  
  formula = substitute(~arma(p,q) + garch(a,b), 
                       list(p=arma_order[1],
                            q=arma_order[2],
                            a=garch_order[1],
                            b=garch_order[2]))
  
  if ((model == "garch"))
  {
    if (distribution == "norm")  # Normal innovations
    {
      fitted.model = garchFit(formula = formula,
                              data = data.series, 
                              cond.dist = "norm", 
                              trace = FALSE)
    }
    
    if (distribution == "std")  # Student t innovations
    {
      fitted.model = garchFit(formula = formula,
                              data = data.series, 
                              cond.dist = "std", 
                              shape = df,
                              include.shape = FALSE,
                              trace = FALSE)
    }
    
    if (distribution == "QMLE")  # QMLE estimation
    {
      fitted.model = garchFit(formula = formula,
                              data = data.series,
                              cond.dist = "QMLE", 
                              trace = FALSE)
    }
    
    # Produce forecasts of mean and standard deviation
    model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
    
    
    model.mean = model.forecast$meanForecast
    model.sd = model.forecast$standardDeviation
    
    # Get residuals (for EVT): standardize through (time dependent) fitted values
    # and standard deviations
    # model.residuals = fGarch::residuals(fitted.model, standardize=TRUE)
    est_sigma = fitted.model@sigma.t
    stdres = data.series / est_sigma
  } else {
    
    ##########################################################
    ####### Use package rugarch for other models
    
    # AR(1) - GJR-GARCH(1,1) model
    if (model == "gjr")
    {
      if (distribution == "norm")  # Normal innovations
      {
        fitted.model = ugarchfit(spec=gjr.spec.norm, 
                                 data=data.series, 
                                 solver=slvr, 
                                 solver.control=slvr.ctrl)
      }
      if (distribution == "std")  # Student t innovations
      {
        fitted.model = ugarchfit(spec=gjr.spec.std, 
                                 data=data.series, 
                                 solver=slvr, 
                                 solver.control=slvr.ctrl)
      }
      
    }
    # AR(1) - component-GARCH(1,1) model
    if (model == "cs")
    {
      if (distribution == "norm")  # Normal innovations
      {
        fitted.model = ugarchfit(spec=cs.spec.norm, 
                                 data=data.series, 
                                 solver=slvr, 
                                 solver.control=slvr.ctrl)
      }
      if (distribution == "std")  # Student t innovations
      {
        fitted.model = ugarchfit(spec=cs.spec.std, 
                                 data=data.series, 
                                 solver=slvr, 
                                 solver.control=slvr.ctrl)
      }
    }
    
    # Make forecasts of tomorrow's expected value and standard deviation 
    # for models from rugarch package
    model.forecast = ugarchforecast(fitted.model, n.ahead=1)
    model.mean = model.forecast@forecast$forecasts[[1]]$series
    model.sd = model.forecast@forecast$forecasts[[1]]$sigma
    
    # Get residuals (for EVT): standardize through (time dependent) fitted values
    # and standard deviations
    # model.residuals = rugarch::residuals(fitted.model, standardize=TRUE)
    est_sigma = fitted.model@sigma.t
    stdres = data.series / est_sigma
  }
  
  #######################################################################
  # Now calculate VaR, ES, VaR-break, ES difference, and excess residuals
  
  ############################################################
  # Peak-Over-Threshold estimates
  prediction_i = c(date, next.return.std[1], return.sd[1], model.mean, model.sd)#, model.var, model.break)
  
  
  for (j in 1:length(qs)) {
    q = qs[j]
    k = length(stdres) * (1 - q_fit) # Determine threshold: fit GPD on q_fit of the data
    EVTmodel.threshold = (sort(stdres, decreasing = TRUE))[(k + 1)]
    # Fit GPD to residuals
    EVTmodel.fit = gpd.fit(
      xdat = stdres,
      threshold = EVTmodel.threshold,
      npy = NULL,
      show = FALSE
    )
    
    ############################################################
    # Base model estimates
    
    if (distribution == "norm" || distribution == "QMLE")
    {
      model.var = var.normal(q, mean=model.mean, sd=model.sd)
      
      #model.es = es.normal(mean=model.mean, sd=model.sd, probs=qs)
      
    }
    
    if (distribution == "std")
    {
      model.var = var.student(q, mean=model.mean, sd=model.sd, df=df)
      #model.es = es.student(mean=model.mean, sd=model.sd, probs=qs, df=df)
    }
    
    # VaR-break
    model.break = (next.return.std > model.var)
    
    # Difference between actual loss and ES estimate
    #model.diff = (next.return.std - model.es)
    # Excess residuals (page 294, McNeil-Frey) 
    #model.exres = model.diff / model.sd
    
    
    # EVT ESTIMATES
    
    # Extract scale and shape parameter estimates
    EVTmodel.scale = EVTmodel.fit$mle[1]
    EVTmodel.shape = EVTmodel.fit$mle[2]
    # Estimate quantiles
    Nu = EVTmodel.fit$nexc
    EVTmodel.zq = var.gpd(q, EVTmodel.threshold, EVTmodel.scale, EVTmodel.shape, length(data.series), Nu)
    # Calculate VaR
    EVTmodel.var = model.mean + model.sd * EVTmodel.zq
    # Calculate the Expected Shortfall
    # EVTmodel.es = model.mean + model.sd * es.gpd(var=EVTmodel.zq, threshold=EVTmodel.threshold, scale=EVTmodel.scale, shape=EVTmodel.shape)
    # VaR-break
    EVTmodel.break = (next.return.std > EVTmodel.var)
    
    # Difference between actual loss and ES estimate
    #EVTmodel.diff = (next.return.std - EVTmodel.es)
    
    # Exceedance residuals (page 294, McNeil-Frey)
    # Note that these are created for all observations;
    # only those on dates of VaR-breaks should be used
    # in the bootstrap test
    #EVTmodel.exres = EVTmodel.diff / model.sd
    
    q_data = c(model.var, model.break, EVTmodel.threshold, EVTmodel.var, EVTmodel.break, EVTmodel.zq)
    prediction_i = c(prediction_i, q_data)
  }
  
  return (prediction_i)
}