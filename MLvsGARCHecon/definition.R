# VaR for a Generalized Pareto Distribution (GDP)
var.normal = function(probs, mean, sd)
{
  var = mean + sd * qnorm(p = probs)
  return(var)
}

# VaR for a Student t distribution
var.student = function(probs, mean, sd, df)
{
  scaling.factor = sqrt((df-2) / df)
  var = mean + sd * (scaling.factor * qt(p = probs, df = df) )
  return(var)
}

# VaR for a Generalized Pareto Distribution (GDP)
var.gpd = function(probs, threshold, scale, shape, n, Nu)
{
  var = threshold + (scale / shape) * (((n / Nu) * (1 - probs)) ^ (-shape) - 1)
  return(var)
}

# ES for a normal distribution
es.normal = function(probs, mean, sd)
{
  es = mean + sd * (dnorm(x=qnorm(p=probs)) / (1-probs))
  return(es)
}

# ES for a Student t distribution
es.student = function(probs, mean, sd, df)
{
  scaling.factor = sqrt((df-2)/df)
  factor1 = dt(x=qt(p=probs, df=df), df=df) / (1-probs)
  factor2 = (df + (qt(p=probs, df=df))^2 ) / (df-1)
  es = mean + sd * scaling.factor * factor1 * factor2
  return(es)
}

# ES for a GPD
es.gpd = function(var, threshold, scale, shape)
{
  es = var / (1-shape) + (scale - shape * threshold) / (1-shape)
  return(es)
}


# Tail proba for a GPD
tail.gpd = function(x, threshold, scale, shape, n, Nu)
{
  proba = Nu/n * ( 1 + shape * (x - threshold)/scale )^(-1/shape)
  return(proba)
}