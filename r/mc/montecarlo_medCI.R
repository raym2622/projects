n = c(50, 200, 600) # 3 sample sizes
mcrep = 200   # specify the num of replications

# calc the CI for bootstrapped samples (resample 1000* from a sample of size n)
cinterval_boot = function(x, n_boot = 1000) {
  mat = sample(x, n_boot * length(x), replace = TRUE)   # bootstrap
  dim(mat) = c(n_boot, length(x))
  mn = apply(mat, 1, median)     # calc the medians for each row
  lcb_boot = quantile(mn, 0.025)  # lower bound using the quantile func
  ucb_boot = quantile(mn, 0.975)  # upper bound using the quantile func
  return(c(lcb_boot, ucb_boot))
}
# coverage prob and average CI width for bootstrapped samples (200 samples)
quant_ci_boot = function(xmat, target, mcrep, n_boot = 1000) {
  m = qnorm(1-0.05/2)                    # 95% CI multiplier
  boot = apply(xmat, 1, cinterval_boot)  # apply the func above to each sample
  lcb_boot = boot[1,1:mcrep]             # lower bounds
  ucb_boot = boot[2,1:mcrep]             # upper bounds
  meanwidth = mean(ucb_boot - lcb_boot)  # mean CI width
  cvrg_prob_boot = mean((lcb_boot < target) & (ucb_boot > target)) # cvrg prob
  return(c(cvrg_prob_boot, meanwidth))
}

# vectorization and robust estimator for CI's and cvrg prob of all reps
quant_ci_robust = function(xmat, target, mcrep) {
  mn = apply(xmat, 1, median)       # sample medians for each rep
  m = qnorm(1-0.05/2)
  mads = apply(xmat, 1, mad)        # median abs devi for each row
  lcb_robust = mn - m * mads / sqrt(ncol(xmat))   # lower bound
  ucb_robust = mn + m * mads / sqrt(ncol(xmat))   # upper bound
  meanwidth = mean(ucb_robust - lcb_robust)       # mean CI width
  cvrg_prob_robust = mean((lcb_robust < target) & (ucb_robust > target))
  return(c(cvrg_prob_robust, meanwidth))
}

# Gamma
median_gamma = qgamma(0.5, shape = 3, rate = 1)  # true median for gamma dist
for (i in n) {
  set.seed(111)
  x = rgamma(i*mcrep, shape = 3, rate = 1)  # draw samples from gamma 
  dim(x) = c(mcrep, i)          # reshape the sample matrix
  trans1 = t(as.matrix(quant_ci_boot(x, median_gamma, mcrep))) # bootstrap
  trans2 = t(as.matrix(quant_ci_robust(x, median_gamma)))  # robust method
  # format the results
  rownames(trans1) = paste("sample size", i, ":", sep = " ")
  colnames(trans1) = c("cvrg_prob_boot", "mean_width")
  colnames(trans2) = c("cvrg_prob_robust", "mean_width")
}


# Exponential 
median_exp = qexp(0.5)  # true median for exp dist
for (i in n) {
  set.seed(111)
  x = rexp(i*mcrep)      # draw samples from exp dist
  dim(x) = c(mcrep, i)   # reshape the sample matrix
  trans1 = t(as.matrix(quant_ci_boot(x, median_exp, mcrep))) # bootstrap
  trans2 = t(as.matrix(quant_ci_robust(x, median_exp)))    # robust method
  # format the results
  rownames(trans1) = paste("sample size", i, ":", sep = " ")
  colnames(trans1) = c("cvrg_prob_boot", "mean_width")
  colnames(trans2) = c("cvrg_prob_robust", "mean_width")
  print(cbind(trans1, trans2))
}


# Std Normal
median_norm = qnorm(0.5)  # true median for the std normal dist
for (i in n) {
  set.seed(111)
  x = rnorm(i*mcrep)          # draw samples from std norm dist
  dim(x) = c(mcrep, i)        # reshape the sample matrix
  trans1 = t(as.matrix(quant_ci_boot(x, median_norm, mcrep))) # bootstrap
  trans2 = t(as.matrix(quant_ci_robust(x, median_norm)))   # robust method
  # format the results
  rownames(trans1) = paste("sample size", i, ":", sep = " ")
  colnames(trans1) = c("cvrg_prob_boot", "mean_width")
  colnames(trans2) = c("cvrg_prob_robust", "mean_width")
  print(cbind(trans1, trans2))
}