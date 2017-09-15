# EM-Algorithm
EM algorithm coded in Python. Algorithms output predicted data, number of iterations until convergence, and the optimized mean vector and covariance matrix.\\
To test the code, I randomly generate 500 observations from a trivariate normal distribution (see code for the mu and sigma used to generate this data). I then randomly insert NAs into the data to be predicted using the EM-algorithm. The data generation was done in Python. I saved this data as a .csv file, which I then loaded into R to use in that version of the algorithm.
