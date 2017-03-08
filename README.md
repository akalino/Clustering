# Clustering
Shifting from an R based system to Python necessitated the re-writing
of some packages and algorithms we needed that did not have great Python
support. One of these algorithms was CLARA clustering. The CLARA
clustering does well on large data sets such as those we face in
clustering and was shown to outperform similar methods but to its
focus on k-medoids measures rather than mean measures. The k-medoids
algorithm differs from the traditional k-means algorithm in that it
chooses data points from the input set as cluster centers rather than
computing the center as a mean of data points. Choosing individual data
points as centers allows for a mode rigid structure that is less prone
to drift and in some instances can form clusters with a tighter fit.
The CLARA algorithm extends the k-medoids algorithm by selecting
potentially optimal data points from small samples of the data set.
This pre-computation of potential centers allows for a reduction in
complexity of the medoid search and provides both speed and scalability
to the algorithm. In addition to algorithmic speed-ups from the smaller
sized sampling, additional speed-ups have been added to our internal
implementation of the CLARA algorithm. Due to the sub-sampling, there
are cases where distances could potentially be re-computed. To avoid
this duplication, distances for each point are cached in a dictionary
after they are computed.

In addition to CLARA, there were several input variables that required
Bayesian estimation. The BayesEstimates class initializes estimation parameters,
applied the binomial beta distribution and runs the maximum likelihood estimator.
The bayesian approach allows for both data normalization and imputation of missing variables
needed for the CLARA algorithm.

Future hopes are to submit a pull request to add the CLARA algorithm to sklearn.