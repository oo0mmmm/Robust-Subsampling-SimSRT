# Robust-Subsampling-SimSRT
## Implementation codes for SimSRT algorithm. 

In Section 2.3 (SimSRT) of the main article, we show that by a carefully chosen ratio. Robustness can be obtained following the classic ERM training pipeline with a specifically selected subset of full dataset. The proportion of random subsample to the desired total subsample size is $1/(1+\rho)$, and consequently the size of uniform subsample is given by $n\rho/(1+\rho)$ where $n$ denotes the desired total subsample size. **Therefore, SimSRT can regarded as a pure data subsampling tool that works in the stage of data preparation without modifying the codes in training pipeline.** File `SimSRT.ipynb` offers one way to select robust subsample via uniform design method and one-nearest-neighbor approximation. Running function `SimSRT(data, design, n, $\rho$)` will return a list of indices corresponding to uniform subsample and random subsample of the original full dataset given by `data`.


## Examples

We give our training codes for experiments in the main article.




