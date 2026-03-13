# Robust-Subsampling-SimSRT
## Implementation codes for SimSRT algorithm. 

In Section 2.3 (SimSRT) of the main article, we show that by a carefully chosen ratio. Robustness can be obtained following the classic ERM training pipeline with a specifically selected subset of full dataset. The proportion of random subsample to the desired total subsample size is $1/(1+\rho)$, and consequently the size of uniform subsample is given by $n\rho/(1+\rho)$ where $n$ denotes the desired total subsample size. **Therefore, SimSRT can regarded as a pure data subsampling tool that works in the stage of data preparation without modifying the codes in training pipeline.** File `SimSRT.ipynb` offers one way to select robust subsample via uniform design method and one-nearest-neighbor approximation. We provide all the necessary code for simulation studied and real-world dataset experiments in different sub-folder. For more details, we refer to Method Section in the main article.







