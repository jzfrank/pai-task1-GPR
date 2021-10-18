# pai_task1_blueberry
PAI task 1: Gaussian Process

- The best model so far is the NaiveGP. The larger the N is, the more subsamples we select, the better the model performance. In particular, if N = 4000, we get cost around 20; if N = 8000, we get cost around 12. That is, we could simply pass the task by the naive model with a reasonably good computing resources. If one has a server, we could even let N = 15000, which will give the full model. 
- However, we should not be satisfied with conquering the task with brutal force. So we have tried some other methods. For example, from gpytorch, we have tried [Variational Approximate GPs](https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html), but the result is far from satisfactory. Indeed, the result gives us cost at around 2000! DEBUG NEEDED
- Alternatively, we may propose some other methods to try to improve the result. 
