# AI assisted research

## Stochastic Gradient Langevin Dynamics

The [paper, “Bayesian Learning via Stochastic Gradient Langevin Dynamics”](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf), introduces SGLD, a method that combines stochastic-gradient updates with appropriately scaled Gaussian noise to draw approximate samples from a Bayesian posterior. Its use of minibatches makes posterior inference practical for large datasets, while decreasing the step size over time supports convergence to the target distribution.

[View the SGLD diagnostic figure](sgld/figures/sgld_diagnostics.png).
