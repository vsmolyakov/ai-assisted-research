# AI assisted research

## Stochastic Gradient Langevin Dynamics

The paper [“Bayesian Learning via Stochastic Gradient Langevin Dynamics”](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf), introduces SGLD, a method that combines stochastic-gradient updates with appropriately scaled Gaussian noise to draw approximate samples from a Bayesian posterior. Its use of minibatches makes posterior inference practical for large datasets, while decreasing the step size over time supports convergence to the target distribution.

For a dataset of $N$ observations and a minibatch $B_t$ of size $n$, the update is

$$
\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\left(\nabla_\theta \log p(\theta_t) + \frac{N}{n}\sum_{x_i \in B_t}\nabla_\theta \log p(x_i \mid \theta_t)\right) + \eta_t,\qquad \eta_t \sim \mathcal{N}(0, \epsilon_t I).
$$

Here, $\epsilon_t$ is the step size; the minibatch gradient is scaled by $N/n$, and the Gaussian noise has covariance $\epsilon_t I$.

![SGLD diagnostic plots](sgld/figures/sgld_diagnostics.png)
