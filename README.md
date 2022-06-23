# You Only Derive Once (YODO): Automatic Differentiation for Efficient Sensitivity Analysis in Bayesian Networks

Check out our paper [here](https://arxiv.org/abs/2206.08687).

Sensitivity analysis measures the influence of a Bayesian network's parameters on a quantity of interest defined by the network, such as the probability of a variable taking a specific value. In particular, the so-called sensitivity value measures the quantity of interest's partial derivative with respect to the network's conditional probabilities. However, finding such values in large networks with thousands of parameters can become computationally very expensive.

YODO uses automatic differentiation combined with exact inference to obtain all sensitivity values in a single pass. Our method first marginalizes the whole network once using e.g. variable elimination and then backpropagates this operation (using PyTorch) to obtain the gradient with respect to all input parameters. Doing this, one can rank all parameters by importance in a few seconds at most:

<p align="center"><img src="https://github.com/rballester/yodo/blob/main/images/plot_example.jpg" width="600" title="Example visualization"></p>

See this [notebook](https://github.com/rballester/yodo/blob/main/example.ipynb) for an example that visualizes the most influential parameters of the [*hailfinder* Bayesian network](https://www.bnlearn.com/bnrepository/discrete-large.html#hailfinder).

**Main dependences**:

- [*NumPy*](https://numpy.org/)
- [*pgmpy*](https://github.com/pgmpy/pgmpy) (for reading networks and moralizing Bayesian networks)
- [*PyTorch*](https://pytorch.org/) (as numerical and autodiff backend)
- [*opt_einsum*](https://github.com/dgasmith/opt_einsum) (for efficient marginalization)
- [*gmtorch*](https://github.com/rballester/gmtorch) (for operations with Markov random fields)

## Citation

If you use this package, please cite our paper:

```
@misc{BL:22,
  doi = {10.48550/ARXIV.2206.08687},
  url = {https://arxiv.org/abs/2206.08687},
  author = {Ballester-Ripoll, Rafael and Leonelli, Manuele},
  keywords = {Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {You Only Derive Once ({YODO}): Automatic Differentiation for Efficient Sensitivity Analysis in {B}ayesian Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
