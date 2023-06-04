import gmtorch as gm
import re
import os
import random
from pathlib import Path
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import pgmpy.readwrite


def yodo(g, probability, given=None):
    """
    For every parameter of an MRF, finds:
        - Its sensitivity value
        - Its vertex proximity
        - Its second derivative
        - The highest derivative (in absolute value) in the interval [0, 1]

    Example:
    >>> g = pgmpy.readwrite.BIFReader("networks/alarm.bif").get_model()
    >>> yodo(g, probability={'CVP': 'HIGH'}, given={'HISTORY': 'TRUE'})

    :param g: a `pgmpy.models.BayesianNetwork`
    :param probability: a dictionary with one key-value pair: {variable: value}
    :param given: [optional] a dictionary for all conditional evidence {variable: value}
    :return: a dictionary {tuple of names: dict}, where dict contains:
        - 'cpt': the original parameters
        - 'derivative'
        - 'sensitivity_value'
        - 'proximity'
        - 'second_derivative'
        - 'largest_first_derivative'
    """

    # Convert state names to integers
    # Example: {'variable': 'high'} could become {'variable': 2}
    probability = {key: g.get_cpds(key).state_names[key].index(probability[key]) for key in probability}
    if given is not None:
        given = {key: g.get_cpds(key).state_names[key].index(given[key]) for key in given}

    g = g.to_markov_model()
    g = gm.from_pgmpy(g).to(torch.float64)
    g.requires_grad_()

    numerator = g.detach().clone()
    numerator.requires_grad_()
    denominator = g.detach().clone()
    denominator.requires_grad_()

    if given is None:
        # Marginal probability case: the function of interest if P(Y_O = y_O)
        numerator.set_evidence(probability, mode='mask')
    else:
        # Conditional probability case: the function of interest if P(Y_O = y_O | y_E = y_E)
        numerator.set_evidence({**probability, **given}, mode='mask')
        denominator.set_evidence(given, mode='mask')

    out_numerator = numerator[[]]
    out_numerator.backward()
    out_denominator = denominator[[]]
    out_denominator.backward()

    result = {}
    for nodes in numerator.factors.keys():
        f = g.factors[nodes].to(torch.float64)
        eps = 1e-9  # CPT entries equal to 0 break the proportional covariation formula, so we perturb them           
        if f.max() > 1-eps:
            f = g.factors[nodes] * (1 - 2*eps) + eps

        def get_coefficients(graph, out):
            """
            Find the coefficients a, b for the straight line y = a*theta + b
            """

            grad = graph.factors[nodes].grad
            if hasattr(graph.factors[nodes], 'mask'):
                grad *= graph.factors[nodes].mask
            num = -torch.sum(grad*f, dim=0, keepdim=True) + grad*f
            denom = 1-f
            grad = grad + num / denom
            a = grad
            b = out.item() - f*a
            return a, b

        # Numerator coefficients (c1 and c2)
        c1, c2 = get_coefficients(numerator, out_numerator)

        # Denominator coefficients (c3 and c4)
        c3, c4 = get_coefficients(denominator, out_denominator)

        # Compute f'(\theta_i)
        derivative = (c1*c4 - c2*c3) / (f*c3 + c4)**2

        # Compute vertex proximity (Der Gaag et al., "Sensitivity analysis of probabilistic networks", 2007)
        s = -c4/c3
        t = c1/c3
        r = c2/c3 + s*t
        vertex = (s < 0)*(s + torch.sqrt(torch.abs(r))) + (s > 0)*(s - torch.sqrt(torch.abs(r)))
        proximity = torch.abs(f - vertex)

        # Compute second derivative
        second = 2*c3*(-c1 + c3*(c1*f + c2)/(c3*f + c4))/(c3*f + c4)**2

        # Compute the largest |f'| in the interval [0, 1] (can be infinite if the hyperbola is centered inside the interval)
        max_first = torch.maximum(
            (torch.abs(c1*c4 - c2*c3)/c4**2).rename(None),
            (torch.abs(c1*c4 - c2*c3)/(c3 + c4)**2).rename(None)
        )
        infinity = torch.logical_and(-c4/c3 > 0, -c4/c3 < 1).rename(None)
        max_first[infinity] = float('inf')

        # Add obtained sensitivity values for this factor to the result dictionary
        result[nodes] = {
            'cpt': f.detach(),
            'derivative': derivative.detach(),
            'sensitivity_value': torch.abs(derivative).detach(),
            'proximity': proximity.detach(),
            'second_derivative': second.detach(),
            'largest_first_derivative': max_first.detach()
        }
    return result


def plot(g, probability, given=None, nbars=20, figsize=None, filename=None):
    """
    Find and show a barplot of the most influential parameters in a Bayesian network.

    Example:
    >>> g = pgmpy.readwrite.BIFReader("networks/alarm.bif").get_model()
    >>> plot(g, probability={'CVP': 'HIGH'}, given={'HISTORY': 'TRUE'})

    :param g: a `pgmpy.models.BayesianNetwork`
    :param probability: a dictionary with one key-value pair: {variable: value}
    :param given: [optional] a dictionary for all conditional evidence {variable: value}
    :param nbars: how many parameters to display
    :param filename: the file to save the figure to. If None (default), it will be shown on screen
    """

    if figsize is None:
        figsize = (10, 7.5)
    result = yodo(g, probability=probability, given=given)

    results_sorted = []
    for nodes in result:
        f = result[nodes]['cpt']
        idx = np.unravel_index(np.arange(f.numel()), f.shape)
        for i in range(len(idx[0])):
            cond = ['{} = {}'.format(f.names[j], idx[j][i]) for j in range(len(idx))]
            if len(cond) == 1:
                name = cond[0]
            else:
                name = cond[0] + ' | ' + ', '.join(cond[1:])
            results_sorted.append([
                name,
                f[tuple(index[i] for index in idx)].item(),
                result[nodes]['derivative'][tuple(index[i] for index in idx)].item(),
            ])
    results_sorted = sorted(results_sorted, key=lambda x: np.abs(x[2]), reverse=True)

    results_sorted = results_sorted[:nbars]
    plt.figure(figsize=figsize)
    plt.barh([r[0] for r in results_sorted[::-1]], [np.abs(r[2]) for r in results_sorted[::-1]], color=['tab:blue' if r[2] > 0 else 'tab:red' for r in results_sorted[::-1]])
    plt.xlabel('Sensitivity value')
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
