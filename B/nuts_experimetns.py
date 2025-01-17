#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:34:27 2025

@author: jhk
"""

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)
from sklearn import model_selection
import pyro.contrib.gp as gp

import torch
import numpy as np
import seaborn as sns
import arviz as az  


import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch

def kernel(X, Z, variance, lengthscale):
    # Expand X and Z to calculate pairwise squared distances
    X = X.unsqueeze(-1) if X.dim() == 1 else X  # Ensure X is 2D
    Z = Z.unsqueeze(-1) if Z.dim() == 1 else Z  # Ensure Z is 2D
    pairwise_sq_dists = (X - Z.T).pow(2)  # Pairwise squared differences
    calc = variance.pow(2) * torch.exp(-0.5 * pairwise_sq_dists / lengthscale.pow(2))
    return calc


def model(X, Y): #måske lognormal for at sikre os positive tal
    variance = pyro.sample("variance", dist.Normal(4, 0.1))
    lengthscale = pyro.sample("lengthscale", dist.Normal(0.1, 0.02))
    noise = pyro.sample("noise", dist.Normal(0, .01))
    # Covariance matrix
    K = kernel(X, X, variance, lengthscale) + torch.eye(X.shape[0]) * noise.pow(2)
    # Sample from the multivariate normal
    y = pyro.sample(    
        "y", dist.MultivariateNormal(torch.zeros(X.shape[0]), covariance_matrix=K), obs=Y
    )
    return y


q = 2 
def func(x):
    return - np.power((np.sin(6*np.pi*x)),2) + 6*np.power(x,2) - 5*np.power(x,4) + (3/2)
x= np.linspace(0,1,100)
y = func(x)
def datafunc(x):
    return func(x) + np.random.normal(0, 0.01, func(x).shape)
def gendata(l):
    x = np.array(range(1,l))/(l-1)
    y = datafunc(x)   
    return x,y
data  = gendata(30)
X_train, X_test, y_train, y_test = model_selection.train_test_split(data[0], data[1], test_size= 10)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(x,y)
plt.show()
print ("Area less than q", np.sum(y)/100 <q )

#X_train = torch.tensor([[1.0], [2.0], [3.0]])  # Example data
#y_train = torch.tensor([1.0, 2.0, 3.0])  # Example data

warmup_steps = 100##[10, 50, 100, 300]
num_chains = 4

print(f"Warmup steps: {warmup_steps}, Chains: {num_chains}")
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
nuts_kernel = NUTS(model, step_size=0.1)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=warmup_steps, num_chains=num_chains)
mcmc.run(X_train, y_train)
#mcmc.summary()
posterior_samples = mcmc.get_samples()
#print(posterior_samples)
nuts_inference = az.from_pyro(mcmc)
nuts_summary = az.summary(nuts_inference, var_names=["variance", "lengthscale", "noise"])
print(nuts_summary)
pyro.clear_param_store()