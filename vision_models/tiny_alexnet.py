import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jnr
from equinox import nn


class TinyAlexNet(eqx.Module):
    features: list
    head: list
    
    def __init__(self, n_classes, key):
        keys = jnr.split(key, 8)
        
