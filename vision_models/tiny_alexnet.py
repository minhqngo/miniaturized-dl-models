import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import nn


class TinyAlexNet(eqx.Module):
    features: list
    head: list
    
    def __init__(self, n_classes, key):
        keys = jr.split(key, 8)
        
        self.features = [
            nn.Conv2d(3, 32, kernel_size=3, padding=1, key=keys[0]),
            nn.BatchNorm(32, axis_name="batch"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, key=keys[1]),
            nn.BatchNorm(64, axis_name="batch"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, key=keys[2]),
            nn.BatchNorm(128, axis_name="batch"),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, key=keys[3]),
            nn.BatchNorm(128, axis_name="batch"),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, key=keys[4]),
            nn.BatchNorm(128, axis_name="batch"),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        
        self.head = [
            nn.Dropout(p=0.5),
            nn.Linear(128 * 4 * 4, 256, key=keys[5]),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128, key=keys[6]),
            nn.Linear(128, n_classes, key=keys[7]),
        ]
        
    def __call__(self, x, state, key=None, inference=False):
        for layer in self.features:
            if isinstance(layer, nn.BatchNorm):
                x, state = layer(x, state=state, inference=inference)
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)
                x = jax.nn.relu(x)
        
        x = jnp.ravel(x)
        
        keys = jr.split(key, 2) if key is not None else (None, None)
        
        x = self.head[0](x, key=keys[0], inference=inference)
        x = self.head[1](x)
        x = jax.nn.relu(x)
        
        x = self.head[2](x, key=keys[1], inference=inference)
        x = self.head[3](x)
        x = jax.nn.relu(x)
        
        x = self.head[4](x)
        return x, state
    
    
if __name__ == '__main__':
    key = jr.PRNGKey(0)
    model, state = nn.make_with_state(TinyAlexNet)(n_classes=10, key=key)
    
    params = eqx.filter(model, eqx.is_array)
    count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total Parameters: {count:,}")
    
    dummy_input = jr.normal(key, (3, 32, 32))
    output, _ = model(dummy_input, state=state, inference=True)
    
    eqx.tree_serialise_leaves("tiny_alexnet.eqx", model)
