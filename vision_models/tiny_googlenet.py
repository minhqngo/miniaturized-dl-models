import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import nn


# Conv -> BN -> ReLU
class ConvBlock(eqx.Module):
    layers: list
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, key):
        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, key=key, use_bias=False),
            nn.BatchNorm(out_channels, axis_name="batch")
        ]
        
    def __call__(self, x, state, inference=False):
        x = self.layers[0](x)
        x, state = self.layers[1](x, state=state, inference=inference)
        x = jax.nn.relu(x)
        return x, state
    
    
class InceptionBlock(eqx.Module):
    branch_1: ConvBlock
    branch_2: list
    branch_3: list
    branch_4: list
    
    def __init__(self, in_c, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj, key):
        keys = jr.split(key, 5)
        
        self.branch_1 = ConvBlock(in_c, ch1x1, kernel_size=1, padding=0, key=keys[0])
        
        self.branch_2 = [
            ConvBlock(in_c, ch3x3_reduce, kernel_size=1, padding=0, key=keys[1]),
            ConvBlock(ch3x3_reduce, ch3x3, kernel_size=3, padding=1, key=keys[2])
        ]
        
        self.branch_3 = [
            ConvBlock(in_c, ch5x5_reduce, kernel_size=1, padding=0, key=keys[3]),
            ConvBlock(ch5x5_reduce, ch5x5, kernel_size=5, padding=2, key=keys[4])
        ]
        
        self.branch_4 = [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_c, pool_proj, kernel_size=1, padding=0, key=keys[4])
        ]
        
    def __call__(self, x, state, inference=False):
        out_1, state = self.branch_1(x, state, inference)
        
        out_2, state = self.branch_2[0](x, state, inference)
        out_2, state = self.branch_2[1](out_2, state, inference)
        
        out_3, state = self.branch_3[0](x, state, inference)
        out_3, state = self.branch_3[1](out_3, state, inference)
        
        out_4 = self.branch_4[0](x)
        out_4, state = self.branch_4[1](out_4, state, inference)
        
        return jnp.concatenate([out_1, out_2, out_3, out_4], axis=0), state
    
    
class TinyGoogLeNet(eqx.Module):
    stem: ConvBlock
    blocks: list
    head: nn.Linear
    
    def __init__(self, n_classes, key=None):
        keys = jr.split(key, 20)
        
        self.stem = ConvBlock(3, 64, kernel_size=3, padding=1, key=keys[0])
        
        self.blocks = [
            InceptionBlock(64, 32, 48, 64, 8, 16, 16, key=keys[1]),
            InceptionBlock(128, 32, 48, 64, 8, 16, 16, key=keys[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            InceptionBlock(128, 64, 64, 96, 16, 32, 32, key=keys[3]),
            InceptionBlock(224, 64, 64, 96, 16, 32, 32, key=keys[4]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            InceptionBlock(224, 64, 64, 96, 16, 32, 32, key=keys[5]),
            InceptionBlock(224, 64, 64, 96, 16, 32, 32, key=keys[6])
        ]
        
        self.head = [
            nn.Dropout(p=0.4),
            nn.Linear(224, n_classes, key=keys[7])
        ]
        
    def __call__(self, x, state, key=None, inference=False):
        x, state = self.stem(x, state, inference)
        
        for block in self.blocks:
            if isinstance(block, InceptionBlock):
                x, state = block(x, state, inference)
            elif isinstance(block, nn.MaxPool2d):
                x = block(x)
        
        x = jnp.mean(x, axis=(1, 2))
        
        x = self.head[0](x, key=key, inference=inference)
        x = self.head[1](x)
        
        return x, state
    
    
if __name__ == '__main__':
    key = jr.PRNGKey(0)
    model, state = nn.make_with_state(TinyGoogLeNet)(n_classes=10, key=key)
    
    params = eqx.filter(model, eqx.is_array)
    count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total Parameters: {count:,}")
    
    dummy_input = jr.normal(key, (3, 32, 32))
    output, _ = model(dummy_input, state=state, inference=True)
    
    eqx.tree_serialise_leaves("tiny_googlenet.eqx", model)
