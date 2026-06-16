import jax.numpy as jnp
from jax import grad , vmap,jit
def predict(params,inputs):
    for W,b in params:
        outputs = jnp.dot(inputs,W) + b
        inputs = jnp.tanh(outputs)
    return outputs
def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.mean((predictions - targets) ** 2)
gradient_fun=jit(grad(loss))
perexample_grads=jit(vmap(grad(loss),in_axes=(None,0)),in_shardings=..., out_shardings=...)

