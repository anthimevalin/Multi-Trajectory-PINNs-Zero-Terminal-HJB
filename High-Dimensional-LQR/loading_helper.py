from __future__ import annotations
import os, json, pickle
from typing import Callable, Optional, Tuple, Dict, Union
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import glorot_normal, zeros, normal
import ml_collections
import orbax.checkpoint as ocp
from orbax.checkpoint import utils as ocp_utils
from orbax.checkpoint import type_handlers
from jax.sharding import SingleDeviceSharding
import jax.random as jran

activation_fn = {
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}

def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")

def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = jran.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init

class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v
        bias = self.param("bias", self.bias_init, (self.features,))
        y = jnp.dot(x, kernel) + bias

        return y

class MLP(nn.Module):
    arch_name: Optional[str]="MLP"
    hidden_dim: Tuple[int]=(32, 16)
    out_dim: int=1
    activation: str="tanh"
    periodicity: Union[None, Dict]=None
    fourier_emb: Union[None, Dict]=None
    reparam: Union[None, Dict]=None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.hidden_dim)):
            x = Dense(features=self.hidden_dim[i], reparam=self.reparam)(x)
            x = self.activation_fn(x)
        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x



def ann_gen(config):
    ann = None
    reparam = None
    if config.ann_reparam=="weight_fact":
        reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 0.5, "stddev": 0.1})

    if config.ann_str == "MLP":
        ann = MLP(arch_name=config.ann_str,
                  hidden_dim=config.ann_hidden_dim,
                  out_dim=config.ann_out_dim,
                  activation=config.ann_activation_str,
                  periodicity=config.ann_periodicity,
                  fourier_emb=config.ann_fourier_emb,
                  reparam=reparam)

    return ann



def load_model_and_history(run_dir: str, tag: str):
    """
    Loads:
      {run_dir}/{tag}_config.json
      {run_dir}/{tag}_params/     
      {run_dir}/{tag}_history.pkl
    Returns: (model_fn, params, config, history)
    """
    run_dir = os.path.abspath(run_dir)
    
    # 1) config
    with open(os.path.join(run_dir, f"{tag}_config.json"), "r") as f:
        cfg = ml_collections.ConfigDict(json.load(f))

    # 2) target tree 
    ann = ann_gen(cfg)
    dummy = jnp.ones((1, int(cfg.ann_in_dim)))
    params0 = ann.init(jax.random.PRNGKey(0), dummy)

    # 3) restore params 
    ckpt = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    ckpt_dir = os.path.join(run_dir, f"{tag}_params")
    try:
        restore_args = ocp_utils.restore_args_from_target(params0)
        params = ckpt.restore(ckpt_dir, item=params0, restore_args=restore_args)
    except Exception:
        sharding = SingleDeviceSharding(jax.devices()[0])
        restore_args = jax.tree_util.tree_map(
            lambda x: type_handlers.ArrayRestoreArgs(
                global_shape=x.shape, dtype=x.dtype, sharding=sharding
            ),
            params0,
        )
        params = ckpt.restore(ckpt_dir, item=params0, restore_args=restore_args)

    # 4) history
    with open(os.path.join(run_dir, f"{tag}_history.pkl"), "rb") as f:
        history = pickle.load(f)

    # 5) model
    model_fn = jax.jit(lambda x: ann.apply(params, x))
    return model_fn, params, cfg, history
