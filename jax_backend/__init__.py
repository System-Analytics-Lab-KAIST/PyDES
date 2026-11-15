"""Optional JAX backend for PyDES.

Dont import/use this folder directly, call env.run_simulation_parallel(n, backend="jax").

A model opts in with a class attribute on its main model,
    jax_port = "module:function",
naming a function (env, n_instance) -> list of results, same as the base run_simulation_parallel.

Note, this file never imports JAX. The JAX modules load only once backend="jax" is actually used.
"""
import importlib
import os

BACKENDS = ("pydes", "jax")
ENV_VAR = "PYDES_BACKEND"


def backend() -> str:
    """Active default backend: PYDES_BACKEND or "pydes"."""
    return resolve()


def resolve(name: str | None = None) -> str:
    b = name or os.environ.get(ENV_VAR, "pydes")

    if b not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {b!r}")

    return b


def run_parallel(env, n_instance: int) -> list:
    """Environment.run_simulation_parallel on the JAX port of env.main_model."""
    model = env.main_model
    if model is None:
        raise ValueError("no main model, call env.initial_events(main_model) first")

    target = getattr(type(model), "jax_port", None)
    if target is None:
        raise NotImplementedError(f"{type(model).__name__} has no jax_port, use backend=\"pydes\"")

    module, func = target.split(":")

    return getattr(importlib.import_module(module), func)(env, n_instance)
