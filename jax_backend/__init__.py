"""Optional JAX backend for PyDES.

Dont import/use this folder directly, call one of
    env.run_simulation_parallel(n, backend="jax")
    Environment.run_multiple(work_fn, param_list, backend="jax")

A model opts in with a jax_port = "module:function" attribute,
    on its main model class for run_simulation_parallel: function (env, n_instance) -> list of results,
    on its work function for run_multiple: function (param_list) -> list of results,
same results as the base call. Note, run_multiple is the one for process models (PModel with yield),
since the base run_simulation_parallel cannot pickle their generators.

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

    return _port(type(model), type(model).__name__)(env, n_instance)


def run_multiple_jax(work_fn, param_list) -> list:
    """Environment.run_multiple on the JAX port of work_fn."""
    return _port(work_fn, work_fn.__name__)(param_list)


def _port(owner, name: str):
    target = getattr(owner, "jax_port", None)
    if target is None:
        raise NotImplementedError(f"{name} has no jax_port, use backend=\"pydes\"")

    module, func = target.split(":")

    return getattr(importlib.import_module(module), func)
