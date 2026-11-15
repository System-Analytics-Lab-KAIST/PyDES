# jax_backend - optional JAX backend for PyDES

Adds a second optional backend that runs same model as one compiled JAX program and many replications at once.
Results equal the base backend value for value.

## Installation

The base backend works as is.

JAX is only needed for `backend="jax"`, and assumes an NVIDIA GPU with CUDA 12:

```bash
pip install -r requirements-jax.txt            # jax[cuda12]
python -c "import jax; print(jax.devices())"   # should list a CUDA device
```

Without an NVIDIA GPU, switch the two lines in `requirements-jax.txt` to the commented-out `jax[cpu]`. Results are the same, only potentially slower for many replications.

(Note, PyDES never imports JAX unless `backend="jax"` is used.)

## Usage

Same call as before, with one optional parameter:

```python
env.run_simulation_parallel(64)                  # base PyDES (default)
env.run_simulation_parallel(64, backend="jax")   # JAX
```

Alternatively, can fix for a whole run with `PYDES_BACKEND=jax`. An explicit `backend=` wins.
Replication `r` of the JAX backend equals replication `r` of the base backend.

A model supports `backend="jax"` once its main model class names its JAX port, e.g. `jax_port = "mcmsq_jax:run_parallel"`.
The port is a function `(env, n_instance) -> list of results`, same as the base `run_simulation_parallel`.

## Why JAX is faster, and when it is not

A discrete-event simulation is sequential and does little work per event, so the base backend spends most of its time in the Python interpreter.
JAX primarily improves runtime through i) `jit` -> compiles the event loop into one program, and ii) `vmap` runs many replications in lockstep, on CPU vector units or a GPU, instead of one Python process each.
Note, usually on short runs JAX **is not faster**, and compilation costs seconds. A GPU is recommended for many replications.

