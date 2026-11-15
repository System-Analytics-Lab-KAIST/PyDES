# jax_backend - optional JAX backend for PyDES

Adds a second optional backend that runs same model as a compiled JAX program and many replications at once.
Results equal the base backend exactly (bit-wise). 

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
Environment.run_multiple(run_mcmsq, [(seed,) for seed in range(64)])                  # base PyDES (default)
Environment.run_multiple(run_mcmsq, [(seed,) for seed in range(64)], backend="jax")   # JAX

env.run_simulation_parallel(64, backend="jax")   # same switch, for event-only models
```

Alternatively, can fix for a whole run with `PYDES_BACKEND=jax`. An explicit `backend=` wins.
Replication `r` of the JAX backend equals replication `r` of the base backend.

A model supports `backend="jax"` once it names its JAX port with a `jax_port = "module:function"` attribute:
- on the work function for `run_multiple`, e.g. `run_mcmsq.jax_port = "examples.mcmsq.mcmsq_jax:run_multiple"`, port `(param_list) -> list of results`
- on the main model class for `run_simulation_parallel`, port `(env, n_instance) -> list of results`

Note, process models (`PModel` with `yield`) need `run_multiple`: the base `run_simulation_parallel` cannot pickle their generators.

Example: `examples/mcmsq/` is the multi-class, multi-server queue of lecture IE532 L3 with its JAX port.

## Why JAX is faster, and when it is not

A discrete-event simulation is sequential and does little work per event, so the base backend spends most of its time in the Python interpreter.
JAX primarily improves runtime through i) `jit` -> compiles the event loop into one program, and ii) `vmap` runs many replications in lockstep, on CPU vector units or a GPU, instead of one Python process each.
Note, usually on short runs JAX **is not faster**, and compilation costs seconds. A GPU is recommended for many replications.

