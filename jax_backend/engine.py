"""JAX backend Event Loop: EvtMsgQueue.runSimulation compiled.

We represent a model here as a set of handlers, one per event kind, each a function
(sim, state, arg, do) -> (sim, state). The engine pops the next event, advances the clock
and calls every handler with do = "this event is yours", until a handler calls stop(),
the heap runs empty, or an error flag is set. run_batch runs many replications in lockstep (with vmap).

Note, handlers change state only where do is True and only the few cells they touch
(set_at, select, schedule(..., do)). No lax.switch or lax.cond over the whole state:
under vmap those run every branch and copy the full state on every event.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import lax

from . import heap as hp

jax.config.update("jax_enable_x64", True)  # Match default/base PyDES float64 compute

ERR_HEAP_FULL = 1
ERR_PAST = 2
ERR_STREAM = 4
ERR_MAX_EVENTS = 8
ERR_MODEL = 256  # first bit free for model-defined errors

SMALL = 4096  # arrays up to this size are written by a masked select, which XLA fuses; larger by a scatter
EVENTS_PER_ROUND = 16  # events per replication between two checks of "any replication still running" (a GPU-host sync)

_ERR_TEXT = {
        ERR_HEAP_FULL: "event heap full, raise heap_cap",
        ERR_PAST: "event scheduled in the past",
        ERR_STREAM: "random stream exhausted, raise stream_len",
        ERR_MAX_EVENTS: "max_events reached"
    }


class Sim(NamedTuple):
    heap: hp.Heap
    tnow: jax.Array      # float64[]
    running: jax.Array   # bool[]
    err: jax.Array       # int32[], bit flags
    n_events: jax.Array  # int32[], events processed, canceled ones included


Handler = Callable[[Sim, Any, jax.Array, jax.Array], tuple[Sim, Any]]


@dataclass(frozen=True, eq=False)
class Model:
    """Handlers indexed by event kind. Hashed by identity => jit compiles once per model."""
    name: str
    handlers: Sequence[Handler]
    errors: tuple[tuple[int, str], ...] = ()  # model-defined (bit, text)


def new_sim(heap_cap: int, n_arg: int = 1) -> Sim:
    return Sim(hp.empty(heap_cap, n_arg), jnp.float64(0.0), jnp.bool_(True), jnp.int32(0), jnp.int32(0))


def flag(sim: Sim, cond, bit: int) -> Sim:
    return sim._replace(err=sim.err | jnp.where(cond, bit, 0).astype(jnp.int32))


def mul(a, b):
    """a * b rounded on its own, as Python does. Note, under jit XLA otherwise fuses a product
    and a following add into one FMA (single rounding), which changes the last bits."""
    return lax.optimization_barrier(jnp.multiply(a, b))


def select(do, new, old):
    """new where do, else old, for any pytree. Keep it for small state."""
    return jax.tree.map(lambda a, b: jnp.where(do, a, b), new, old)


def set_at(arr, i, v, do=True):
    """arr.at[i].set(v) where do, a one-cell write also under vmap. i indexes the first axis."""
    if arr.size > SMALL:
        return arr.at[i].set(jnp.where(do, v, arr[i]))
    m = ((jnp.arange(arr.shape[0]) == i) & do).reshape((-1,) + (1,) * (arr.ndim - 1))
    return jnp.where(m, jnp.asarray(v, arr.dtype), arr)


def schedule(sim: Sim, dt, kind, arg=0, do=True) -> Sim:
    """EvtMsgQueue.scheduleEvtMsg4 if do: event at Tnow + dt."""
    h, full = hp.push(sim.heap, sim.tnow + lax.optimization_barrier(dt), kind, arg, do)  # no FMA with dt
    return flag(sim._replace(heap=h), full, ERR_HEAP_FULL)


def cancel(sim: Sim, kind, arg=0, arg_mask=True, do=True) -> tuple[Sim, jax.Array]:
    """EvtMsgQueue.cancelEvtMsg if do."""
    h, found = hp.cancel_first(sim.heap, kind, arg, arg_mask, do)
    return sim._replace(heap=h), found


def stop(sim: Sim, do=True) -> Sim:
    """EvtMsgQueue.stop_running if do."""
    return sim._replace(running=sim.running & ~do)


def _active(sim: Sim, max_events: int):
    return sim.running & (sim.heap.size > 0) & (sim.err == 0) & (sim.n_events < max_events)


def _step(model: Model, sim: Sim, state, active):
    """EvtMsgQueue.runNextEvent if active: drop canceled event without advancing the clock."""
    h, ev = hp.pop(sim.heap, active)
    fire = active & ev.live
    sim = sim._replace(heap=h, n_events=sim.n_events + active)
    sim = flag(sim, fire & (ev.time < sim.tnow), ERR_PAST)
    sim = sim._replace(tnow=jnp.where(fire, ev.time, sim.tnow))
    for kind, handler in enumerate(model.handlers):
        sim, state = handler(sim, state, ev.arg, fire & (ev.kind == kind))

    return sim, state


def _finish(sim: Sim, max_events: int) -> Sim:
    return flag(sim, sim.running & (sim.heap.size > 0) & (sim.err == 0) & (sim.n_events >= max_events),
                ERR_MAX_EVENTS)


@partial(jax.jit, static_argnames=("model", "max_events"))
def run(model: Model, sim: Sim, state, max_events: int):
    """One replication until stop, empty heap or error."""
    sim, state = lax.while_loop(lambda c: _active(c[0], max_events),
                                lambda c: _step(model, *c, jnp.bool_(True)), (sim, state))
    return _finish(sim, max_events), state


@partial(jax.jit, static_argnames=("model", "max_events"))
def run_batch(model: Model, sim: Sim, state, max_events: int):
    """Many replications in lockstep. A step of a replication that is done changes nothing."""
    def step(sim, state):
        return _step(model, sim, state, _active(sim, max_events))

    def round_(c):
        return lax.fori_loop(0, EVENTS_PER_ROUND, lambda _, c: jax.vmap(step)(*c), c)

    sim, state = lax.while_loop(lambda c: jnp.any(jax.vmap(_active, (0, None))(c[0], max_events)), round_,
                                (sim, state))
    return jax.vmap(_finish, (0, None))(sim, max_events), state


def check(model: Model, err) -> None:
    """Raise with appropriate msg if any replication flags error."""
    err = int(jnp.bitwise_or.reduce(jnp.atleast_1d(err)))
    if err:
        texts = {**_ERR_TEXT, **dict(model.errors)}
        raise RuntimeError(f"{model.name}: " + "; ".join(t for b, t in texts.items() if err & b))
