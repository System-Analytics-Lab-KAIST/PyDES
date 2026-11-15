"""JAX backend Event Loop: EvtMsgQueue.runSimulation compiled.

We represent a model here as a set of handlers, one per event kind, each a function
(sim, state, arg) -> (sim, state). The engine pops the next event, advances the clock
and dispatches with lax.switch, until a handler calls stop(), the heap runs empty,
or an error flag is set. run_batch runs many replications in lockstep (with vmap).
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


Handler = Callable[[Sim, Any, jax.Array], tuple[Sim, Any]]


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


def schedule(sim: Sim, dt, kind, arg=0) -> Sim:
    """EvtMsgQueue.scheduleEvtMsg4: event at Tnow + dt."""
    h, full = hp.push(sim.heap, sim.tnow + dt, kind, arg)
    return flag(sim._replace(heap=h), full, ERR_HEAP_FULL)


def cancel(sim: Sim, kind, arg=0, arg_mask=True) -> tuple[Sim, jax.Array]:
    """EvtMsgQueue.cancelEvtMsg."""
    h, found = hp.cancel_first(sim.heap, kind, arg, arg_mask)
    return sim._replace(heap=h), found


def stop(sim: Sim) -> Sim:
    """EvtMsgQueue.stop_running."""
    return sim._replace(running=jnp.bool_(False))


def _active(sim: Sim, max_events: int):
    return sim.running & (sim.heap.size > 0) & (sim.err == 0) & (sim.n_events < max_events)


def _step(model: Model, sim: Sim, state):
    """EvtMsgQueue.runNextEvent: drop canceled event without advancing the clock."""
    h, ev = hp.pop(sim.heap)
    sim = sim._replace(heap=h, n_events=sim.n_events + 1)

    def fire(c):
        sim, state = c
        sim = flag(sim, ev.time < sim.tnow, ERR_PAST)
        return lax.switch(ev.kind, model.handlers, sim._replace(tnow=ev.time), state, ev.arg)

    return lax.cond(ev.live, fire, lambda c: c, (sim, state))


def _finish(sim: Sim, max_events: int) -> Sim:
    return flag(sim, sim.running & (sim.heap.size > 0) & (sim.err == 0) & (sim.n_events >= max_events),
                ERR_MAX_EVENTS)


@partial(jax.jit, static_argnames=("model", "max_events"))
def run(model: Model, sim: Sim, state, max_events: int):
    """One replication until stop, empty heap or error."""
    sim, state = lax.while_loop(lambda c: _active(c[0], max_events),
                                lambda c: _step(model, *c), (sim, state))
    return _finish(sim, max_events), state


@partial(jax.jit, static_argnames=("model", "max_events"))
def run_batch(model: Model, sim: Sim, state, max_events: int):
    def step(sim, state):
        return lax.cond(_active(sim, max_events), lambda c: _step(model, *c), lambda c: c, (sim, state))

    sim, state = lax.while_loop(lambda c: jnp.any(jax.vmap(_active, (0, None))(c[0], max_events)),
                                lambda c: jax.vmap(step)(*c), (sim, state))
    return jax.vmap(_finish, (0, None))(sim, max_events), state


def check(model: Model, err) -> None:
    """Raise with appropriate msg if any replication flags error."""
    err = int(jnp.bitwise_or.reduce(jnp.atleast_1d(err)))
    if err:
        texts = dict(_ERR_TEXT, **dict(model.errors))
        raise RuntimeError(f"{model.name}: " + "; ".join(t for b, t in texts.items() if err & b))
