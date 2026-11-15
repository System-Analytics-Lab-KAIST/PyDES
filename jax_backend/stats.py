"""Statistics of Utility.py as array state, same formulas in the same order (float64)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

INF = jnp.inf


class TimeAvg(NamedTuple):
    """Utility.TimeAverage without history."""
    initT: jax.Array
    prevT: jax.Array
    prevV: jax.Array
    sum: jax.Array
    XX: jax.Array
    min: jax.Array
    max: jax.Array


def time_avg(t, v=0.0) -> TimeAvg:
    f = jnp.float64
    return TimeAvg(f(t), f(t), f(v), f(0.0), f(0.0), f(INF), f(-INF))


def state_change(ta: TimeAvg, t, v) -> TimeAvg:
    """TimeAverage.stateChange."""
    v = jnp.asarray(v, jnp.float64)
    later = t > ta.initT
    dt = t - ta.prevT

    return TimeAvg(ta.initT, jnp.asarray(t, jnp.float64), v,
                   ta.sum + ta.prevV * dt, ta.XX + ta.prevV * ta.prevV * dt,
                   jnp.where(later & (v < ta.min), v, ta.min), jnp.where(later & (v > ta.max), v, ta.max))


def time_avg_value(ta: TimeAvg, t) -> tuple[TimeAvg, jax.Array]:
    """TimeAverage.timeAvg(): closes the current interval first, as the original does."""
    dT = t - ta.initT
    closed = state_change(ta, t, ta.prevV)
    ta = jax.tree.map(lambda a, b: jnp.where(dT <= 0.0, a, b), ta, closed)
    return ta, jnp.where(dT <= 0.0, 0.0, ta.sum / jnp.where(dT <= 0.0, 1.0, dT))


def time_avg_reset(ta: TimeAvg, t) -> TimeAvg:
    """TimeAverage.reset(v=None): keeps prevV."""
    return time_avg(t, ta.prevV)


class Acc(NamedTuple):
    """Utility.Accumulator without history."""
    n: jax.Array
    X: jax.Array
    XX: jax.Array
    min: jax.Array
    max: jax.Array


def acc() -> Acc:
    f = jnp.float64
    return Acc(jnp.int32(0), f(0.0), f(0.0), f(INF), f(-INF))


def acc_add(a: Acc, x) -> Acc:
    """Accumulator.append."""
    x = jnp.asarray(x, jnp.float64)
    return Acc(a.n + 1, a.X + x, a.XX + x * x, jnp.where(x < a.min, x, a.min), jnp.where(x > a.max, x, a.max))


def acc_avg(a: Acc) -> jax.Array:
    """Accumulator.avg."""
    return jnp.where(a.n == 0, 0.0, a.X / jnp.maximum(a.n, 1))


class StateTime(NamedTuple):
    """Utility.StateTime: time per state plus counters, one row."""
    S: jax.Array      # float64[n_states]
    prevS: jax.Array  # int32[]
    initT: jax.Array
    prevT: jax.Array


def state_time(n_states: int, t, init_state: int) -> StateTime:
    return StateTime(jnp.zeros(n_states, jnp.float64), jnp.int32(init_state), jnp.float64(t), jnp.float64(t))


def set_state(st: StateTime, t, s) -> StateTime:
    """StateTime.setState."""
    return StateTime(st.S.at[st.prevS].add(t - st.prevT), jnp.int32(s), st.initT, jnp.asarray(t, jnp.float64))


def add_count(st: StateTime, col, n=1) -> StateTime:
    """StateTime.addCount."""
    return st._replace(S=st.S.at[col].add(n))
