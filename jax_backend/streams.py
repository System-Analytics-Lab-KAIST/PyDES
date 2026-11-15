"""Random numbers for JAX backend.

JAX's random generator differs from numpy's, so the same results need the same numbers.
To solve this issue, we use the exact RndNumGen.get() the base model would call, in same order,
and the JAX model reads them from an array.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .engine import ERR_STREAM, Sim, flag


class Stream(NamedTuple):
    values: jax.Array  # [stream_len]
    pos: jax.Array     # int32[]


def predraw(gen, n: int, dtype=np.float64) -> np.ndarray:
    """n successive gen.get() values; gen is an InputStat.RndNumGen."""
    return np.array([gen.get() for _ in range(n)], dtype=dtype)


def stream(values) -> Stream:
    return Stream(jnp.asarray(values), jnp.int32(0))


def draw(sim: Sim, st: Stream) -> tuple[Sim, Stream, jax.Array]:
    """Next value. A completed stream sets ERR_STREAM, causing it to repeat its last value."""
    n = st.values.shape[0]
    sim = flag(sim, st.pos >= n, ERR_STREAM)
    return sim, st._replace(pos=st.pos + 1), st.values[jnp.minimum(st.pos, n - 1)]
