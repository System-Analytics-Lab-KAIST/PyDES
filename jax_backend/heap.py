"""Fixed-capacity event heap on arrays (exact order as Python's heapq)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class Heap(NamedTuple):
    time: jax.Array  # float64[cap]
    kind: jax.Array  # int32[cap], event type
    arg: jax.Array   # int32[cap, n_arg], event payload
    live: jax.Array  # bool[cap], False once canceled
    size: jax.Array  # int32[]


class Item(NamedTuple):
    time: jax.Array
    kind: jax.Array
    arg: jax.Array
    live: jax.Array


def empty(cap: int, n_arg: int = 1) -> Heap:
    return Heap(time=jnp.zeros(cap, jnp.float64), kind=jnp.zeros(cap, jnp.int32),
                arg=jnp.zeros((cap, n_arg), jnp.int32), live=jnp.zeros(cap, bool),
                size=jnp.int32(0))


def _less(a, b):
    return a < b  # EvtMsg.__lt__: time only


def _cap(h: Heap) -> int:
    return h.time.shape[0]


def _levels(h: Heap) -> int:
    """Upper bound on the levels a sift moves: the depth of index cap - 1 is floor(log2(cap))."""
    return _cap(h).bit_length()


def _get(h: Heap, i) -> Item:
    i = jnp.clip(i, 0, _cap(h) - 1)
    return Item(h.time[i], h.kind[i], h.arg[i], h.live[i])


def _put(h: Heap, i, it: Item, do=True) -> Heap:
    """h[i] = it where do. Note, a masked select instead of a scatter, so XLA fuses the writes of a sift."""
    m = (jnp.arange(_cap(h)) == i) & do
    return h._replace(time=jnp.where(m, it.time, h.time), kind=jnp.where(m, it.kind, h.kind),
                      arg=jnp.where(m[:, None], it.arg, h.arg), live=jnp.where(m, it.live, h.live))


def _siftdown(h: Heap, startpos, pos, do=True) -> Heap:
    """heapq._siftdown with a fixed number of rounds.

    A loop that stops on the data would make the GPU wait for the host each round.
    """
    new = _get(h, pos)

    def body(_, c):
        h, pos, done = c
        parentpos = (jnp.maximum(pos, 1) - 1) >> 1
        parent = _get(h, parentpos)
        up = ~done & (pos > startpos) & _less(new.time, parent.time)
        return _put(h, pos, parent, up), jnp.where(up, parentpos, pos), ~up

    h, pos, _ = lax.fori_loop(0, _levels(h), body, (h, pos, ~jnp.asarray(do)))
    return _put(h, pos, new, do)


def _siftup(h: Heap, pos, do=True) -> Heap:
    """heapq._siftup, loop as _siftdown."""
    endpos, startpos = h.size, pos
    new = _get(h, pos)

    def body(_, c):
        h, pos = c
        childpos = 2 * pos + 1
        go = do & (childpos < endpos)
        rightpos = childpos + 1
        right = (rightpos < endpos) & ~_less(_get(h, childpos).time, _get(h, rightpos).time)
        childpos = jnp.where(right, rightpos, childpos)
        return _put(h, pos, _get(h, childpos), go), jnp.where(go, childpos, pos)

    h, pos = lax.fori_loop(0, _levels(h), body, (h, pos))
    return _siftdown(_put(h, pos, new, do), startpos, pos, do)


def push(h: Heap, time, kind, arg, do=True) -> tuple[Heap, jax.Array]:
    """heapq.heappush if do. Returns the heap and an overflow flag. A full heap is left unchanged."""
    full = do & (h.size >= _cap(h))
    ok = do & ~full
    arg = jnp.broadcast_to(jnp.asarray(arg, jnp.int32), h.arg.shape[1:])
    it = Item(jnp.asarray(time, jnp.float64), jnp.asarray(kind, jnp.int32), arg, jnp.bool_(True))
    h = _put(h, h.size, it, ok)._replace(size=h.size + ok)
    return _siftdown(h, 0, jnp.maximum(h.size - 1, 0), ok), full


def pop(h: Heap, do=True) -> tuple[Heap, Item]:
    """heapq.heappop if do. The heap must not be empty then."""
    ok = do & (h.size > 0)
    last = _get(h, jnp.maximum(h.size - 1, 0))
    h = h._replace(size=h.size - ok)
    nonempty = ok & (h.size > 0)
    top = _get(h, 0)
    h = _siftup(_put(h, 0, last, nonempty), 0, nonempty)
    return h, jax.tree.map(lambda a, b: jnp.where(nonempty, a, b), top, last)


def _match(h: Heap, kind, arg, arg_mask):
    in_use = jnp.arange(h.time.shape[0]) < h.size
    same_arg = jnp.all((h.arg == jnp.asarray(arg, jnp.int32)) | ~jnp.asarray(arg_mask, bool), axis=1)
    return in_use & h.live & (h.kind == kind) & same_arg


def cancel_first(h: Heap, kind, arg, arg_mask, do=True) -> tuple[Heap, jax.Array]:
    """EvtMsgQueue.cancelEvtMsg if do: the first live match in list order. arg_mask False = any value."""
    m = _match(h, kind, arg, arg_mask)
    i = jnp.argmax(m)
    found = m[i] & do
    return h._replace(live=h.live & ~((jnp.arange(_cap(h)) == i) & found)), found


def cancel_all(h: Heap, kind, arg, arg_mask) -> Heap:
    """EvtMsgQueue.cancelEvtMsgAll without pause."""
    return h._replace(live=h.live & ~_match(h, kind, arg, arg_mask))
