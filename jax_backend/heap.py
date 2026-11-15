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


def _get(h: Heap, i) -> Item:
    return Item(h.time[i], h.kind[i], h.arg[i], h.live[i])


def _put(h: Heap, i, it: Item) -> Heap:
    return h._replace(time=h.time.at[i].set(it.time), kind=h.kind.at[i].set(it.kind),
                      arg=h.arg.at[i].set(it.arg), live=h.live.at[i].set(it.live))


def _siftdown(h: Heap, startpos, pos) -> Heap:
    new = _get(h, pos)

    def cond(c):
        _, pos, done = c
        return (pos > startpos) & ~done

    def body(c):
        h, pos, _ = c
        parentpos = (pos - 1) >> 1
        parent = _get(h, parentpos)
        up = _less(new.time, parent.time)
        h = lax.cond(up, lambda h: _put(h, pos, parent), lambda h: h, h)
        return h, jnp.where(up, parentpos, pos), ~up

    h, pos, _ = lax.while_loop(cond, body, (h, pos, jnp.bool_(False)))
    return _put(h, pos, new)


def _siftup(h: Heap, pos) -> Heap:
    endpos, startpos = h.size, pos
    new = _get(h, pos)

    def body(c):
        h, pos = c
        childpos = 2 * pos + 1
        rightpos = childpos + 1
        right = (rightpos < endpos) & ~_less(h.time[childpos], h.time[jnp.minimum(rightpos, endpos - 1)])
        childpos = jnp.where(right, rightpos, childpos)
        return _put(h, pos, _get(h, childpos)), childpos

    h, pos = lax.while_loop(lambda c: 2 * c[1] + 1 < endpos, body, (h, pos))
    return _siftdown(_put(h, pos, new), startpos, pos)


def push(h: Heap, time, kind, arg) -> tuple[Heap, jax.Array]:
    """heapq.heappush. Returns the heap and an overflow flag. A full heap is left unchanged."""
    full = h.size >= h.time.shape[0]
    arg = jnp.broadcast_to(jnp.asarray(arg, jnp.int32), h.arg.shape[1:])
    it = Item(jnp.asarray(time, jnp.float64), jnp.asarray(kind, jnp.int32), arg, jnp.bool_(True))

    def do(h):
        h = _put(h, h.size, it)._replace(size=h.size + 1)
        return _siftdown(h, 0, h.size - 1)

    return lax.cond(full, lambda h: h, do, h), full


def pop(h: Heap) -> tuple[Heap, Item]:
    """heapq.heappop. The heap must not be empty."""
    last = _get(h, h.size - 1)
    h = h._replace(size=h.size - 1)

    def nonempty(h):
        top = _get(h, 0)
        return _siftup(_put(h, 0, last), 0), top

    return lax.cond(h.size > 0, nonempty, lambda h: (h, last), h)


def _match(h: Heap, kind, arg, arg_mask):
    in_use = jnp.arange(h.time.shape[0]) < h.size
    same_arg = jnp.all((h.arg == jnp.asarray(arg, jnp.int32)) | ~jnp.asarray(arg_mask, bool), axis=1)
    return in_use & h.live & (h.kind == kind) & same_arg


def cancel_first(h: Heap, kind, arg, arg_mask) -> tuple[Heap, jax.Array]:
    """EvtMsgQueue.cancelEvtMsg: the first live match in list order. arg_mask False = any value."""
    m = _match(h, kind, arg, arg_mask)
    i = jnp.argmax(m)
    found = m[i]
    return h._replace(live=h.live.at[i].set(h.live[i] & ~found)), found


def cancel_all(h: Heap, kind, arg, arg_mask) -> Heap:
    """EvtMsgQueue.cancelEvtMsgAll without pause."""
    return h._replace(live=h.live & ~_match(h, kind, arg, arg_mask))
