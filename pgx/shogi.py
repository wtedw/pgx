import jax.numpy as jnp
from flax.struct import dataclass

#   0 空白
#   1 先手歩
#   2 先手香車
#   3 先手桂馬
#   4 先手銀
#   5 先手角
#   6 先手飛車
#   7 先手金
#   8 先手玉
#   9 先手と
#  10 先手成香
#  11 先手成桂
#  12 先手成銀
#  13 先手馬
#  14 先手龍
#  15 後手歩
#  16 後手香車
#  17 後手桂馬
#  18 後手銀
#  19 後手角
#  20 後手飛車
#  21 後手金
#  22 後手玉
#  23 後手と
#  24 後手成香
#  25 後手成桂
#  26 後手成銀
#  27 後手馬
#  28 後手龍


# fmt: off
INIT_PIECE_BOARD = jnp.int8([16, 0, 15, 0, 0, 0, 1, 0, 2, 17, 19, 15, 0, 0, 0, 1, 6, 3, 18, 0, 15, 0, 0, 0, 1, 0, 4, 22, 0, 15, 0, 0, 0, 1, 0, 7, 23, 0, 15, 0, 0, 0, 1, 0, 8, 22, 0, 15, 0, 0, 0, 1, 0, 7, 18, 0, 15, 0, 0, 0, 1, 0, 4, 17, 20, 15, 0, 0, 0, 1, 5, 3, 16, 0, 15, 0, 0, 0, 1, 0, 2])
# fmt: on


@dataclass
class State:
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,)
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)


def init():
    """Initialize Shogi State.
    >>> s = init()
    >>> s.piece_board.reshape((9, 9))
    Array([[16,  0, 15,  0,  0,  0,  1,  0,  2],
           [17, 19, 15,  0,  0,  0,  1,  6,  3],
           [18,  0, 15,  0,  0,  0,  1,  0,  4],
           [22,  0, 15,  0,  0,  0,  1,  0,  7],
           [23,  0, 15,  0,  0,  0,  1,  0,  8],
           [22,  0, 15,  0,  0,  0,  1,  0,  7],
           [18,  0, 15,  0,  0,  0,  1,  0,  4],
           [17, 20, 15,  0,  0,  0,  1,  5,  3],
           [16,  0, 15,  0,  0,  0,  1,  0,  2]], dtype=int8)
    >>> jnp.rot90(s.piece_board.reshape((9, 9)), k=3)
    Array([[16, 17, 18, 22, 23, 22, 18, 17, 16],
           [ 0, 20,  0,  0,  0,  0,  0, 19,  0],
           [15, 15, 15, 15, 15, 15, 15, 15, 15],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
           [ 0,  5,  0,  0,  0,  0,  0,  6,  0],
           [ 2,  3,  4,  7,  8,  7,  4,  3,  2]], dtype=int8)
    """
    return State()
