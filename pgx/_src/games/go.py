# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple, Optional

import jax
import numpy as np
from jax import Array, lax
from jax import numpy as jnp

ZOBRIST_BOARD = jax.random.randint(jax.random.PRNGKey(12345), (3, 19 * 19, 2), 0, 2**31 - 1, jnp.uint32)

# ADJ[size][p, q] = 1 iff q is an on-board 4-neighbor of p. Precomputed numpy constants so
# that neighbor aggregation (summing/any-ing over neighbors) becomes a single matmul
# `ADJ @ vec`.
ADJ = {}
for _size in range(1, 20):
    _n = _size * _size
    _adj = np.zeros((_n, _n), dtype=np.int32)
    for _xy in range(_n):
        _r, _c = _xy // _size, _xy % _size
        for _dr, _dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < _size and 0 <= _nc < _size:
                _adj[_xy, _nr * _size + _nc] = 1
    ADJ[_size] = _adj


class GameState(NamedTuple):
    step_count: Array = jnp.int32(0)
    # ids of representative stone (smallest) in the connected stones
    board: Array = jnp.zeros(19 * 19, dtype=jnp.int32)  # b > 0, w < 0, empty = 0
    board_history: Array = jnp.full((8, 19 * 19), 2, dtype=jnp.int32)  # for obs
    num_captured: Array = jnp.zeros(2, dtype=jnp.int32)  # (b, w)
    consecutive_pass_count: Array = jnp.int32(0)
    ko: Array = jnp.int32(-1)  # by SSK
    is_psk: Array = jnp.bool_(False)
    hash_history: Array = jnp.zeros((19 * 19 * 2, 2), dtype=jnp.uint32)

    @property
    def color(self) -> Array:
        return self.step_count % 2


class Game:
    def __init__(
        self, size: int = 19, komi: float = 7.5, history_length: int = 8, max_termination_steps: Optional[int] = None
    ):
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = size * size * 2 if max_termination_steps is None else max_termination_steps

    def init(self) -> GameState:
        return GameState(
            board=jnp.zeros(self.size**2, dtype=jnp.int32),
            board_history=jnp.full((self.history_length, self.size**2), 2, dtype=jnp.int32),
            hash_history=jnp.zeros((self.max_termination_steps, 2), dtype=jnp.uint32),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        state = state._replace(ko=jnp.int32(-1))
        # update state
        state = lax.cond(
            (action < self.size * self.size),
            lambda: _apply_action(state, action, self.size),
            lambda: _apply_pass(state),
        )
        # update board history
        board_history = jnp.roll(state.board_history, self.size**2)
        board_history = board_history.at[0].set(jnp.clip(state.board, -1, 1).astype(jnp.int32))
        state = state._replace(board_history=board_history)
        # check PSK
        hash_ = _compute_hash(state)
        state = state._replace(hash_history=state.hash_history.at[state.step_count].set(hash_))
        state = state._replace(is_psk=_is_psk(state))
        # increment turns
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        my_sign, _ = _signs(color)

        def _make(i):
            c = jnp.int32([1, -1])[i % 2] * my_sign
            return state.board_history[i // 2] == c

        log = jax.vmap(_make)(jnp.arange(self.history_length * 2))
        color = jnp.full_like(log[0], color)  # b = 0, w = 1
        return jnp.vstack([log, color]).transpose().reshape((self.size, self.size, -1))

    def legal_action_mask(self, state: GameState) -> Array:
        # some logic is inspired by OpenSpiel's Go implementation
        is_empty = state.board == 0
        my_sign, opp_sign = _signs(state.color)
        in_atari = _atari_per_cell(state, self.size)
        has_liberty = (state.board * my_sign > 0) & ~in_atari
        can_kill = (state.board * opp_sign > 0) & in_atari

        # A move is legal if the empty point has a neighbor that is empty, killable, or a
        # friendly chain with a liberty. "has such a neighbor" = (ADJ @ ok) > 0, replacing
        # the vmapped per-cell neighbor gather.
        ok = (is_empty | can_kill | has_liberty).astype(jnp.int32)
        adj_ok = (ADJ[self.size] @ ok) > 0
        mask = is_empty & adj_ok
        mask = lax.select(state.ko == -1, mask, mask.at[state.ko].set(False))
        return jnp.append(mask, True)  # pass is always legal

    def is_terminal(self, state: GameState) -> Array:
        two_consecutive_pass = state.consecutive_pass_count >= 2
        timeover = self.max_termination_steps <= state.step_count
        return two_consecutive_pass | state.is_psk | timeover

    def rewards(self, state: GameState) -> Array:
        scores = _count_scores(state, self.size)
        is_black_win = scores[0] - self.komi > scores[1]
        rewards = lax.select(is_black_win, jnp.float32([1, -1]), jnp.float32([-1, 1]))
        to_play = state.color
        rewards = lax.select(state.is_psk, jnp.float32([-1, -1]).at[to_play].set(1.0), rewards)
        rewards = lax.select(self.is_terminal(state), rewards, jnp.zeros(2, dtype=jnp.float32))
        return rewards


def _apply_pass(state: GameState) -> GameState:
    return state._replace(consecutive_pass_count=state.consecutive_pass_count + 1)


def _apply_action(state: GameState, action, size) -> GameState:
    state = state._replace(consecutive_pass_count=0)
    my_sign, opp_sign = _signs(state.color)

    # remove killed stones
    adj_ixs = _adj_ixs(action, size)
    adj_ids = state.board[adj_ixs]
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)
    chain_ix = jnp.abs(adj_ids) - 1
    is_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
    single_liberty = (idx_squared_sum[chain_ix] // idx_sum[chain_ix]) - 1
    is_killed = (adj_ixs != -1) & (adj_ids * opp_sign > 0) & is_atari & (single_liberty == action)
    surrounded_stones = (state.board[:, None] == adj_ids) & (is_killed[None, :])
    num_captured = jnp.count_nonzero(surrounded_stones)
    ko_ix = jnp.nonzero(is_killed, size=1)[0][0]
    ko_may_occur = ((adj_ixs == -1) | (state.board[adj_ixs] * opp_sign > 0)).all()
    state = state._replace(
        board=jnp.where(surrounded_stones.any(axis=-1), 0, state.board),
        num_captured=state.num_captured.at[state.color].add(num_captured),
        ko=lax.select(ko_may_occur & (num_captured == 1), adj_ixs[ko_ix], -1),
    )

    # set stone
    state = state._replace(board=state.board.at[action].set((action + 1) * my_sign))

    # merge adjacent chains
    is_my_chain = state.board[adj_ixs] * my_sign > 0
    should_merge = (adj_ixs != -1) & is_my_chain
    new_id = state.board[action]
    tgt_ids = state.board[adj_ixs]
    smallest_id = jnp.min(jnp.where(should_merge, jnp.abs(tgt_ids), 9999))
    smallest_id = jnp.minimum(jnp.abs(new_id), smallest_id) * my_sign
    mask = (state.board == new_id) | (should_merge[None, :] & (state.board[:, None] == tgt_ids[None, :])).any(axis=-1)
    state = state._replace(board=jnp.where(mask, smallest_id, state.board))

    return state


def _count(state: GameState, size):
    # Pseudo-liberty counting. Rewritten from per-cell gathers
    # (state[adj_ixs]) + an (N, N) broadcast-compare into two matmuls
    N = size * size
    board = jnp.abs(state.board)
    is_empty = board == 0
    arange1 = jnp.arange(1, N + 1)
    feats = jnp.stack(
        [is_empty.astype(jnp.int32), jnp.where(is_empty, arange1, 0), jnp.where(is_empty, arange1**2, 0)],
        axis=1,
    )  # (N, 3): per-cell (is_empty, idx, idx^2)

    # Neighbor aggregation as a single matmul (replaces the vmapped gather over adj_ixs).
    neigh = ADJ[size] @ feats  # (N, 3): per-cell (num_pseudo, idx_sum, idx_squared_sum)

    # Segment-sum the per-cell neighbor features by chain id via a one-hot matmul
    # (replaces vmap over (board == x + 1) for every candidate chain id). Class 0 is
    # "empty"; chain id k lives in column k, and the caller indexes result[id - 1].
    onehot = jax.nn.one_hot(board, N + 1, dtype=jnp.int32)  # (N, N+1)
    chain = (onehot.T @ neigh)[1:]  # (N, 3): drop the empty class
    return chain[:, 0], chain[:, 1], chain[:, 2]


def _atari_per_cell(state: GameState, size) -> Array:
    """(N,) bool: True where a cell's chain is in atari (single liberty). Empty cells
    are False. Replaces the per-chain gather `idx_sum[abs(board) - 1]` in
    legal_action_mask with a one-hot matmul: compute the atari predicate per chain,
    then scatter it back to cells. Integer math (see _count) is preserved."""
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)
    atari_chain = (idx_sum**2) == (idx_squared_sum * num_pseudo)  # (N,) indexed by chain_id - 1
    board = jnp.abs(state.board)
    onehot = jax.nn.one_hot(board - 1, size * size, dtype=jnp.int32)  # (N, N); empty -> all-zero row
    return (onehot @ atari_chain.astype(jnp.int32)) > 0


def _signs(color):
    return jnp.int32([[1, -1], [-1, 1]])[color]  # (my_sign, opp_sign)


def _adj_ixs(xy, size):
    dx, dy = jnp.int32([-1, +1, 0, 0]), jnp.int32([0, 0, -1, +1])
    xs, ys = xy // size + dx, xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)  # -1 if out of board


def _compute_hash(state: GameState):
    board = jnp.clip(state.board, -1, 1)
    to_reduce = ZOBRIST_BOARD[board, jnp.arange(board.shape[-1])]
    return lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))


def _is_psk(state: GameState):
    not_passed = state.consecutive_pass_count == 0
    curr_hash = state.hash_history[state.step_count]
    has_same_hash = (curr_hash == state.hash_history).all(axis=-1).sum() > 1
    return not_passed & has_same_hash


def _count_scores(state: GameState, size):
    # Tromp-Taylor area score for both colors. Territory is found by flooding the empty
    # points reachable from each color's opponent; whatever empty stays unreached is that
    # color's territory. The two color floods run on the same graph, so we batch them into
    # a single (N, 2) flood instead of two separate while_loops: column 0 floods from white
    # (-> black's territory), column 1 floods from black (-> white's territory). On the MXU
    # a width-2 matmul costs ~the same as width-1, so this ~halves the dominant flood cost
    # (one dot_general + one reduce_or per iteration instead of two). Bit-exact: the columns
    # are independent and share the same fixed point.
    b0 = jnp.clip(state.board, -1, 1)  # black: +1, white: -1
    boards = jnp.stack([b0, -b0], axis=1)  # (N, 2): per color, my stone: +1, opp stone: -1
    adj = ADJ[size]

    def fill_opp(x):
        b, _ = x  # (N, 2)
        adj_has_opp = (adj @ (b == -1).astype(jnp.int32)) > 0  # one matmul for both colors
        mask = (b == 0) & adj_has_opp
        return jnp.where(mask, -1, b), mask.any()

    boards, _ = lax.while_loop(lambda x: x[1], fill_opp, (boards, True))
    territory = (boards == 0).sum(axis=0)  # (2,): (black territory, white territory)
    stones = jnp.array([jnp.count_nonzero(state.board > 0), jnp.count_nonzero(state.board < 0)])
    return territory + stones


def _count_ji(state: GameState, color: int, size: int):
    # Single-color territory count (empty points reachable only through my stones). Kept
    # for tests; the scoring hot path uses the batched two-color flood in _count_scores.
    board = jnp.clip(state.board * color, -1, 1)  # my stone: 1, opp stone: -1
    adj = ADJ[size]

    def fill_opp(x):
        b, _ = x
        adj_has_opp = (adj @ (b == -1).astype(jnp.int32)) > 0
        mask = (b == 0) & adj_has_opp
        return jnp.where(mask, -1, b), mask.any()

    board, _ = lax.while_loop(lambda x: x[1], fill_opp, (board, True))
    return (board == 0).sum()
