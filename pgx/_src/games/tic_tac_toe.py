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
import jax.numpy as jnp
from jax import Array


class GameState(NamedTuple):
    color: Array = jnp.int32(0)  # 0 = X, 1 = O
    # 0 1 2
    # 3 4 5
    # 6 7 8
    board: Array = -jnp.ones(9, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        board = state.board.at[action].set(state.color)
        idx = jnp.int32([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
        won = (board[idx] == state.color).all(axis=1).any()
        winner = jax.lax.select(won, state.color, -1)
        return state._replace(  # type: ignore
            board=state.board.at[action].set(state.color),
            color=(state.color + 1) % 2,
            winner=winner,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color

        # 1. Get the grid shape (3x3)
        grid = state.board.reshape((3, 3))

        # 2. Create the My/Opponent planes
        my_board = (grid == color)
        opp_board = (grid == (1 - color))

        # 3. Create the Color plane (All 0s for Player 0, All 1s for Player 1)
        # This tells the network "Which player am I?" (First or Second)
        color_plane = jnp.full((3, 3), color, dtype=jnp.bool_)

        # 4. Create the Ones plane (Always all 1s)
        # This helps CNNs handle borders and provides a constant bias
        ones_plane = jnp.ones((3, 3), dtype=jnp.bool_)

        # Stack them to get shape (3, 3, 4)
        return jnp.stack([my_board, opp_board, color_plane, ones_plane], -1)


    def legal_action_mask(self, state: GameState) -> Array:
        return state.board < 0

    def is_terminal(self, state: GameState) -> Array:
        return (state.winner >= 0) | jnp.all(state.board != -1)

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )
