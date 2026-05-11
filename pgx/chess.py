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

import warnings

import jax
import jax.numpy as jnp

from typing import Optional

import pgx.core as core
from pgx._src.games.chess import INIT_LEGAL_ACTION_MASK, Game, GameState, _flip
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pgx.experimental.cutils import pack_mask

INIT_LEGAL_ACTION_BITMASK = pack_mask(INIT_LEGAL_ACTION_MASK)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Optional[Array] = None  # not stored; use legal_action_bitmask
    legal_action_bitmask: Optional[Array] = INIT_LEGAL_ACTION_BITMASK  # 146 uint32 words
    # observation: Array = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])  # [0, 1] or [1, 0]
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "chess"

    @staticmethod
    def _from_fen(fen: str):
        from pgx.experimental.chess import from_fen

        warnings.warn(
            "State._from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
            DeprecationWarning,
        )
        return from_fen(fen)

    def _to_fen(self) -> str:
        from pgx.experimental.chess import to_fen

        warnings.warn(
            "State._to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
            DeprecationWarning,
        )
        return to_fen(self)


class Chess(core.Env):
    def __init__(self, *, auto_terminate: bool = True):
        super().__init__()
        self.game = Game(auto_terminate=auto_terminate)

    def _init(self, key: PRNGKey) -> State:
        x = GameState()
        # Always keep the same player order so that column 0 is Player 1
        _player_order = jnp.int32([0, 1])
        state = State(  # type: ignore
            current_player=_player_order[x.color],
            _player_order=_player_order,
            _x=x,
        )
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self.game.step(state._x, action)
        legal_action_mask = self.game.legal_action_mask(x)
        state = state.replace(  # type: ignore
            _x=x,
            legal_action_bitmask=pack_mask(legal_action_mask),
            terminated=self.game.is_terminal(x, legal_action_mask),
            rewards=self.game.rewards(x, legal_action_mask)[state._player_order],
            current_player=state._player_order[x.color],
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        color = jax.lax.select(state.current_player == player_id, state._x.color, 1 - state._x.color)
        x = jax.lax.cond(state.current_player == player_id, lambda: state._x, lambda: _flip(state._x))
        return self.game.observe(x, color)

    def _check_legality(self, state: core.State, action: Array) -> Array:
        assert isinstance(state, State)
        word = action // jnp.int32(32)
        bit = (action % jnp.int32(32)).astype(jnp.uint32)
        return ((state.legal_action_bitmask[word] >> bit) & jnp.uint32(1)).astype(jnp.bool_)

    def _set_terminal_mask(self, state: core.State) -> core.State:
        assert isinstance(state, State)
        return state.replace(legal_action_bitmask=jnp.full_like(state.legal_action_bitmask, jnp.iinfo(jnp.uint32).max))  # type: ignore

    @property
    def id(self) -> core.EnvId:
        return "chess"

    @property
    def version(self) -> str:
        return "v2"

    @property
    def num_actions(self) -> int:
        return 4672

    @property
    def num_players(self) -> int:
        return 2


def _from_fen(fen: str):
    from pgx.experimental.chess import from_fen

    warnings.warn(
        "_from_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.from_fen instead.",
        DeprecationWarning,
    )
    return from_fen(fen)


def _to_fen(state: State):
    from pgx.experimental.chess import to_fen

    warnings.warn(
        "_to_fen is deprecated. Will be removed in the future release. Please use pgx.experimental.chess.to_fen instead.",
        DeprecationWarning,
    )
    return to_fen(state)
