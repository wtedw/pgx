from functools import partial
import jax
import jax.numpy as jnp
from typing import Literal
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = None  # Will be set based on size
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = None  # Will be set based on size
    _step_count: Array = jnp.int32(0)
    # --- Hex specific ---
    _size: Array = None  # Will be set based on size
    _turn: Array = jnp.int32(0)
    _board: Array = None  # Will be set based on size

    @property
    def env_id(self) -> core.EnvId:
        return f"hexnoswap_{self._size}x{self._size}"

class Hexnoswap(core.Env):
    def __init__(self, *, size: int = 11):
        super().__init__()
        assert isinstance(size, int)
        assert 3 <= size <= 17, "Hex board size must be between 3 and 17"
        self.size = size

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(0)  # First player always starts
        size = self.size
        return State(
            current_player=current_player,
            observation=jnp.zeros((size, size, 3), dtype=jnp.bool_),  # Removed swap channel
            legal_action_mask=jnp.ones(size * size, dtype=jnp.bool_),  # Removed swap action
            _size=jnp.int32(size),
            _board=jnp.zeros(size * size, jnp.int32)
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return _step(state, action, size=self.size)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id, size=self.size)

    @property
    def id(self) -> core.EnvId:
        return f"hexnoswap_{self.size}x{self.size}"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

def _step(state: State, action: Array, size: int) -> State:
    set_place_id = action + 1
    board = state._board.at[action].set(set_place_id)
    neighbour = _neighbour(action, size)

    def merge(i, b):
        adj_pos = neighbour[i]
        return jax.lax.cond(
            (adj_pos >= 0) & (b[adj_pos] > 0),
            lambda: jnp.where(b == b[adj_pos], set_place_id, b),
            lambda: b,
        )

    board = jax.lax.fori_loop(0, 6, merge, board)
    won = _is_game_end(board, size, state._turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    state = state.replace(
        current_player=1 - state.current_player,
        _turn=1 - state._turn,
        _board=board * -1,
        rewards=reward,
        terminated=won,
        legal_action_mask=state.legal_action_mask.at[:].set(board == 0),
    )

    return state

def _observe(state: State, player_id: Array, size) -> Array:
    board = jax.lax.select(
        player_id == state.current_player,
        state._board.reshape((size, size)),
        -state._board.reshape((size, size)),
    )

    my_board = board * 1 > 0
    opp_board = board * -1 > 0
    ones = jnp.ones_like(my_board)
    color = jax.lax.select(player_id == state.current_player, state._turn, 1 - state._turn)
    color = color * ones

    return jnp.stack([my_board, opp_board, color], 2, dtype=jnp.bool_)

def _neighbour(xy, size):
    """
        (x,y-1)   (x+1,y-1)
    (x-1,y)    (x,y)    (x+1,y)
       (x-1,y+1)   (x,y+1)
    """
    x = xy // size
    y = xy % size
    xs = jnp.array([x, x + 1, x - 1, x + 1, x - 1, x])
    ys = jnp.array([y - 1, y - 1, y, y, y + 1, y + 1])
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)

def _is_game_end(board, size, turn):
    top, bottom = jax.lax.cond(
        turn == 0,
        lambda: (board[:size], board[-size:]),
        lambda: (board[::size], board[size - 1 :: size]),
    )

    def check_same_id_exist(_id):
        return (_id > 0) & (_id == bottom).any()

    return jax.vmap(check_same_id_exist)(top).any()