# Optimised Hex environment *with* swap support.

from functools import partial
import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE  = jnp.bool_(True)

@dataclass
class State(core.State):
    current_player   : Array = jnp.int32(0)
    observation      : Array = None          # set in _init
    rewards          : Array = jnp.float32([0.0, 0.0])
    terminated       : Array = FALSE
    truncated        : Array = FALSE
    legal_action_mask: Array = None          # set in _init
    _step_count      : Array = jnp.int32(0)  # increments every move
    # --- Hex specific ---
    _size  : Array = None                    # set in _init
    _turn  : Array = jnp.int32(1)            # 1 ≙ “South–North”, 0 ≙ “West–East”
    _board : Array = None                    # flat array, 0 = empty, >0 = stone ID

    @property
    def env_id(self) -> core.EnvId:
        return f"hexswap_{int(self._size)}x{int(self._size)}"


class Hexswap(core.Env):
    """Hex with pie‑rule swap.  Second player may choose the swap action once."""

    # ‑‑‑ meta data ‑‑‑ -------------------------------------------------------
    @property
    def id(self) -> core.EnvId:  return f"hexswap_{self.size}x{self.size}"
    @property
    def version(self) -> str   : return "v0"
    @property
    def num_players(self) -> int: return 2
    # -------------------------------------------------------------------------

    def __init__(self, *, size: int = 11):
        super().__init__()
        assert 3 <= size <= 17, "Hex board size must be between 3 and 17"
        self.size = int(size)

    # ------------------------- core API --------------------------------------
    def _init(self, key: PRNGKey) -> State:          # (no randomness needed)
        size  = self.size
        board = jnp.zeros(size * size, jnp.int32)

        # chan‑0  my stones, chan‑1  opp stones, chan‑2  colour, chan‑3  to‑play
        obs_shape = (size, size, 4)

        # legal actions:  size² board positions  +1  “swap”
        legal = jnp.ones(size * size + 1, dtype=jnp.bool_)
        legal = legal.at[-1].set(FALSE)              # swap not allowed on move‑0

        return State(
            current_player   = jnp.int32(0),         # Black always starts
            observation      = jnp.zeros(obs_shape, jnp.bool_),
            legal_action_mask= legal,
            _size            = jnp.int32(size),
            _board           = board,
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return jax.lax.cond(
            action == self.size * self.size,      # last index ⇒ swap
            lambda: _swap(state, self.size),
            lambda: _place(state, action, self.size)
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id, self.size)
    # -------------------------------------------------------------------------


# ============================= helpers =======================================

def _place(state: State, action: Array, size: int) -> State:
    """Regular stone placement."""
    set_place_id   = action + 1                              # IDs are 1‑based
    one_hot_action = jax.nn.one_hot(action, state._board.size, dtype=state._board.dtype)
    board          = state._board + one_hot_action * set_place_id

    # fast union–find merge (same as before)
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
    reward = jax.lax.cond(won,
                          lambda: jnp.ones(2, jnp.float32),
                          lambda: jnp.zeros(2, jnp.float32))

    legal     = jnp.concatenate([
                    board == 0,                     # empty cells
                    jnp.array([state._step_count == 1], jnp.bool_)  # swap allowed only once
               ])

    return state.replace(
        current_player   = 1 - state.current_player,
        _turn            = 1 - state._turn,
        _board           = board * -1,              # point of view trick
        rewards          = reward,
        terminated       = won,
        legal_action_mask= legal,
    )


def _swap(state: State, size: int) -> State:
    """Pie‑rule swap: the second player takes the first stone."""
    # locate the single existing stone
    ix        = jnp.nonzero(state._board, size=1)[0]
    row, col  = ix // size, ix % size
    new_ix    = col * size + row                     # transpose (r,c) -> (c,r)
    set_id    = new_ix + 1

    board = (state._board
             .at[ix     ].set(0)
             .at[new_ix ].set(set_id))

    legal = jnp.concatenate([board == 0, jnp.array([FALSE], jnp.bool_)])

    return state.replace(
        current_player   = 1 - state.current_player,
        _turn            = 1 - state._turn,
        _board           = board * -1,
        legal_action_mask= legal,
    )


def _observe(state: State, player_id: Array, size: int) -> Array:
    board = jax.lax.select(
        player_id == state.current_player,
        state._board.reshape((size, size)),
        -state._board.reshape((size, size)),
    )
    my_board  = board > 0
    opp_board = board < 0
    colour    = jax.lax.select(player_id == state.current_player,
                               state._turn, 1 - state._turn)
    colour    = jnp.broadcast_to(colour, my_board.shape)

    to_play   = jnp.ones_like(my_board)         # always 1 (your own POV)

    return jnp.stack([my_board, opp_board, colour, to_play], axis=2).astype(jnp.bool_)


# ---------- geometry / win‑check helpers (unchanged) -------------------------
def _neighbour(xy, size):
    x, y  = xy // size, xy % size
    xs    = jnp.array([x, x + 1, x - 1, x + 1, x - 1, x])
    ys    = jnp.array([y - 1, y - 1, y, y, y + 1, y + 1])
    on    = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on, xs * size + ys, -1)


def _is_game_end(board, size, turn):
    top, bottom = jax.lax.cond(
        turn == 1,
        lambda: (board[:size],               board[-size:]),
        lambda: (board[::size],              board[size-1::size]),
    )

    def connected(stone_id):
        return (stone_id > 0) & (stone_id == bottom).any()

    return jax.vmap(connected)(top).any()
# ============================================================================
