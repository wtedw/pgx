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
    observation: Array = None  # Will be set based on board size and history channels.
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = None  # Will be set based on board size.
    _step_count: Array = jnp.int32(0)
    # --- Hex-specific fields ---
    _size: Array = None  # Board size.
    _turn: Array = jnp.int32(0)
    _board: Array = None  # Current board state (flattened).
    _board_history: Array = None  # History of past boards (each a flattened board).

    @property
    def env_id(self) -> core.EnvId:
        return f"hexnoswap_{self._size}x{self._size}"

class Hexnoswap(core.Env):
    def __init__(self, *, size: int = 11, history_length: int = 4):
        super().__init__()
        assert isinstance(size, int)
        assert 3 <= size <= 17, "Hex board size must be between 3 and 17"
        self.size = size
        self.history_length = history_length

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(0)  # First player always starts
        size = self.size
        history_length = self.history_length
        # Initialize board: all positions are 0 (empty)
        board = jnp.zeros(size * size, jnp.int32)
        # Initialize board history: fill with the initial board (all zeros)
        board_history = jnp.zeros((history_length, size * size), jnp.int32)
        board_history = board_history.at[0].set(board)
        # Define the observation shape.
        # Here we will use 2 channels per history board plus 3 channels for the current board:
        #   - 2 channels per board: one for positions with a positive (own) marker,
        #                        one for positions with a negative (opp) marker.
        #   - 3 channels for the current board: same as above plus one constant channel for turn.
        obs_channels = history_length * 2 + 3
        observation = jnp.zeros((size, size, obs_channels), dtype=jnp.bool_)
        legal_action_mask = jnp.ones(size * size, dtype=jnp.bool_)
        return State(
            current_player=current_player,
            observation=observation,
            legal_action_mask=legal_action_mask,
            _size=jnp.int32(size),
            _turn=jnp.int32(0),
            _board=board,
            _board_history=board_history
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        state = _step(state, action, size=self.size)
        state = _update_history(state, history_length=self.history_length, size=self.size)
        return state

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id, size=self.size, history_length=self.history_length)

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
    # Place a piece: the action indicates the board index to mark.
    set_place_id = action + 1
    board = state._board.at[action].set(set_place_id)
    neighbour = _neighbour(action, size)

    # Merge adjacent pieces if they have the same ID.
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
        _board=board * -1,  # flip the board so that the perspective alternates.
        rewards=reward,
        terminated=won,
        legal_action_mask=state.legal_action_mask.at[:].set(board == 0),
    )
    return state

def _update_history(state: State, history_length: int, size: int) -> State:
    # Roll the board history along the 0th axis and insert the current board at index 0.
    board_history = jnp.roll(state._board_history, shift=1, axis=0)
    board_history = board_history.at[0].set(state._board)
    return state.replace(_board_history=board_history)

def _observe(state: State, player_id: Array, size: int, history_length: int) -> Array:
    # For a consistent observation from the perspective of the given player,
    # we flip the board history (and current board) if player_id does not match state.current_player.
    def transform(board):
        return jax.lax.select(player_id == state.current_player, board, -board)

    current_board = transform(state._board).reshape((size, size))
    hist = transform(state._board_history).reshape((history_length, size, size))

    # Define a simple feature extractor for a board:
    # Two channels: one where positions with positive values (own pieces) are True,
    # and one where positions with negative values (opponent pieces) are True.
    def board_features(board):
        my_board = board > 0
        opp_board = board < 0
        return jnp.stack([my_board, opp_board], axis=-1)  # shape: (size, size, 2)

    # Apply feature extraction to each board in the history.
    hist_feat = jax.vmap(board_features)(hist)  # shape: (history_length, size, size, 2)
    # Reshape history features to have shape: (size, size, history_length * 2)
    hist_feat = jnp.transpose(hist_feat, (1, 2, 0, 3)).reshape((size, size, history_length * 2))

    # Extract features for the current board.
    current_feat = board_features(current_board)  # shape: (size, size, 2)
    # Add one extra constant channel encoding the current turn (from the observer’s perspective).
    turn_channel = jnp.full((size, size, 1), fill_value=(player_id == state.current_player), dtype=jnp.bool_)
    current_feat = jnp.concatenate([current_feat, turn_channel], axis=-1)  # shape: (size, size, 3)

    # Concatenate the history features and the current board features.
    obs = jnp.concatenate([hist_feat, current_feat], axis=-1)
    return obs

def _neighbour(xy, size: int):
    """
        Neighbour positions for a given cell (using axial/offset coordinates):
            (x, y-1)    (x+1, y-1)
        (x-1, y)     (x, y)     (x+1, y)
           (x-1, y+1)    (x, y+1)
    """
    x = xy // size
    y = xy % size
    xs = jnp.array([x, x + 1, x - 1, x + 1, x - 1, x])
    ys = jnp.array([y - 1, y - 1, y, y, y + 1, y + 1])
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)

def _is_game_end(board, size: int, turn: int):
    # In Hex, one winning condition is that one edge is connected to the opposite edge.
    top, bottom = jax.lax.cond(
        turn == 0,
        lambda: (board[:size], board[-size:]),
        lambda: (board[::size], board[size - 1 :: size]),
    )
    def check_same_id_exist(_id):
        return (_id > 0) & (_id == bottom).any()
    return jax.vmap(check_same_id_exist)(top).any()
