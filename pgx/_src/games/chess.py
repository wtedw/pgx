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
import numpy as np
from jax import Array, lax

EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))  # opponent: -1 * piece
# MAX_TERMINATION_STEPS = 512  # from AlphaZero paper
MAX_TERMINATION_STEPS = 256  # from AlphaZero paper

# prepare precomputed values here (e.g., available moves, map to label, etc.)

# index: a1: 0, a2: 1, ..., h8: 63
INIT_BOARD = jnp.int32([4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2, 3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5, 6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3, 2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4])  # fmt: skip
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h

# Action: AlphaZero style label (4672 = 64 x 73)
# * [0:9]  underpromotions
#     plane // 3 == 0: rook, 1: bishop, 2: knight
#     plane  % 3 == 0: up  , 1: right,  2: left
# * [9:73] normal moves (queen:56 + knight:8)
#   51                   22                   50
#      52                21                49
#         53             20             48
#            54          19          47
#               55       18       46
#                  56    17    45
#                     57 16 44
#   23 24 25 26 27 28 29  X 30 31 32 33 34 35 36
#                     43 15 58
#                  42    14    59
#               41       13       60
#            40          12          61
#         39             11             62
#      38                10                64
#   37                    9                   64
FROM_PLANE = -np.ones((64, 73), dtype=np.int32)
TO_PLANE = -np.ones((64, 64), dtype=np.int32)  # ignores underpromotion
zeros, seq, rseq = [0] * 7, list(range(1, 8)), list(range(-7, 0))
# down, up, left, right, down-left, down-right, up-right, up-left, knight, and knight
dr = rseq[::] + seq[::] + zeros[::] + zeros[::] + rseq[::] + seq[::] + seq[::-1] + rseq[::-1]
dc = zeros[::] + zeros[::] + rseq[::] + seq[::] + rseq[::] + seq[::] + rseq[::] + seq[::]
dr += [-1, +1, -2, +2, -1, +1, -2, +2]
dc += [-2, -2, -1, -1, +2, +2, +1, +1]
for from_ in range(64):
    for plane in range(73):
        if plane < 9:  # underpromotion
            to = from_ + [+1, +9, -7][plane % 3] if from_ % 8 == 6 else -1
            if 0 <= to < 64:
                FROM_PLANE[from_, plane] = to
        else:  # normal moves
            r = from_ % 8 + dr[plane - 9]
            c = from_ // 8 + dc[plane - 9]
            if 0 <= r < 8 and 0 <= c < 8:
                to = c * 8 + r
                FROM_PLANE[from_, plane] = to
                TO_PLANE[from_, to] = plane

INIT_LEGAL_ACTION_MASK = np.zeros(64 * 73, dtype=np.bool_)
ixs = [89, 90, 652, 656, 673, 674, 1257, 1258, 1841, 1842, 2425, 2426, 3009, 3010, 3572, 3576, 3593, 3594, 4177, 4178]
INIT_LEGAL_ACTION_MASK[ixs] = True

LEGAL_DEST = -np.ones((7, 64, 27), np.int32)  # LEGAL_DEST[0, :, :] == -1
LEGAL_DEST_NEAR = -np.ones((64, 16), np.int32)  # king and knight moves
LEGAL_DEST_FAR = -np.ones((64, 19), np.int32)   # queen moves except king moves
CAN_MOVE = np.zeros((7, 64, 64), dtype=np.bool_)
for from_ in range(64):
    legal_dest = {p: [] for p in range(7)}
    for to in range(64):
        if from_ == to:
            continue
        r0, c0, r1, c1 = from_ % 8, from_ // 8, to % 8, to // 8
        if (r1 - r0 == 1 and abs(c1 - c0) <= 1) or ((r0, r1) == (1, 3) and abs(c1 - c0) == 0):
            legal_dest[PAWN].append(to)
        if (abs(r1 - r0) == 1 and abs(c1 - c0) == 2) or (abs(r1 - r0) == 2 and abs(c1 - c0) == 1):
            legal_dest[KNIGHT].append(to)
        if abs(r1 - r0) == abs(c1 - c0):
            legal_dest[BISHOP].append(to)
        if abs(r1 - r0) == 0 or abs(c1 - c0) == 0:
            legal_dest[ROOK].append(to)
        if (abs(r1 - r0) == 0 or abs(c1 - c0) == 0) or (abs(r1 - r0) == abs(c1 - c0)):
            legal_dest[QUEEN].append(to)
        if from_ != to and abs(r1 - r0) <= 1 and abs(c1 - c0) <= 1:
            legal_dest[KING].append(to)
    for p in range(1, 7):
        LEGAL_DEST[p, from_, : len(legal_dest[p])] = legal_dest[p]
        CAN_MOVE[p, from_, legal_dest[p]] = True
    dests = list(set(legal_dest[KING]) | set(legal_dest[KNIGHT]))
    LEGAL_DEST_NEAR[from_, : len(dests)] = dests
    dests = list(set(legal_dest[QUEEN]).difference(set(legal_dest[KING])))
    LEGAL_DEST_FAR[from_, : len(dests)] = dests

BETWEEN = -np.ones((64, 64, 6), dtype=np.int32)
for from_ in range(64):
    for to in range(64):
        r0, c0, r1, c1 = from_ % 8, from_ // 8, to % 8, to // 8
        if not (abs(r1 - r0) == 0 or abs(c1 - c0) == 0 or abs(r1 - r0) == abs(c1 - c0)):
            continue
        dr, dc = max(min(r1 - r0, 1), -1), max(min(c1 - c0, 1), -1)
        for i in range(6):
            r, c = r0 + dr * (i + 1), c0 + dc * (i + 1)
            if r == r1 and c == c1:
                break
            BETWEEN[from_, to, i] = c * 8 + r

FROM_PLANE, TO_PLANE, INIT_LEGAL_ACTION_MASK, LEGAL_DEST, LEGAL_DEST_NEAR, LEGAL_DEST_FAR, CAN_MOVE, BETWEEN = (
    jnp.array(x) for x in (FROM_PLANE, TO_PLANE, INIT_LEGAL_ACTION_MASK, LEGAL_DEST, LEGAL_DEST_NEAR, LEGAL_DEST_FAR, CAN_MOVE, BETWEEN)
)

FROM_PLANE_FLAT = FROM_PLANE.flatten() # Shape: (4672,)
TO_PLANE_FLAT = TO_PLANE.flatten()              # shape (64*64 = 4096,)
LEGAL_DEST_FLAT = LEGAL_DEST.reshape(-1, 27)             # (7*64, 27)

keys = jax.random.split(jax.random.PRNGKey(12345), 4)
ZOBRIST_BOARD = jax.random.randint(keys[0], shape=(64, 13, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_SIDE = jax.random.randint(keys[1], shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_CASTLING = jax.random.randint(keys[2], shape=(4, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_EN_PASSANT = jax.random.randint(keys[3], shape=(65, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
INIT_ZOBRIST_HASH = jnp.uint32([1455170221, 1478960862])


class GameState(NamedTuple):
    color: Array = jnp.int32(0)  # w: 0, b: 1
    board: Array = INIT_BOARD  # (64,)
    castling_rights: Array = jnp.ones([2, 2], dtype=jnp.bool_)  # my queen, my king, opp queen, opp king
    en_passant: Array = jnp.int32(-1)
    halfmove_count: Array = jnp.int32(0)  # number of moves since the last piece capture or pawn move
    fullmove_count: Array = jnp.int32(1)  # increase every black move
    hash_history: Array = jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(INIT_ZOBRIST_HASH)
    board_history: Array = jnp.zeros((8, 64), dtype=jnp.int32).at[0, :].set(INIT_BOARD)
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    step_count: Array = jnp.int32(0)


class Action(NamedTuple):
    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    underpromotion: Array = jnp.int32(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: Array):
        from_, plane = label // 73, label % 73
        underpromotion = lax.select(plane >= 9, -1, plane // 3)
        one_hot = jax.nn.one_hot(label, 4672, dtype=FROM_PLANE_FLAT.dtype)
        to = one_hot @ FROM_PLANE_FLAT           # (<batch>,) result

        return Action(from_=from_, to=to, underpromotion=underpromotion)

    def _to_label(self):
        flat_idx = self.from_ * 64 + self.to            # 0 … 4095
        plane    = (jax.nn.one_hot(flat_idx, 4096,      # one-hot × table
                                   dtype=TO_PLANE_FLAT.dtype)
                    @ TO_PLANE_FLAT)                    # same shape as flat_idx

        return self.from_ * 73 + plane


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        state = _apply_move(state, Action._from_label(action))
        state = _flip(state)
        state = _update_history(state)
        state = state._replace(legal_action_mask=_legal_action_mask(state))
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        """
        This function is modified to remove gathers.
        The inner `make` function is vmapped over board and hash history slices
        directly, avoiding explicit indexing like `state.board_history[i]`.
        """
        if color is None:
            color = state.color
        ones = jnp.ones((1, 8, 8), dtype=jnp.float32)

        def make(board_history_slice, hash_history_slice):
            # board_history_slice is (64,), hash_history_slice is (2,)
            board = jnp.rot90(board_history_slice.reshape((8, 8)), k=1)

            def piece_feat(p):
                return (board == p).astype(jnp.float32)

            my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
            opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

            h = hash_history_slice
            # Compare current h against the entire history to find repetitions
            rep = (state.hash_history == h).all(axis=1).sum() - 1
            rep = lax.select((h == 0).all(), 0, rep)
            rep0 = ones * (rep == 0)
            rep1 = ones * (rep >= 1)
            return jnp.vstack([my_pieces, opp_pieces, rep0, rep1])

        # Vmap over the first 8 history states and their corresponding hashes
        board_features = jax.vmap(make)(state.board_history, state.hash_history[:8]).reshape(-1, 8, 8)

        return jnp.vstack(
            [
                board_features,
                color * ones,
                (state.step_count / MAX_TERMINATION_STEPS) * ones,
                state.castling_rights.flatten()[:, None, None] * ones,
                (state.halfmove_count.astype(jnp.float32) / 100.0) * ones,
            ]
        ).transpose((1, 2, 0))

    def legal_action_mask(self, state: GameState) -> Array:
        return state.legal_action_mask

    def is_terminal(self, state: GameState) -> Array:
        terminated = ~state.legal_action_mask.any()
        terminated |= state.halfmove_count >= 100
        terminated |= has_insufficient_pieces(state)
        rep = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        terminated |= rep >= 2
        terminated |= MAX_TERMINATION_STEPS <= state.step_count
        return terminated

    def rewards(self, state: GameState) -> Array:
        is_checkmate = (~state.legal_action_mask.any()) & _is_checked(state)
        return lax.select(
            is_checkmate,
            jnp.ones(2, dtype=jnp.float32).at[state.color].set(-1),
            jnp.zeros(2, dtype=jnp.float32),
        )


def _update_history(state: GameState):
    board_history = jnp.roll(state.board_history, 64)
    board_history = board_history.at[0].set(state.board)
    hash_hist = jnp.roll(state.hash_history, 2)
    hash_hist = hash_hist.at[0].set(_zobrist_hash(state))
    return state._replace(board_history=board_history, hash_history=hash_hist)


def has_insufficient_pieces(state: GameState):
    # uses the same condition as OpenSpiel
    num_pieces = (state.board != EMPTY).sum()
    num_pawn_rook_queen = ((jnp.abs(state.board) >= ROOK) | (jnp.abs(state.board) == PAWN)).sum() - 2  # two kings
    num_bishop = (jnp.abs(state.board) == BISHOP).sum()
    coords = jnp.arange(64).reshape((8, 8))
    black_coords = jnp.hstack((coords[::2, ::2].ravel(), coords[1::2, 1::2].ravel()))
    pieces_on_black = _pieces_at(state.board, black_coords)
    num_bishop_on_black = (jnp.abs(pieces_on_black) == BISHOP).sum()
    is_insufficient = False
    # king vs king
    is_insufficient |= num_pieces <= 2
    # king vs king + (knight or bishop)
    is_insufficient |= (num_pieces == 3) & (num_pawn_rook_queen == 0)
    # king + bishop* vs king + bishop* (bishops are on same color tile)
    is_bishop_all_on_black = num_bishop_on_black == num_bishop
    is_bishop_all_on_white = num_bishop_on_black == 0
    is_insufficient |= (num_pieces == num_bishop + 2) & (is_bishop_all_on_black | is_bishop_all_on_white)

    return is_insufficient


def _apply_move(state: GameState, a: Action) -> GameState:
    piece = _pieces_at(state.board, a.from_)
    # en passant
    is_en_passant = (state.en_passant >= 0) & (piece == PAWN) & (state.en_passant == a.to)
    removed_pawn_pos = a.to - 1
    state = state._replace(
        # Select between the new updated board and the original board
        # to avoid gather operation `state.board[removed_pawn_pos]`
        board=lax.select(
            is_en_passant,
            _set_pieces(state.board, jnp.array([removed_pawn_pos],  jnp.int32), jnp.array([EMPTY], jnp.int32)),
            state.board
        )
    )
    is_en_passant = (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2)
    state = state._replace(en_passant=lax.select(is_en_passant, (a.to + a.from_) // 2, -1))
    # update counters
    captured = (_pieces_at(state.board, a.to) < 0) | is_en_passant
    state = state._replace(
        halfmove_count=lax.select(captured | (piece == PAWN), 0, state.halfmove_count + 1),
        fullmove_count=state.fullmove_count + jnp.int32(state.color == 1),
    )
    # castling
    board = state.board
    is_queen_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 16)
    board_q         = _set_pieces(_set_pieces(board, 0, EMPTY), 24, ROOK)
    board           = lax.select(is_queen_side_castling, board_q, board)

    is_king_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 48)
    board_k         = _set_pieces(_set_pieces(board, 56, EMPTY), 40, ROOK)
    board           = lax.select(is_king_side_castling, board_k, board)

    state = state._replace(board=board)
    # update castling rights
    cond = jnp.bool_([[(a.from_ != 32) & (a.from_ != 0), (a.from_ != 32) & (a.from_ != 56)], [a.to != 7, a.to != 63]])
    state = state._replace(castling_rights=state.castling_rights & cond)
    # promotion to queen
    piece = lax.select((piece == PAWN) & (a.from_ % 8 == 6) & (a.underpromotion < 0), QUEEN, piece)
    # underpromotion
    is_underpromotion = a.underpromotion >= 0

    # Start with a default value (can be anything, it will be overwritten)
    promoted_piece = piece

    # Unroll the selection logic using a series of conditional checks
    promoted_piece = jnp.where(a.underpromotion == 0, ROOK,   promoted_piece)
    promoted_piece = jnp.where(a.underpromotion == 1, BISHOP, promoted_piece)
    promoted_piece = jnp.where(a.underpromotion == 2, KNIGHT, promoted_piece)

    # Only apply the update if it was an underpromotion, otherwise keep the original piece value
    piece = jnp.where(is_underpromotion, promoted_piece, piece)

    # actually move
    board = _set_pieces(board, a.from_, EMPTY)
    board = _set_pieces(board, a.to,    piece)
    state = state._replace(board=board)

    return state


def _flip_pos(x: Array):  # e.g., 37 <-> 34, -1 <-> -1
    return lax.select(x == -1, x, (x // 8) * 8 + (7 - (x % 8)))


def _flip(state: GameState) -> GameState:
    return state._replace(
        board=-jnp.flip(state.board.reshape(8, 8), axis=1).flatten(),
        color=(state.color + 1) % 2,
        en_passant=_flip_pos(state.en_passant),
        castling_rights=state.castling_rights[::-1],
        board_history=-jnp.flip(state.board_history.reshape(-1, 8, 8), axis=-1).reshape(-1, 64),
    )


# ---------------------------------------------------------------------
# helper: gather-free LEGAL_DEST[piece, from_]  ➜  (27,) int32
# ---------------------------------------------------------------------

def _legal_dest(piece: Array, frm: Array) -> Array:
    flat = piece * 64 + frm                              # 0 … 447
    oh   = jax.nn.one_hot(flat, 7 * 64, dtype=jnp.int32) # (448,)
    return oh @ LEGAL_DEST_FLAT                          # (27,)

def _legal_action_mask(state: GameState) -> Array:
    def legal_normal_moves(from_):
        piece = _pieces_at(state.board, from_)

        def legal_label(to):
            ok = (from_ >= 0) & (piece > 0) & (to >= 0) & (_pieces_at(state.board, to) <= 0)

            # Replace the slow BETWEEN gather with the on-the-fly calculation
            path_is_clear = _is_path_clear(state.board, from_, to)

            # Since the CAN_MOVE array is large, the one hot + matmul trick is not as performant.
            # We instead compute the `can_move` boolean on the fly
            can_move_bool = _can_move_on_the_fly(piece, from_, to)
            ok &= can_move_bool & path_is_clear

            c0, c1 = from_ // 8, to // 8
            pawn_should = ((c1 == c0) & (_pieces_at(state.board, to) == EMPTY)) | ((c1 != c0) & (_pieces_at(state.board, to) < 0))
            ok &= (piece != PAWN) | pawn_should

            return lax.select(ok, Action(from_=from_, to=to)._to_label(), -1)

        return jax.vmap(legal_label)(_legal_dest(piece, from_))

    def legal_en_passants():
        to = state.en_passant

        def legal_labels(from_):
            ok = (from_ >= 0) & (from_ < 64) & (to >= 0) & (_pieces_at(state.board, from_) == PAWN) & (_pieces_at(state.board, to - 1) == -PAWN)
            a = Action(from_=from_, to=to)
            return lax.select(ok, a._to_label(), -1)

        return jax.vmap(legal_labels)(jnp.int32([to - 9, to + 7]))

    def is_not_checked(label):
        a = Action._from_label(label)
        return ~_is_checked(_apply_move(state, a))

    def legal_underpromotions(mask):
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (_pieces_at(state.board, a.from_) == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to)._to_label()]
            return lax.select(ok, label, -1)

        labels = jnp.int32([from_ * 73 + i for i in range(9) for from_ in [6, 14, 22, 30, 38, 46, 54, 62]])
        return jax.vmap(legal_labels)(labels)

    # normal move and en passant

    # The original jnp.nonzero(..., size=...) is slow. This pattern replaces it.
    # 1. Create a boolean mask of our pieces.
    is_my_piece = state.board > 0

    # 2. Create an array where valid positions have their index, others have -1.
    indices_or_sentinel = jnp.where(is_my_piece, jnp.arange(64), -1)

    # 3. Use top_k to sort the valid indices to the front.
    # Surprisingly, this is much faster.
    possible_piece_positions, _ = lax.top_k(indices_or_sentinel, k=16)

    a1 = jax.vmap(legal_normal_moves)(possible_piece_positions).flatten()
    a2 = legal_en_passants()
    actions = jnp.hstack((a1, a2))  # include -1
    # filter out -1. 200 is big enough for normal play.

    # The original code used jnp.nonzero to filter out -1s, which is slow.
    # We replace it with lax.top_k, which is a single, fast sort operation.
    # It efficiently collects all valid moves (>=0) at the front of a fixed-size array.
    actions, _ = lax.top_k(actions, k=200)


    # Filter actions by checking for suicides (moves that leave the king in check).
    # The result is a dense array of valid action labels, padded with -1.
    valid_actions = jnp.where(jax.vmap(is_not_checked)(actions), actions, -1)

    # 1. Create the base update mask from standard legal moves.
    # This converts the array of action labels into a single boolean mask.
    base_update_mask = jax.nn.one_hot(
        valid_actions, 64 * 73 + 1, dtype=jnp.bool_
    ).any(axis=0)

    # 2. Create the update mask for legal underpromotions.
    # The legal_underpromotions function needs a mask to check against.
    underpromo_actions = legal_underpromotions(base_update_mask)
    underpromo_mask = jax.nn.one_hot(
        underpromo_actions, 64 * 73 + 1, dtype=jnp.bool_
    ).any(axis=0)


    # 3. Create update masks for the two castling moves (without scattering).
    # Condition 1 & 2 We check if castling is legal and if the relevant squares are not attacked.
    b = state.board
    can_castle_queen_side = state.castling_rights[0, 0]
    q_indices = jnp.int32([0, 8, 16, 24, 32])
    q_pieces = _pieces_at(b, q_indices)
    q_expected = jnp.int32([ROOK, EMPTY, EMPTY, EMPTY, KING])
    can_castle_queen_side &= (q_pieces == q_expected).all()

    can_castle_king_side = state.castling_rights[0, 1]
    k_indices = jnp.int32([32, 40, 48, 56])
    k_pieces = _pieces_at(b, k_indices)
    k_expected = jnp.int32([KING, EMPTY, EMPTY, ROOK])
    can_castle_king_side &= (k_pieces == k_expected).all()

    ### Create a boolean mask for each castling move directly.
    # Condition 3: Check if king passes through or into check
    not_checked = ~jax.vmap(_is_attacked, in_axes=(None, 0))(state, jnp.int32([16, 24, 32, 40, 48]))

    # Combine all three conditions for the final decision
    final_can_castle_q = can_castle_queen_side & not_checked[:3].all()
    final_can_castle_k = can_castle_king_side & not_checked[2:].all()

    # Create a boolean mask for each castling move using the final, correct condition.
    arange = jnp.arange(64 * 73 + 1)
    castle_q_mask = (arange == 2364) & final_can_castle_q
    castle_k_mask = (arange == 2367) & final_can_castle_k

    # 4. Combine all masks using a final, element-wise logical OR.
    # This single operation replaces all the previous scatter updates.
    mask = base_update_mask | underpromo_mask | castle_q_mask | castle_k_mask

    return mask[:-1]

# ==============================================================================
# PRE-COMPUTED TABLES FOR OPTIMIZED `_is_attacked`
# ==============================================================================
# For a given square, lists all squares from which an opponent's piece can attack.
# These tables are indexed by the *attacked* square.
KNIGHT_ATTACKS = -np.ones((64, 8), np.int32)
KING_ATTACKS = -np.ones((64, 8), np.int32)
PAWN_ATTACKS = -np.ones((64, 2), np.int32)

# For a given square, lists all squares in each of the 8 sliding directions.
RAYS = -np.ones((64, 8, 7), np.int32)
# Directions: E, NE, N, NW, W, SW, S, SE
# N: +1, E: +8, S: -1, W: -8
RAY_DELTAS = np.array([8, 9, 1, -7, -8, -9, -1, 7])

# Maps piece type to the rays it can move along.
# The order matches RAY_DELTAS: E, NE, N, NW, W, SW, S, SE
PIECE_RAY_MAP = np.zeros((7, 8), dtype=bool)
PIECE_RAY_MAP[BISHOP, [1, 3, 5, 7]] = True  # Diagonal directions
PIECE_RAY_MAP[ROOK, [0, 2, 4, 6]] = True    # Cardinal directions
PIECE_RAY_MAP[QUEEN, :] = True

for sq in range(64):
    # This coordinate system must be consistent: rank is the row, file is the column.
    # In pgx chess, from_ % 8 is rank-like and from_ // 8 is file-like.
    rank, file = sq % 8, sq // 8

    # --- Knight attacks ---
    knight_moves = []
    for dr, df in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
        nr, nf = rank + dr, file + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            knight_moves.append(nf * 8 + nr)
    KNIGHT_ATTACKS[sq, :len(knight_moves)] = knight_moves

    # --- King attacks ---
    king_moves = []
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            if dr == 0 and df == 0: continue
            nr, nf = rank + dr, file + df
            if 0 <= nf < 8 and 0 <= nr < 8:
                king_moves.append(nf * 8 + nr)
    KING_ATTACKS[sq, :len(king_moves)] = king_moves

    # --- Pawn attacks ---
    # To attack square (rank, file), an opponent pawn must be on the next rank up (rank + 1)
    # and an adjacent file (file +/- 1).
    pawn_attackers = []
    if rank < 7:  # Can't be attacked from above rank 7
        if file > 0:  # Attacker from the left-file
            pawn_attackers.append((file - 1) * 8 + (rank + 1))
        if file < 7:  # Attacker from the right-file
            pawn_attackers.append((file + 1) * 8 + (rank + 1))
    PAWN_ATTACKS[sq, :len(pawn_attackers)] = pawn_attackers

    # --- Ray casting ---
    for i, delta in enumerate(RAY_DELTAS):
        ray = []
        for k in range(1, 8):
            curr_sq = sq + k * delta
            if not (0 <= curr_sq < 64): break

            # Robust wrap-around check
            curr_rank, curr_file = curr_sq % 8, curr_sq // 8
            dist = max(abs(curr_rank - rank), abs(curr_file - file))
            if dist != k: break # The move wrapped around the board edge
            ray.append(curr_sq)
        RAYS[sq, i, :len(ray)] = ray

# Convert all new tables to JAX arrays
KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS, RAYS, PIECE_RAY_MAP = (
    jnp.array(x) for x in (KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS, RAYS, PIECE_RAY_MAP)
)

def _is_attacked(state: GameState, pos: Array):
    """
    A fully vectorized, gather-free check for whether a square is attacked.
    This version uses ray-casting for sliding pieces and direct lookups for others,
    avoiding expensive vmap operations.
    """
    board = state.board

    # Create a one-hot vector to serve as a selector for the given position `pos`.
    one_hot_pos = jax.nn.one_hot(pos, 64, dtype=jnp.int32)

    # --- 1. Direct attacks (Knight, Pawn, King) ---
    knight_attack_squares = one_hot_pos @ KNIGHT_ATTACKS
    knights = _pieces_at(board, knight_attack_squares)
    is_attacked_by_knight = (knights == -KNIGHT).any()

    pawn_attack_squares = one_hot_pos @ PAWN_ATTACKS
    pawns = _pieces_at(board, pawn_attack_squares)
    is_attacked_by_pawn = (pawns == -PAWN).any()

    king_attack_squares = one_hot_pos @ KING_ATTACKS
    kings = _pieces_at(board, king_attack_squares)
    is_attacked_by_king = (kings == -KING).any()

    by_near = is_attacked_by_knight | is_attacked_by_pawn | is_attacked_by_king

    # --- 2. Sliding attacks (Rook, Bishop, Queen) ---

    # Use tensordot to select the (8, 7) slice from the RAYS table.
    ray_squares = jnp.tensordot(one_hot_pos, RAYS, axes=([0], [0]))
    pieces_on_rays = _pieces_at(board, ray_squares)  # Shape: (8, 7)

    # Find the first piece encountered in each of the 8 directions.
    is_blocker = pieces_on_rays != EMPTY
    # `argmax` gives the distance (0-6) to the first blocker in each ray.
    dist_to_first_blocker = jnp.argmax(is_blocker, axis=1)
    ray_has_blocker = is_blocker.any(axis=1)

    # Use one-hot multiplication to select the piece at the blocker's position.
    one_hot_dist = jax.nn.one_hot(dist_to_first_blocker, 7, dtype=board.dtype)
    first_blocker_piece = (pieces_on_rays * one_hot_dist).sum(axis=1)
    # Ignore the result if the ray was actually empty.
    first_blocker_piece = jnp.where(ray_has_blocker, first_blocker_piece, EMPTY)

    # Check if the blocking piece is an opponent's sliding piece that can attack along that ray.
    is_opponent_blocker = first_blocker_piece < 0

    # Check if the piece type can move along the given ray direction.
    one_hot_piece_type = jax.nn.one_hot(jnp.abs(first_blocker_piece), 7, dtype=jnp.bool_)
    can_piece_attack_on_ray = (one_hot_piece_type * PIECE_RAY_MAP.T).sum(axis=1)

    by_slider = (is_opponent_blocker & can_piece_attack_on_ray).any()

    return by_near | by_slider

def _is_checked(state: GameState):
    king_pos = jnp.argmin(jnp.abs(state.board - KING))
    return _is_attacked(state, king_pos)


def _zobrist_hash(state: GameState) -> Array:
    hash_ = lax.select(state.color == 0, ZOBRIST_SIDE, jnp.zeros_like(ZOBRIST_SIDE))
    one_hot_board = jax.nn.one_hot(state.board + 6, 13, dtype=jnp.uint32)
    to_reduce = jnp.einsum('sp,sph->sh', one_hot_board, ZOBRIST_BOARD, preferred_element_type=jnp.uint32)
    hash_ ^= lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))
    to_reduce = jnp.where(state.castling_rights.reshape(-1, 1), ZOBRIST_CASTLING, 0)
    hash_ ^= lax.reduce(to_reduce, 0, lax.bitwise_xor, (0,))

    # Optimized version:
    # 1. Create a one-hot vector from the en passant index.
    #    The modulo handles the -1 case, mapping it to the last element (index 64).
    safe_idx = state.en_passant % 65
    one_hot_en_passant = jax.nn.one_hot(safe_idx, 65, dtype=jnp.uint32)

    # 2. Use matrix multiplication to select the correct hash value.
    en_passant_hash = one_hot_en_passant @ ZOBRIST_EN_PASSANT
    hash_ ^= en_passant_hash

    return hash_


def _pieces_at(board: Array, idx: Array) -> Array:
    """Return board[idx] without gathers; `idx` may contain –1.

    –1  →  EMPTY (0)                # matches original PGX semantics
    """
    idx   = jnp.asarray(idx, jnp.int32)

    valid = idx >= 0                       # mask of real squares
    safe  = jnp.where(valid, idx, 0)       # –1 → 0 just to build one-hot

    # one-hot dot board  ➜  gather-equivalent
    pieces = jax.nn.one_hot(safe, 64, dtype=board.dtype) @ board

    # force EMPTY on sentinels

    return jnp.where(valid, pieces, jnp.int32(EMPTY))


def _one_hot64(idx: Array, dtype=jnp.int32) -> Array:
    """Length-64 one-hot vector (handles scalar or vector idx)."""
    return jax.nn.one_hot(idx, 64, dtype=dtype)          # (..., 64)


def _set_pieces(board: Array, idx: Array, val: Array) -> Array:
    """Update board[idx] ← val (handles scalar or 1-D idx)."""
    idx = jnp.atleast_1d(idx).astype(jnp.int32)         # (k,)
    val = jnp.atleast_1d(val).astype(board.dtype)       # (k,)
    mask = _one_hot64(idx, dtype=board.dtype)           # (k, 64)
    # board * (1-mask)  +  (mask.T @ val)   (64,)
    return board * (1 - mask.max(axis=0)) + (mask.T @ val)


def _is_path_clear(board: Array, from_: Array, to_: Array) -> Array:
    """
    Checks if the path is clear between from_ and to_ for sliding pieces.
    Returns True for non-sliding moves.
    """
    r0, c0 = from_ % 8, from_ // 8
    r1, c1 = to_   % 8, to_   // 8
    dr, dc = r1 - r0, c1 - c0

    is_sliding_path = (dr == 0) | (dc == 0) | (jnp.abs(dr) == jnp.abs(dc))

    step_dr, step_dc = jnp.sign(dr), jnp.sign(dc)
    delta      = step_dc * 8 + step_dr
    path_len   = jnp.maximum(jnp.abs(dr), jnp.abs(dc))

    step_multipliers     = jnp.arange(1, 7)
    intermediate_squares = from_ + step_multipliers * delta

    # Mask to ensure only valid squares (or -1) are passed to the next function
    valid_steps_mask     = step_multipliers < path_len
    safe_sq              = jnp.where(valid_steps_mask, intermediate_squares, -1)

    # Check pieces on the safe, valid squares
    board_pieces         = _pieces_at(board, safe_sq)
    are_between_empty    = (board_pieces == EMPTY).all()

    return ~is_sliding_path | are_between_empty

def _can_move_on_the_fly(piece, from_, to):
    """
    Calculates move geometry on the fly.
    """
    r0, c0 = from_ % 8, from_ // 8
    r1, c1 = to % 8, to // 8
    dr, dc = r1 - r0, c1 - c0

    pawn_one_step = (dr == 1) & (jnp.abs(dc) <= 1)
    # The special two-step move is only from the second rank (index 1)
    pawn_two_step = (r0 == 1) & (dr == 2) & (dc == 0)
    is_pawn_move = pawn_one_step | pawn_two_step

    is_knight_move = (jnp.abs(dr) * jnp.abs(dc) == 2)
    is_bishop_move = (jnp.abs(dr) == jnp.abs(dc))
    is_rook_move = (dr == 0) | (dc == 0)
    is_queen_move = is_bishop_move | is_rook_move
    is_king_move = (jnp.abs(dr) <= 1) & (jnp.abs(dc) <= 1)

    # --- Select the correct check based on the piece type ---
    can_move = jnp.zeros_like(dr, dtype=jnp.bool_) # Default False
    can_move = jnp.where(piece == PAWN, is_pawn_move, can_move)
    can_move = jnp.where(piece == KNIGHT, is_knight_move, can_move)
    can_move = jnp.where(piece == BISHOP, is_bishop_move, can_move)
    can_move = jnp.where(piece == ROOK, is_rook_move, can_move)
    can_move = jnp.where(piece == QUEEN, is_queen_move, can_move)
    can_move = jnp.where(piece == KING, is_king_move, can_move)

    return can_move
