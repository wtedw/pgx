import base64
import os

from pgx.chess import State as ChessState
from pgx.chess import _flip


def _make_chess_dwg(dwg, state: ChessState, config):
    def _set_piece(_x, _y, _type, _dwg, _dwg_g, grid_size):
        PATH = {
            "P": "images/chess/bPawn.svg",
            "N": "images/chess/bKnight.svg",
            "B": "images/chess/bBishop.svg",
            "R": "images/chess/bRook.svg",
            "Q": "images/chess/bQueen.svg",
            "K": "images/chess/bKing.svg",
            "wP": "images/chess/wPawn.svg",
            "wN": "images/chess/wKnight.svg",
            "wB": "images/chess/wBishop.svg",
            "wR": "images/chess/wRook.svg",
            "wQ": "images/chess/wQueen.svg",
            "wK": "images/chess/wKing.svg",
        }
        file_path = PATH[_type]
        with open(
            os.path.join(os.path.dirname(__file__), file_path),
            "rb",
        ) as f:
            b64_img = base64.b64encode(f.read())
        img = _dwg.image(
            "data:image/svg+xml;base64," + b64_img.decode("ascii"),
            insert=((_x + 5), (_y + 5)),
            size=(grid_size * 0.8, grid_size * 0.8),
        )
        _dwg_g.add(img)

        return _dwg_g

    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = 8
    BOARD_HEIGHT = 8

    NUM_TO_CHAR = ["a", "b", "c", "d", "e", "f", "g", "h"]
    PIECES = [
        "",
        "wP",
        "wN",
        "wB",
        "wR",
        "wQ",
        "wK",
        "P",
        "N",
        "B",
        "R",
        "Q",
        "K",
    ]  # k"N"ight
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (
                (BOARD_WIDTH + 1.5) * GRID_SIZE,
                (BOARD_HEIGHT + 1) * GRID_SIZE,
            ),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()
    for i in range(BOARD_WIDTH * BOARD_HEIGHT):
        if (i // BOARD_HEIGHT) % 2 != i % 2:
            fill_color = color_set.p1_outline
        else:
            fill_color = color_set.p2_outline

        x = i % BOARD_WIDTH
        y = i // BOARD_HEIGHT
        board_g.add(
            dwg.rect(
                (x * GRID_SIZE, y * GRID_SIZE),
                (
                    GRID_SIZE,
                    GRID_SIZE,
                ),
                fill=fill_color,
            )
        )

    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_WIDTH * GRID_SIZE,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="3px",
        )
    )

    # dan,suji
    cord = board_g.add(dwg.g(id="cord", fill=color_set.grid_color))
    for i in range(BOARD_WIDTH):
        cord.add(
            dwg.text(
                text=f"{8-i}",
                insert=(
                    (-0.3) * GRID_SIZE,
                    (i + 0.6) * GRID_SIZE,
                ),
                font_size="20px",
                font_family="Serif",
            )
        )
        cord.add(
            dwg.text(
                text=f"{NUM_TO_CHAR[i]}",
                insert=(
                    (i + 0.4) * GRID_SIZE,
                    (8.35) * GRID_SIZE,
                ),
                font_size="20px",
                font_family="Serif",
            )
        )

    if state._x.color == 1:
        state = state.replace(_x=_flip(state._x))  # type: ignore
    # pieces
    pieces_g = dwg.g()
    for i in range(64):
        pi = int(state._x.board[i].item())
        if pi == 0:
            continue
        if pi < 0:
            pi *= -1
            pi += 6
        piece_type = PIECES[pi]
        xy = i
        x = xy // BOARD_HEIGHT  # ChessStateは左下原点
        y = 7 - xy % BOARD_HEIGHT
        pieces_g = _set_piece(
            x * GRID_SIZE,
            y * GRID_SIZE,
            piece_type,
            dwg,
            pieces_g,
            GRID_SIZE,
        )

    board_g.add(pieces_g)
    board_g.translate(10, 0)

    return board_g
