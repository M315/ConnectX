from typing import Optional

EMPTY = 0

def legal_moves(board, config):
    return [col for (col, cell) in enumerate(board[:config.columns]) if cell == EMPTY]

def play(board, column, mark, config):
    columns = config.columns
    rows = config.rows
    row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    board[column + (row * columns)] = mark


def is_win(board, column, mark, config, has_played=True):
    columns = config.columns
    rows = config.rows
    inarow = config.inarow - 1
    row = (
        min([r for r in range(rows) if board[column + (r * columns)] == mark], default=None)
        if has_played
        else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    )
    if row is None:
        return False

    def count(offset_row, offset_column):
        for i in range(1, inarow + 1):
            r = row + offset_row * i
            c = column + offset_column * i
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= columns
                or board[c + (r * columns)] != mark
            ):
                return i - 1
        return inarow

    return (
        count(1, 0) >= inarow  # vertical.
        or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
        or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
        or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
    )

def has_won(board, mark, config):
    won = False
    for col in range(config.columns):
        won |= is_win(board, col, mark, config, has_played=True)
    return won

def game_result(board, config) -> Optional[dict]:
    if all([cell != EMPTY for cell in board]):
        return {1: 0.5, 2: 0.5}
    if has_won(board, 1, config):
        return {1: 1., 2: 0.}
    if has_won(board, 2, config):
        return {1: 0., 2: 1.}
    return None
