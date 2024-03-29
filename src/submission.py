from typing import Optional

import random
import math
import time

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


class MCTSNode:
    def __init__(self, board, config, turn, move=None, parent=None):
        self.board = board
        self.config = config
        self.turn = turn

        self.move = move
        self.parent = parent

        self.children = []
        self.wins = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(legal_moves(self.board, self.config))

    def expand(self):
        for move in legal_moves(self.board, self.config):
            if move not in [child.move for child in self.children]:
                next_board = self.board.copy()
                play(next_board, move, self.turn, self.config)
                self.children.append(MCTSNode(self.board, self.config, 1 if self.turn == 2 else 2, move, self))
                break

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            if child.visits > 0 else float('-inf')
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


    def rollout(self):
        current_rollout_board = self.board.copy()
        turn = self.turn
        while game_result(current_rollout_board, self.config) is None:
            possible_moves = legal_moves(current_rollout_board, self.config)
            move = random.choice(possible_moves)
            play(current_rollout_board, move, turn, self.config)
            turn = 1 if turn == 2 else 2
        result = game_result(current_rollout_board, self.config)
        return result[self.turn]

    def backpropagate(self, result):
        self.wins += result
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)

def MCTS(root):
    t0 = time.time()
    while time.time() - t0 < 1.9:
        node = root
        while game_result(node.board, node.config) is None and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_fully_expanded() and game_result(node.board, node.config) is None:
            node.expand()
        leaf = random.choice(node.children) if node.children else node
        simulation_result = leaf.rollout()
        leaf.backpropagate(simulation_result)
    return max(root.children, key=lambda x: x.visits).move

def mcts_agent(obs, config):
    root = MCTSNode(board=obs["board"], config=config, turn=obs["mark"])
    best_move = MCTS(root)
    return best_move

