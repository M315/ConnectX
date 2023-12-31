# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tqdm
from os import path
from random import choice
from kaggle_environments import make

from utils import play, is_win
from minimax import minimax_agent
from mcts import mcts_agent

EMPTY = 0

def random_agent(obs, config):
    return choice([c for c in range(config.columns) if obs.board[c] == EMPTY])

agents = {"random": random_agent, "minimax": minimax_agent}

def interpreter(state, env):
    columns = env.configuration.columns
    rows = env.configuration.rows

    # Ensure the board is properly initialized.
    board = state[0].observation.board
    if len(board) != (rows * columns):
        board = [EMPTY] * (rows * columns)
        state[0].observation.board = board

    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # Active agent action.
    column = active.action

    # Invalid column, agent loses.
    if column < 0 or active.action >= columns or board[column] != EMPTY:
        active.status = f"Invalid column: {column}"
        inactive.status = "DONE"
        return state

    # Mark the position.
    play(board, column, active.observation.mark, env.configuration)

    # Check for a win.
    if is_win(board, column, active.observation.mark, env.configuration):
        active.reward = 1
        active.status = "DONE"
        inactive.reward = -1
        inactive.status = "DONE"
        return state

    # Check for a tie.
    if all(mark != EMPTY for mark in board):
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    # Swap active agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    columns = env.configuration.columns
    rows = env.configuration.rows
    board = state[0].observation.board

    def print_row(values, delim="|"):
        return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

    row_bar = "+" + "+".join(["---"] * columns) + "+\n"
    out = row_bar
    for r in range(rows):
        out = out + \
            print_row(board[r * columns: r * columns + columns]) + row_bar

    return out


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "connectx.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(dirpath, "connectx.js"))
    with open(jspath) as f:
        return f.read()

def main():
    env = make("connectx", debug=True)

    # Training agent in first position (player 1) against the default random agent.
    trainer = env.train([None, "negamax"])
    config = env.configuration

    res = [0, 0, 0]
    for _ in tqdm.tqdm(range(100)):
        obs = trainer.reset()
        while True:
            env.render()
            action = mcts_agent(obs, config)
            obs, reward, done, info = trainer.step(action)
            if done:
                res[reward + 1] += 1
                break

    print(res)

if __name__ == "__main__":
    main()
