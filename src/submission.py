from random import choice

def agent(observation, configuration):
    board = observation.board
    agent = observation.mark
    columns = configuration.columns
    rows = configuration.rows
    win = configuration.inarow
    
    return choice([c for c in range(columns) if board[c] == 0])
