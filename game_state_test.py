import numpy as np
from game_state import GameState

def test_winner_check():
    state = np.array([[1, -1, 0, 0, 0, 0], 
                      [1, -1, -1, 0, 0, 0],
                      [-1, 1, -1, 0, 0, 0],
                      [-1, 1, 1, 0, 0, 0],
                      [-1, -1, -1, -1, 0, 0],
                      [1, -1, 0, 0, 0, 0],
                      [-1, 1, -1, 0, 0, 0]])
    game_state = GameState()
    game_state.vector = state
    print(game_state.determine_winner())

test_winner_check()