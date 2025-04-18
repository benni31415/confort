import numpy as np
from scipy.signal import convolve2d

class GameState:

    vector = np.array([])

    def __init__(self, previous_state=None, index=None, red=True):
        if previous_state is None:
            self.vector = np.zeros((8, 8))
        else:
            vec = previous_state.vector
            col = vec[index]
            coin = 1 if red == True else -1

            try:
                col = np.concatenate([col[col != 0], [coin], np.zeros(8-col[col != 0].shape[0]-1)])
            # If illegal action is taken, i.e. coin inserted into full column, count game as lost
            except ValueError:
                print("Illegal action taken")
                self.vector = np.full((8, 8), 1-coin)

            vec[index] = col
            self.vector = vec

    def determine_winner(self):
        winning_streek = np.array([[1, 1, 1, 1]])
        vertical_check = self.check_for_sequence(self.vector, winning_streek)
        if vertical_check is not None:
            return vertical_check
        horizontal_check = self.check_for_sequence(self.vector.transpose(), winning_streek)
        if horizontal_check is not None:
            return horizontal_check
        
        winning_diagonal = np.identity(4)
        diagonal_check_1 = self.check_for_sequence(self.vector, winning_diagonal)
        if diagonal_check_1 is not None:
            return diagonal_check_1
        diagonal_check_2 = self.check_for_sequence(self.vector, winning_diagonal[:, ::-1])
        if diagonal_check_2 is not None:
            return diagonal_check_2
        return None
    
    def check_for_sequence(self, array, filter):
        result = convolve2d(array, filter, mode='valid')
        if np.any(result == 4):
            return 1
        if np.any(result == -4):
            return -1
        return None
