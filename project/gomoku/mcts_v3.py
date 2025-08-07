"""
TODO: Implement AlphaGo version of MCTS for Gomoku.
* paper: https://www.davidsilver.uk/wp-content/uploads/2020/03/unformatted_final_mastering_go.pdf
* Some github repos for reference:
    *https://github.com/junxiaosong/AlphaZero_Gomoku
    *https://github.com/PolyKen/15_by_15_AlphaGomoku
"""

from ..player import Player
from ..game import Gomoku

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 5000

import random
SEED = 2024
random.seed(SEED)
    
class GMK_AlphaGoMCTS(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations
    
    def index2coordinate(index, board_size):
        return divmod(index, board_size)


    def coordinate2index(coord, board_size):
        return coord[0] * board_size + coord[1]


    def board2legalvec(board):
        return (board.reshape(-1) == EMPTY).astype(np.float32)


    def random_predict(board, color, last_move):
        """Fake neural net: uniform probabilities and random value."""
        size = board.shape[0]
        legal = board2legalvec(board)
        probs = legal / np.sum(legal)
        value = random.uniform(-1, 1)
        return [probs], value


    def check_rules(board, action_cor, color):
        def count_dir(dx, dy):
            count = 1
            for d in [1, -1]:
                for i in range(1, 5):
                    x = action_cor[0] + dx * i * d
                    y = action_cor[1] + dy * i * d
                    if 0 <= x < board.shape[0] and 0 <= y < board.shape[1]:
                        if board[x][y] == color:
                            count += 1
                        else:
                            break
            return count

        if sum(sum(np.abs(board))) < 9:
            return 'continue'

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            if count_dir(dx, dy) >= 5:
                return 'blackwins' if color == BLACK else 'whitewins'

        if np.all(board != 0):
            return 'full'
        return 'continue'
    
    def __str__(self) -> str:
        return "AlphaGo MTCS Player"