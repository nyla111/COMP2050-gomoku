"""
TODO: Implement Minimax with Alpha-Beta Pruning player for Gomoku.
* You need to implement heuristic evaluation function for non-terminal states.
* Optional: You can implement the function promising_next_moves to explore reduce the branching factor.
"""
from ..player import Player
from ..game import Gomoku
from typing import List, Tuple, Union
import math
import random
SEED = 2024
random.seed(SEED)

DEPTH = 2 # Fix the depth of the search tree.

class GMK_AlphaBetaPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.depth = DEPTH 

    
    def get_move(self, game: Gomoku):
        if game.last_move == (-1, -1):
            mid_size = game.size // 2
            moves = [(mid_size, mid_size), (mid_size - 1, mid_size - 1), (mid_size + 1, mid_size + 1), (mid_size - 1, mid_size + 1), (mid_size + 1, mid_size - 1)]
            move = random.choice(moves)
            while not game.valid_move(move[0], move[1]):
                move = random.choice(moves)
            return move 
        else:
            # Alpha-Beta Pruning: Initialize alpha to negative infinity and beta to positive infinity
            alpha = -math.inf
            beta = math.inf
            choice = self.minimax(game, self.depth, self.letter, alpha, beta)
            move = [choice[0], choice[1]]
        return move

    def minimax(self, game, depth, player_letter, alpha, beta) -> Union[List[int], Tuple[int]]:
        """
        AI function that chooses the best move with alpha-beta pruning.
        :param game: current state of the board
        :param depth: node index in the tree (0 <= depth <= 9)
        :param player_letter: value representing the player
        :param alpha: best value that the maximizer can guarantee
        :param beta: best value that the minimizer can guarantee
        :return: a list or a tuple with [best row, best col, best score]
        """
        move = [-1, -1]
        if depth == 0 or game.game_over():
            return [game.last_move[0], game.last_move[1], self.evaluate(game)]

        if player_letter == self.letter:
            best = [-1, -1, -math.inf]
        else:
            best = [-1, -1, math.inf]

        possible_moves = self.promising_next_moves(game, player_letter)
        for move in possible_moves:
            x, y = move
            if not game.valid_move(x, y):
                continue
            game.set_move(x, y, player_letter)
            next_letter = 'O' if player_letter == 'X' else 'X'
            result = self.minimax(game, depth - 1, next_letter, alpha, beta)
            game.reset_move(x, y)
            
            result[0], result[1] = x, y
            if player_letter == self.letter:
                if result[2] > best[2]:
                    best = result
                alpha = max(alpha, best[2])
            else:
                if result[2] < best[2]:
                    best = result
                beta = min(beta, best[2])
            
            if beta <= alpha:
                break

        return best

    
    def evaluate(self, game, state=None) -> float:
        """
        Define a heuristic evaluation function for the given state when leaf node is reached.
        :return: a float value representing the score of the state
        """

        board = game.board_state if state is None else state
        size = game.size
        player = self.letter
        opponent = 'O' if player == 'X' else 'X'

        def extract_lines(board):
            lines = []

            # Rows and Columns
            for i in range(size):
                lines.append(board[i])  # row
                lines.append([board[j][i] for j in range(size)])  # column

            # Diagonals
            for p in range(2 * size - 1):
                diag1, diag2 = [], []
                for x in range(max(0, p - size + 1), min(size, p + 1)):
                    y = p - x
                    diag1.append(board[x][y])
                    diag2.append(board[x][size - 1 - y])
                if len(diag1) >= 5:
                    lines.append(diag1)
                if len(diag2) >= 5:
                    lines.append(diag2)

            return lines

        def score_pattern(line, symbol):
            line_str = ''.join([c if c is not None else '.' for c in line])
            score = 0
            patterns = {
                f"{symbol*5}":      1_000_000,
                f".{symbol*4}.":      100_000,
                f"{symbol*4}.":        10_000, 
                f".{symbol*4}":        10_000, 
                f".{symbol*3}.":        5_000, 
                f"{symbol*3}..":        1_000, 
                f"..{symbol*3}":        1_000, 
                f"{symbol}.{symbol}{symbol}": 1_000, 
                f".{symbol*2}.":          300,  
                f"{symbol*2}.":           100,
                f".{symbol*2}":           100, 
            }

            for pat, val in patterns.items():
                score += line_str.count(pat) * val
            return score

        total_score = 0
        lines = extract_lines(board)

        for line in lines:
            total_score += score_pattern(line, player)
            total_score -= .5 * score_pattern(line, opponent)  # strong opponent threat penalty

        return total_score
    

    def promising_next_moves(self, game, player_letter) -> List[Tuple[int]]:
        """
        Find the promosing next moves to explore, so that the search space can be reduced.
        :return: a list of tuples with the best moves
        """
        moves = []
        radius = 2
        visited = set()
        for x in range(game.size):
            for y in range(game.size):
                if game.board_state[x][y] is not None:
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < game.size and 0 <= ny < game.size and game.board_state[nx][ny] is None:
                                visited.add((nx, ny))

        moves = list(visited)
        if not moves:
            for i in range(game.size):
                for j in range(game.size):
                    if game.board_state[i][j] is None:
                        moves.append((i, j))
                        break
                if moves:
                    break

        random.shuffle(moves)
        return moves
    
    def __str__(self):
        return "AlphaBeta Player"
