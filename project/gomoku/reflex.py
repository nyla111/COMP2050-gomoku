"""
TODO: Implement Approximate Q-Learning player for Gomoku.
* Extract features from the state-action pair and store in a numpy array.
* Define the size of the feature vector in the feature_size method.
"""


import numpy as np
import math
import random
import time

from ..player import Player
from ..game import Gomoku

WIN = 1
LOSE = -1
DRAW = 0

class GMK_Reflex(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        empty = game.empty_cells()
        opponent = 'O' if self.letter == 'X' else 'X'
        best_score = -float('inf')
        best_move = None

        # 1. Immediate win
        for move in empty:
            sim_game = game.copy()
            sim_game.set_move(*move, self.letter)
            if sim_game.wins(self.letter):
                return move

        # 2. Block opponent win
        for move in empty:
            sim_game = game.copy()
            sim_game.set_move(*move, opponent)
            if sim_game.wins(opponent):
                return move

        # 3. Score and pick best
        for move in empty:
            score = self.evaluate_move(game, move, self.letter)
            block_score = self.evaluate_move(game, move, opponent)
            combined_score = score + 0.75 * block_score
            if combined_score > best_score:
                best_score = combined_score
                best_move = move

        return best_move

    def evaluate_move(self, game, move, player):
        """
        Heuristic: favors center control + matching pieces along directions.
        """
        r, c = move
        board = game.board_state
        size = len(board)
        score = 0
        center = size // 2
        dist = abs(center - r) + abs(center - c)
        score += (size - dist) * 3

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            open_ends = 0

            for step in range(1, 5):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] == player:
                        count += 1
                    elif board[nr][nc] is None:
                        open_ends += 1
                        break
                    else:
                        break

            for step in range(1, 5):
                nr, nc = r - step * dr, c - step * dc
                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] == player:
                        count += 1
                    elif board[nr][nc] is None:
                        open_ends += 1
                        break
                    else:
                        break

            if count >= 4:
                score += 20000
            elif count == 3 and open_ends == 2:
                score += 5000
            elif count == 3:
                score += 1000
            elif count == 2 and open_ends == 2:
                score += 750
            elif count == 2:
                score += 100
            else:
                score += count * 30 + open_ends * 20
        return score

    def __str__(self):
        return "Reflex Player"
