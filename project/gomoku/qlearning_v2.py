"""
TODO: Implement Approximate Q-Learning player for Gomoku.
* Extract features from the state-action pair and store in a dictionary with the format {feature_name: feature_value}.
* Similarly, the weight will be the dictionary in the format {feature_name: weight_value}.
* self.action_history stores the state-action pair for each move in the game.
* We don't use hashing for the board state as we need to extract features.
"""
from typing import List, Tuple, Union, DefaultDict
from tqdm import tqdm
from ..player import Player
from collections import defaultdict

import numpy as np
import random
import os
import pickle
from .bots.beginner import GMK_Beginner
from .bots.intermediate import GMK_Intermediate
from .bots.advanced import GMK_Advanced
from .bots.master import GMK_Master

NUM_EPISODES = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.65
EXPLORATION_RATE = 0.2
SEED = 2024
random.seed(SEED)

class GMK_ApproximateQPlayer(Player):
    def __init__(self, letter, size=15, transfer_player=None):
        super().__init__(letter)
        self.opponent = transfer_player
        self.num_episodes = NUM_EPISODES
        self.learning_rate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EXPLORATION_RATE
        # self.weights = defaultdict(float) # Initialize weights to 0
        self.weights = {
            'immediate-winning-move': 5000,
            'immediate-blocking-move': -4500,
            '#-of-unblocked-four-player': 4000,
            '#-of-unblocked-four-opponent': -3500,
            'blocked-four-player': 1500,
            'blocked-four-opponent': -1200,
            '#-of-unblocked-three-player': 1000,
            '#-of-unblocked-three-opponent': -750,
            '#-of-unblocked-two-player': 100,
            '#-of-unblocked-two-opponent': -60,
            'center-control-player': 1.5,
            'center-control-opponent': -1.5,
            'double-threat-player': 250,
            'double-threat-opponent': -500,
            'bias': 1.0,
        }
        self.action_history = []
        self.board_size = size
        self.feature_extractor = SimpleExtractor()

    def train(self, game, save_filename=None):
        # Main Q-learning algorithm
        opponent_letter = 'X' if self.letter == 'O' else 'O'
        if self.opponent is None:
            opponent = GMK_Master(opponent_letter)
        else:
            opponent = self.opponent(opponent_letter)
            
        print(f"Training {self.letter} player for {self.num_episodes} episodes...")
        game_state = game.copy()
        
        for _ in tqdm(range(self.num_episodes)):               
            game_state.restart()
            opponent.action_history = []
            
            current_player = self if self.letter == 'X' else opponent 
            next_player = self if self.letter == 'O' else opponent
            while True:                
                if isinstance(current_player, GMK_ApproximateQPlayer):     
                    action = current_player.choose_action(game_state)
                    state = copy.deepcopy(game_state.board_state)
                    current_player.action_history.append((state, action)) 
                else:
                    action = current_player.get_move(game_state)    
                
                next_game_state = game_state.copy()
                next_game_state.set_move(action[0], action[1], current_player.letter)
                
                if next_game_state.game_over():
                    reward = 1 if next_game_state.wins(current_player.letter) else -1 if next_game_state.wins(next_player.letter) else 0
                    if isinstance(current_player, GMK_ApproximateQPlayer):
                        current_player.update_rewards(reward)
                    if isinstance(next_player, GMK_ApproximateQPlayer):
                        next_player.update_rewards(-reward)
                    break
                else: 
                    current_player, next_player = next_player, current_player
                    game_state = next_game_state    

            self.letter = 'X' if self.letter == 'O' else 'O'
            opponent.letter = 'X' if opponent.letter == 'O' else 'O'  
            self.action_history = []
        
        print("Training complete. Saving training weights...")
        if save_filename is None:
            save_filename = f'{self.board_size}x{self.board_size}_{NUM_EPISODES}.pkl'
        self.save_weight(save_filename)
    
    def update_rewards(self, reward: float):
        """
        Given the reward at the end of the game, update the weights for each state-action pair in the game with the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * Q(s', a') - Q(s, a)) * f_i(s, a)

        * We need to update the Q-values for each state-action pair in the action history because the reward is only received at the end.
        * Make a call to update_q_values() for each state-action pair in the action history.
        """
        last_state, last_action = self.action_history[-1]
        self.update_q_values(last_state, last_action, last_state, reward)
        for t in range(len(self.action_history) -1, 0, -1):
            reward *= self.gamma 
            next_state, _ = self.action_history[t]
            prev_state, prev_action = self.action_history[t - 1]
            self.update_q_values(prev_state, prev_action, next_state, reward)


    def choose_action(self, game) -> Union[List[int], Tuple[int, int]]:
        """
        Choose action with ε-greedy strategy.
        If random number < ε, choose random action.
        Else choose action with the highest Q-value.
        :return: action
        """
        action = None
        ######### YOUR CODE HERE #########
        legal_actions = self.empty_cells(game.board_state)
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        best_q = float('-inf')
        best_action = None
        for action in legal_actions:
            q = self.q_value(game.board_state, action)
            if q > best_q:
                best_q = q
                best_action = action

        return best_action                                
        ######### YOUR CODE HERE #########
        # return action

    def update_q_values(self, state, action, next_state, reward):
        """
        Given (s, a, s', r), update the weights for the state-action pair (s, a) using the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * Q(s', a') - Q(s, a)) * f_i(s, a)
        :return: None
        """
        ######### YOUR CODE HERE #########
        current_features = self.feature_vector(state, action)
        current_q = self.q_value(state, action)

        max_future_q = 0
        if next_state is not None:
            legal_actions = self.empty_cells(next_state)
            if legal_actions:
                max_future_q = max(self.q_value(next_state, a) for a in legal_actions)

        target = reward + self.gamma * max_future_q
        td_error = target - current_q

        for feature, value in current_features.items():
            self.weights[feature] += self.learning_rate * td_error * value

        ######### YOUR CODE HERE #########

    def feature_vector(self, state, action) -> np.ndarray:
        """
        Extract the feature vector for a given state-action pair.
        :return: feature vector
        """
        return self.feature_extractor.get_features(copy.deepcopy(state), action, self.letter) # Return the feature vector

    def q_value(self, state, action) -> float:
        """
        Compute the Q-value for a given state-action pair as the dot product of the feature vector and the weight vector.
        :return: Q-value
        """
        q_value = 0
        features = self.feature_vector(state, action)
        for feature_name in features.keys():
            q_value += self.weights[feature_name] * features[feature_name]
        return q_value
    
    def save_weight(self, filename):
        """
        Save the weights of the feature vector.
        """
        path = 'project/gomoku/q_weights'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/{filename}', 'wb') as f:
            pickle.dump(dict(self.weights), f)

    def load_weight(self, filename):
        """
        Load the Q-table.
        """
        path = 'project/gomoku/q_weights'
        if not os.path.exists(f'{path}/{filename}'):
            raise FileNotFoundError(f"Weight file '{filename}' not found.")
        with open(f'{path}/{filename}', 'rb') as f:
            dict_weights = pickle.load(f)
            self.weights.update(dict_weights)

    def get_move(self, game):
        self.epsilon = 0  # No exploration
        return self.choose_action(game)
    
    def empty_cells(self, board: List[List[str]]) -> List[Tuple[int, int]]:
        """
        Return a list of empty cells in the board.
        """
        return [(x, y) for x in range(len(board)) for y in range(len(board[0])) if board[x][y] is None]

    def __str__(self):
        return "Approximate Q-Learning Player"

########################### Feature Extractor ###########################
from abc import ABC, abstractmethod
import copy

class FeatureExtractor(ABC):
    @abstractmethod
    def get_features(self, state: List[List[str]], move: Union[List[int], Tuple[int]], player: str) -> DefaultDict[str, float]:
        """
        :param state: current board state
        :param move: move taken by the player
        :param player: current player
        :return: a dictionary {feature_name: feature_value}
        """
        pass

class IdentityExtractor(FeatureExtractor):
    def get_features(self, state, move, player):
        """
        Return 1.0 for all state action pair.
        """
        feats = defaultdict(float)
        key = self.hash_board(state)
        feats[(key, tuple(move))] = 1.0
        return feats
    
    def hash_board(self, board):
        key = ''
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    key += '1'
                elif board[i][j] == 'O':
                    key += '2'
                else:
                    key += '0'
        return key

class SimpleExtractor(FeatureExtractor):
    def get_features(self, state, move, player):
        """
        features: #-of-unblocked-three-player, #-of-unblocked-three-opponent
        """
        opponent = 'X' if player == 'O' else 'O'

        x, y = move
        state = np.array(state)
        state[x][y] = player

        feats = defaultdict(float)
        feats['immediate-winning-move'] = self.immediate_winning_move(player, state)
        feats['immediate-blocking-move'] = self.immediate_blocking_move(opponent, state)

        feats['#-of-unblocked-four-player'] = self.count_open_four(player, state)
        feats['#-of-unblocked-four-opponent'] = self.count_open_four(opponent, state)

        feats['blocked-four-player'] = self.count_blocked_four(player, state)
        feats['blocked-four-opponent'] = self.count_blocked_four(opponent, state)

        feats['#-of-unblocked-three-player'] = self.count_open_three(player, state)
        feats['#-of-unblocked-three-opponent'] = self.count_open_three(opponent, state)

        feats['#-of-unblocked-two-player'] = self.count_open_two(player, state)
        feats['#-of-unblocked-two-opponent'] = self.count_open_two(opponent, state)

        feats['center-control-player'] = self.center_control(player, state)
        feats['center-control-opponent'] = self.center_control(opponent, state)

        feats['double-threat-player'] = self.count_double_threats(player, state)
        feats['double-threat-opponent'] = self.count_double_threats(opponent, state)

        feats['bias'] = 1.0
        
        return feats

    def immediate_winning_move(self, player, board):
        return 1.0 if self.wins(player, board) else 0.0

    def immediate_blocking_move(self, opponent, board):
        return 1.0 if self.wins(opponent, board) else 0.0

    def count_open_four(self, player, board):
        length = 5
        def check_open_four(player, array):
            lst = list(array)
            return lst.count(player) == 4 and lst.count(None) > 0
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_open_four(player, array):
                    threat_cnt += 1

        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_open_four(player, array):
                    threat_cnt += 1

        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

        return threat_cnt
    
    def count_blocked_four(self, player, board): # four stones and one opponent stone or edge
        length = 5
        def check_blocked_four(player, array):
            lst = list(array)
            return lst.count(player) == 4 and lst.count(None) == 1 and (
                (lst[0] is not None and lst[-1] is None) or (lst[-1] is not None and lst[0] is None)
            )
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_blocked_four(player, array):
                    threat_cnt += 1
        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_blocked_four(player, array):
                    threat_cnt += 1
        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_blocked_four(player, array):
                    threat_cnt += 1
                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_blocked_four(player, array):
                    threat_cnt += 1
        return threat_cnt

    def count_open_three(self, player, board): # three stones are consecutive in the middle
        length = 5
        def check_open_three(player, array):
            # lst = list(array)
            # return lst.count(player) == 3 and lst.count(None) < 2
            lst = list(array)
            if lst.count(player) != 3:
                return False
            # Check ends are empty (open)
            if lst[0] is None and lst[-1] is None:
                return True
            return False
        
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size-(length-1)):
                array = board[row,col:col+length]
                is_threat = check_open_three(player, array)
                if is_threat: 
                    threat_cnt += 1              
                    
        ## Read vertically
        for col in range(size):
            for row in range(size-(length-1)):
                array = board[row:row+length,col]
                is_threat = check_open_three(player, array)
                if is_threat: 
                    threat_cnt += 1

        ## Read diagonally
        for row in range(size-(length-1)):
            for col in range(size-(length-1)):
                array = []
                for i in range(length):
                    array.append(board[i+row,i+col])
                is_threat = check_open_three(player, array)
                if is_threat: 
                    threat_cnt += 1              

                array = []
                for i in range(length):
                    array.append(board[i+row,col+length-1-i])
                is_threat = check_open_three(player, array)
                if is_threat: 
                    threat_cnt += 1 

        return threat_cnt

    def count_open_two(self, player, board):
        length = 4
        def check_open_four(player, array):
            lst = list(array)
            return lst[1:3].count(player) == 2 and lst.count(None) == 2
        threat_cnt = 0
        size = len(board)
        for row in range(size):
            for col in range(size - (length - 1)):
                array = board[row, col:col + length]
                if check_open_four(player, array):
                    threat_cnt += 1

        for col in range(size):
            for row in range(size - (length - 1)):
                array = board[row:row + length, col]
                if check_open_four(player, array):
                    threat_cnt += 1

        for row in range(size - (length - 1)):
            for col in range(size - (length - 1)):
                array = [board[row + i, col + i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

                array = [board[row + i, col + length - 1 - i] for i in range(length)]
                if check_open_four(player, array):
                    threat_cnt += 1

        return threat_cnt

    def center_control(self, player, board):
        size = len(board)
        center_start = size // 3
        center_end = 2 * size // 3
        control_cnt = 0
        for row in range(center_start, center_end):
            for col in range(center_start, center_end):
                if board[row, col] == player:
                    control_cnt += 1
        return control_cnt

    def count_double_threats(self, player, board):
        length = 5
        size = len(board)
        double_threats = 0

        for row in range(size):
            for col in range(size):
                if board[row][col] is None:
                    # Temporarily make a move at the empty cell
                    board[row][col] = player
                    threats = 0

                    # Check all lines through this cell
                    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                    for dx, dy in directions:
                        if self.is_threat(player, board, row, col, dx, dy, length):
                            threats += 1
                    if threats >= 2:
                        double_threats += 1
                    # Undo the temporary move
                    board[row][col] = None
        return double_threats

    def is_threat(self, player, board, row, col, dx, dy, length):
        count = 0
        size = len(board)
        # Check in both directions
        for d in range(-length + 1, length):
            x, y = row + d * dx, col + d * dy
            if 0 <= x < size and 0 <= y < size and board[x][y] == player:
                count += 1
            elif 0 <= x < size and 0 <= y < size and board[x][y] is None:
                continue
            else:
                count = 0
            if count == 4:
                return True
        return False

    def wins(self, player, board):
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(size):
            for y in range(size):
                if board[x][y] == player:
                    for dx, dy in directions:
                        if self.check_direction(player, board, x, y, dx, dy):
                            return True
        return False
    
    def check_direction(self, player, board, x, y, dx, dy):
        count = 0
        for _ in range(5):
            if 0 <= x < len(board) and 0 <= y < len(board) and board[x][y] == player:
                count += 1
            x += dx
            y += dy
        return count >= 5

    