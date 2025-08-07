"""
TODO: Implement Approximate Q-Learning player for Gomoku.
* Extract features from the state-action pair and store in a dictionary with the format {feature_name: feature_value}.
* Similarly, the weight will be the dictionary in the format {feature_name: weight_value}.
"""
from typing import List, Tuple, Union, DefaultDict
from tqdm import tqdm
from ..player import Player
from collections import defaultdict

import numpy as np
import random
import os
import pickle
import time

NUM_EPISODES = 100
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
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
        self.weights = defaultdict(lambda: 0) # Initialize weights to 0
        self.action_history = []
        self.board_size = size
        self.feature_extractor = SimpleExtractor()

    def train(self, game, save_filename=None):
        # Main Q-learning algorithm
        opponent_letter = 'X' if self.letter == 'O' else 'O'
        if self.opponent is None:
            opponent = GMK_ApproximateQPlayer(opponent_letter)
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
        """
        ######### YOUR CODE HERE #########
        last_state, last_action = self.action_history[-1]
        self.update_q_values(last_state, last_action, last_state, reward)
        for t in range(len(self.action_history) -1, 0, -1):
            reward = self.gamma * reward
            next_state, _ = self.action_history[t]
            current_state, current_action = self.action_history[t - 1]
            self.update_q_values(current_state, current_action, next_state, reward)
        ######### YOUR CODE HERE #########

    def choose_action(self, game) -> Union[List[int], Tuple[int, int]]:
        """
        Choose action with ε-greedy strategy.
        If random number < ε, choose random action.
        Else choose action with the highest Q-value.
        :return: action
        """
        action = None
        ######### YOUR CODE HERE #########
        state = np.array(copy.deepcopy(game.board_state))
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(game.empty_cells())
        else:
            # Choose the action with the highest Q-value
            empty_cells = game.empty_cells()                          
            empty_q_values = [self.q_value(state, move) for move in empty_cells]      
            max_q_value = max(empty_q_values)                                          
            max_q_indices = [i for i in range(len(empty_cells)) if empty_q_values[i] == max_q_value]    
            max_q_index = random.choice(max_q_indices)                                 
            action = tuple(empty_cells[max_q_index])                                           
        ######### YOUR CODE HERE #########
        return action

    def update_q_values(self, state, action, next_state, reward):
        """
        Given (s, a, s', r), update the weights for the state-action pair (s, a) using the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * Q(s', a') - Q(s, a)) * f_i(s, a)
        :return: None
        """
        ######### YOUR CODE HERE #########
        current_q_value = self.q_value(state, action)
        next_q_value = max([self.q_value(next_state, next_action) for next_action in self.empty_cells(next_state)])
        td_target = reward + self.gamma * next_q_value
        td_error = td_target - current_q_value
        features = self.feature_vector(state, action)
        for feature_name in features.keys():
            self.weights[feature_name] += self.learning_rate * td_error * features[feature_name]
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
        feats = defaultdict(lambda: 0.0)
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

        feats = defaultdict(lambda: 0.0)
        feats['#-of-unblocked-three-player'] = self.count_open_three(player, state)
        feats['#-of-unblocked-three-opponent'] = self.count_open_three(opponent, state)
        feats['bias'] = 1.0
        
        return feats
    
    def count_open_three(self, player, board):
        length = 5
        def check_open_three(player, array):
            lst = list(array)
            return lst.count(player) == 3 and lst.count(None) < 2
        
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
    