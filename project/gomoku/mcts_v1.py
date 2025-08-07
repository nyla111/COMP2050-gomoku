"""
TODO: Implement the standard MCTS player for Gomoku.
* tree policy: UCB1
* rollout policy: random
"""

import numpy as np
import math
import multiprocessing as mp
from functools import partial

from ..player import Player
from ..game import Gomoku

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 2000

import random
SEED = 2024
random.seed(SEED)

class TreeNode():
    def __init__(self, game_state: Gomoku, player_letter: str, parent=None, parent_action=None):
        self.player = player_letter
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0
        self.Q = 0
    
    def select(self) -> 'TreeNode':
        """
        Select the best child node based on UCB1 formula. Keep selecting until a leaf node is reached.
        """
        leaf_node = self
        while not leaf_node.is_terminal_node():
            if leaf_node.is_leaf_node():
                return leaf_node
            leaf_node = leaf_node.best_child()
        return leaf_node
    
    def expand(self) -> 'TreeNode':
        """
        Expand the current node by adding all possible child nodes. Return one of the child nodes for simulation.
        """
        child_node = None
        if self.is_terminal_node():
            return self
            
        possible_moves = self.game_state.empty_cells()
        if not possible_moves:  # If no moves available
            return self
            
        next_player = 'O' if self.player == 'X' else 'X'
        
        # Create all possible child nodes
        for move in possible_moves:
            child_game_state = self.game_state.copy()
            child_game_state.set_move(move[0], move[1], self.player)
            new_child = TreeNode(child_game_state, next_player, parent=self, parent_action=move)
            self.children.append(new_child)
        
        # Return a randomly selected child node
        child_node = random.choice(self.children) if self.children else self
        return child_node
    
    def simulate(self) -> int:
        """
        Run simulation from the current node until the game is over. Return the result of the simulation.
        """
        result = 0
        # If we're already at a terminal state, evaluate it
        if self.is_terminal_node():
            if self.game_state.wins(self.player):
                return WIN
            elif self.game_state.wins('X' if self.player == 'O' else 'O'):
                return LOSE
            return DRAW

        simulate_game = self.game_state.copy()
        current_player = self.player
        opponent = 'O' if current_player == 'X' else 'X'
        
        while True:
            empty = simulate_game.empty_cells()
            if not empty:
                return DRAW

            move = random.choice(empty)
            simulate_game.set_move(move[0], move[1], current_player)

            if simulate_game.wins(current_player):
                result = WIN if current_player == self.player else LOSE
                break
                
            current_player, opponent = opponent, current_player
        return result
    
    def backpropagate(self, result: int):
        """
        Backpropagate the result of the simulation to the root node.
        """
        self.N += 1
        self.Q += result
        if self.parent:
            self.parent.backpropagate(-result)
        pass
            
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal_node(self) -> bool:
        return self.game_state.game_over()
    
    def best_child(self) -> 'TreeNode':
        return max(self.children, key=lambda c: c.ucb())
    
    def ucb(self, c=math.sqrt(2)) -> float:
        return self.Q / (1+self.N) + c * np.sqrt(np.log(self.parent.N) / (1+self.N))


SIMULATIONS_PER_WORKER = 10  # Rollouts per worker process
MAX_SIMULATION_DEPTH = 30  # Early cutoff depth


class GMK_NaiveMCTS(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations

    def get_move(self, game: Gomoku):
        root = TreeNode(game, self.letter)

        num_workers = mp.cpu_count()
        num_batches = math.ceil(self.num_simulations / SIMULATIONS_PER_WORKER)

        # Pre-expand leaves to parallelize simulations
        simulation_leaves = []
        for _ in range(num_batches):
            leaf = root.select()
            if not leaf.is_terminal_node():
                leaf = leaf.expand()
            simulation_leaves.append(leaf)

        # Prepare the simulation pool
        with mp.Pool(processes=num_workers) as pool:
            # Each process gets a leaf and performs multiple simulations
            simulate_fn = partial(_simulate_batch_wrapper, num_simulations=SIMULATIONS_PER_WORKER)
            batch_results = pool.map(simulate_fn, simulation_leaves)

        # Backpropagate results
        for leaf, results in zip(simulation_leaves, batch_results):
            for result in results:
                leaf.backpropagate(result)

        # Select the best move
        best_child = max(root.children, key=lambda c: c.N)
        return best_child.parent_action

    def __str__(self) -> str:
        return "Naive MCTS Player"


# Helper for batch simulation
def _simulate_batch_wrapper(node: TreeNode, num_simulations: int) -> list:
    return [node.simulate() for _ in range(num_simulations)]
