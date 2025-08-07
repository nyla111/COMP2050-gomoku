import numpy as np
import math
import random
from multiprocessing import Pool, cpu_count
from ..player import Player
from ..game import Gomoku


# Constants
WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 20
MAX_SIMULATION_DEPTH = 20
SIMULATIONS_PER_BATCH = 4
SEED = 2024
random.seed(SEED)




class TreeNode:
    def __init__(self, game_state: Gomoku, player_letter: str, parent=None, parent_action=None):
        self.player = player_letter
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0
        self.Q = 0
        self.untried_moves = game_state.empty_cells()  # Cache empty cells for efficiency


    def select(self) -> 'TreeNode':
        node = self
        while not node.is_terminal_node():
            if node.is_leaf_node():
                return node
            node = node.best_child()
        return node


    def expand(self, k_expand=10) -> 'TreeNode':
        if self.is_terminal_node() or not self.untried_moves:
            return self

        # Sort and select top moves
        scored_moves = [(move, self.heuristic_score(self.game_state, self.player, move))
                        for move in self.untried_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        next_player = 'O' if self.player == 'X' else 'X'
        for move, _ in scored_moves[:k_expand]:
            child_state = self.game_state.copy()
            child_state.set_move(move[0], move[1], self.player)
            new_child = TreeNode(child_state, next_player, parent=self, parent_action=move)
            self.children.append(new_child)

        self.untried_moves = [m for m in self.untried_moves if m not in scored_moves[:k_expand]]  # Update untried moves
        return random.choice(self.children) if self.children else self


    @staticmethod
    def simulate(game: Gomoku, start_player: str, max_depth=MAX_SIMULATION_DEPTH) -> int:
        sim_game = game.copy()
        current_player = start_player
        opponent = 'O' if current_player == 'X' else 'X'
        steps = 0
        total_cells = sim_game.size * sim_game.size

        while not sim_game.game_over():
            empty_cells = sim_game.empty_cells()
            if not empty_cells or steps >= max_depth:
                return DRAW

            # Heuristic-based sampling
            scored_moves = [(move, TreeNode.heuristic_score(sim_game, current_player, move))
                            for move in empty_cells]
            move = max(scored_moves, key=lambda x: x[1])[0] if scored_moves else random.choice(empty_cells)
            sim_game.set_move(move[0], move[1], current_player)
            if sim_game.wins(current_player):
                return WIN if current_player == start_player else LOSE

            current_player, opponent = opponent, current_player
            steps += 1


        return DRAW


    @staticmethod
    def count_pattern_lines(game, move, player, patterns):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        x, y = move
        score = 0

        for dx, dy in directions:
            line = ""
            positions = []

            # Look back up to 5 steps
            for i in range(-5, 6):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < game.size and 0 <= ny < game.size:
                    cell = game.board_state[nx][ny]
                    line += cell if cell in ['X', 'O', '.'] else '.'
                    positions.append((nx, ny))
                else:
                    line += " "  # padding for edge of board
                    positions.append(None)

            # Check every 6-length sliding window in the 11-char line
            for i in range(len(line) - 5):
                window = line[i:i+6]
                for pattern, value in patterns.items():
                    if pattern in window:
                        score += value
                        break  # prevent double-counting overlapping patterns

        return score


    @staticmethod
    def heuristic_score(game: Gomoku, player: str, move: tuple) -> int:
        opponent = 'O' if player == 'X' else 'X'

        # Simulate player's move
        temp = game.copy()
        temp.set_move(move[0], move[1], player)
        if temp.wins(player):
            return 1_500_000  # Immediate win

        # Simulate opponent's move at same spot (to test if blocking is necessary)
        temp_block = game.copy()
        temp_block.set_move(move[0], move[1], opponent)
        if temp_block.wins(opponent):
            return 1_000_000  # Must block opponent's winning move

        # Define pattern-based scoring
        def get_patterns(symbol):
            return {
                f".{symbol*4}.":              100_000,
                f"{symbol*4}.":               10_000,
                f".{symbol*4}":               10_000,
                f".{symbol*3}.":               5_000,
                f"{symbol*3}..":               1_000,
                f"..{symbol*3}":               1_000,
                f"{symbol}.{symbol}{symbol}":  5_000,
                f".{symbol*2}.":                 300,
                f"{symbol*2}.":                  100,
                f".{symbol*2}":                  100,
            }

        # Offensive score
        player_patterns = get_patterns(player)
        offensive_score = TreeNode.count_pattern_lines(temp, move, player, player_patterns)

        # Defensive score (simulate what happens if opponent had moved here)
        opponent_patterns = get_patterns(opponent)
        defensive_score = TreeNode.count_pattern_lines(temp_block, move, opponent, opponent_patterns)

        return offensive_score * 1.5 + defensive_score * 0.7  # weight defense slightly less


    def backpropagate(self, result: int):
        node = self
        while node:
            node.N += 1
            node.Q += result
            result = -result
            node = node.parent


    def is_leaf_node(self) -> bool:
        return not self.children


    def is_terminal_node(self) -> bool:
        return self.game_state.game_over()


    def best_child(self, c=math.sqrt(2)) -> 'TreeNode':
        return max(self.children, key=lambda child: child.ucb(c))


    def ucb(self, c=math.sqrt(2)) -> float:
        if self.N == 0:
            return float('inf')
        exploitation = self.Q / self.N
        exploration = c * math.sqrt(math.log(self.parent.N + 1) / self.N) if self.parent else 0
        return exploitation + exploration


# Worker for parallel simulation
def _batch_simulate_wrapper(args):
    game_state, player = args
    return [TreeNode.simulate(game_state, player, MAX_SIMULATION_DEPTH) for _ in range(SIMULATIONS_PER_BATCH)]


# Main MCTS Player
class GMK_BetterMCTS(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations


    def get_move(self, game: Gomoku):
        root = TreeNode(game.copy(), self.letter)
        simulation_tasks = []

        for _ in range(self.num_simulations // SIMULATIONS_PER_BATCH):
            leaf = root.select()
            child = leaf.expand()
            simulation_tasks.append((child, (child.game_state.copy(), child.player)))
       
        with Pool(cpu_count()) as pool:
            results = pool.map(_batch_simulate_wrapper, [args for _, args in simulation_tasks])

        for (node, _), batch_result in zip(simulation_tasks, results):
            for res in batch_result:
                node.backpropagate(res)

        return max(root.children, key=lambda c: c.N).parent_action


    def __str__(self):
        return "Better MCTS Player"


