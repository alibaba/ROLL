import random
import numpy as np
from typing import List, Optional
from gymnasium.utils import seeding


def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0]))

    while frontier:
        r, c = frontier.pop()
        if (r, c) not in discovered:
            discovered.add((r, c))
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                r_new, c_new = r + dr, c + dc
                if 0 <= r_new < max_size and 0 <= c_new < max_size:
                    if board[r_new][c_new] == "G":
                        return True
                    if board[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """
    Generates a random valid map with a path from start (S) to goal (G).
    Args:
        size: The size of the map.
        p: The probability of generating a hole (H).
        seed: The seed for the random number generator.
    Returns:
        A list of strings representing the map.
    """
    np_random, _ = seeding.np_random(seed)

    while True:
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        start_r, start_c = np_random.integers(size, size=2)
        goal_r, goal_c = np_random.integers(size, size=2)

        if (start_r, start_c) != (goal_r, goal_c):
            board[start_r][start_c], board[goal_r][goal_c] = "S", "G"
            if is_valid(board, size):
                break

    return ["".join(row) for row in board]
