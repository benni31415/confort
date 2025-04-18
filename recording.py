from dataclasses import dataclass

from game_state import GameState


@dataclass
class Recording:
    game_state: GameState
    action: int
    reward: float