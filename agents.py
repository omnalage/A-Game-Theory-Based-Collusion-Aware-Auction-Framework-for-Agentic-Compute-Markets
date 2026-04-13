from dataclasses import dataclass

@dataclass
class Agent:
    agent_id: str
    true_value: float
    is_collusive: bool = False