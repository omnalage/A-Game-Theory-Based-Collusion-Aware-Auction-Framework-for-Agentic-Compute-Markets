import numpy as np
from collections import defaultdict

class BidAdjuster:
    def __init__(self):
        self.history = defaultdict(list)

    def adjust(self, agent_id, bid, floor):
        hist = self.history[agent_id]

        if len(hist) == 0:
            adjusted = max(bid, floor)
        else:
            avg = np.mean(hist)
            dynamic_floor = max(floor, 0.9 * avg)
            adjusted = max(bid, dynamic_floor)

        self.history[agent_id].append(adjusted)
        return adjusted