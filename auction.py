import random

class Oracle:
    def get_price(self):
        return random.uniform(3.0, 4.0)

class VCGAuction:
    def __init__(self, oracle):
        self.oracle = oracle

    def run(self, bids, floor=None):
        if floor is None:
            floor = self.oracle.get_price()

        valid = {k: v for k, v in bids.items() if v >= floor}
        if not valid:
            return None

        s = sorted(valid.items(), key=lambda x: x[1], reverse=True)

        winner, win_bid = s[0]
        second = s[1][1] if len(s) > 1 else floor

        # Vickrey (second-price) payment rule.
        payment = second

        return winner, payment, win_bid