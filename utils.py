import pandas as pd
import random

DATASET_PATH = "ml_hardware.csv"

hardware = pd.read_csv(DATASET_PATH)
hardware = hardware[hardware["Type"] == "GPU"]

def get_gpu_cost():
    return hardware["Release price (USD)"].sample().values[0]

def cost_to_value(cost):
    return cost / 10000

def generate_agents(n=50):
    agents = []
    sample = hardware.sample(n)

    for i, row in sample.iterrows():
        value = cost_to_value(row["Release price (USD)"])
        collusive = (i >= int(0.7 * n))  # 30% collusion
        agents.append((value, collusive))

    return agents