import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ml_mr.simulation import Simulation, Variant

# ==== Config ====
n = 10000
n_snps = 10

# ==== Setup Simulation ====
sim = Simulation(n=n)

# Add genetic instruments
snps = [Variant(name=f"snp_{i}", frequency=np.random.uniform(0.1, 0.5)) for i in range(n_snps)]
sim.add_variables(snps)

# Add confounder
sim.add_variable("U", np.random.normal(0, 1, size=n))

# ==== Generate and add exposure ====
def generate_T1(sim):
    G = np.column_stack([sim.get_variable_data(f"snp_{i}") for i in range(n_snps)])
    U = sim.get_variable_data("U")
    return G @ np.random.normal(0.1, 0.02, n_snps) + 0.5 * U + np.random.normal(0, 1, sim.n)

sim.add_variable("T1", generate_T1(sim))

# ==== Generate and add multiple outcomes (based only on T1 + U) ====
def generate_Y1(sim):
    T1 = sim.get_variable_data("T1")
    U = sim.get_variable_data("U")
    return 0.6 * T1 + 0.7 * U + np.random.normal(0, 1, sim.n)

def generate_Y2(sim):
    T1 = sim.get_variable_data("T1")
    U = sim.get_variable_data("U")
    return -0.2 * T1 + 0.6 * U + np.random.normal(0, 1, sim.n)

def generate_Y3(sim):
    T1 = sim.get_variable_data("T1")
    U = sim.get_variable_data("U")
    return 0.1 * T1 + 0.5 * U + np.random.normal(0, 1, sim.n)

sim.add_variable("Y1", generate_Y1(sim))
sim.add_variable("Y2", generate_Y2(sim))
sim.add_variable("Y3", generate_Y3(sim))

# ==== Extract all variables as arrays ====
G = np.column_stack([sim.get_variable_data(f"snp_{i}") for i in range(n_snps)])
T1 = sim.get_variable_data("T1")
Y1 = sim.get_variable_data("Y1")
Y2 = sim.get_variable_data("Y2")
Y3 = sim.get_variable_data("Y3")
U = sim.get_variable_data("U")

# ==== Save separate files ====
os.makedirs("simulated_data_mv", exist_ok=True)

pd.DataFrame(G, columns=[f"snp_{i}" for i in range(n_snps)]).to_csv("simulated_data_mv/X.csv", index=False)
pd.DataFrame({"T1": T1}).to_csv("simulated_data_mv/T.csv", index=False)
pd.DataFrame({"Y1": Y1, "Y2": Y2, "Y3": Y3}).to_csv("simulated_data_mv/Y.csv", index=False)
pd.DataFrame({"U": U}).to_csv("simulated_data_mv/Z.csv", index=False)

# ==== Merge for CLI ====
os.makedirs("simulated_data", exist_ok=True)
merged = pd.concat([
    pd.DataFrame(G, columns=[f"snp_{i}" for i in range(n_snps)]),
    pd.DataFrame({"T1": T1}),
    pd.DataFrame({"Y1": Y1, "Y2": Y2, "Y3": Y3}),
    pd.DataFrame({"U": U})
], axis=1)

merged.to_csv("simulated_data/merged_mv.csv", index=False)

print("✅ Simulated data with one exposure and multivariable outcomes saved.")
