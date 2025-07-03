# simulate_mv_data_for_quantile_iv.py

import numpy as np
import pandas as pd
import os
from ml_mr.simulation import Simulation, Variant

n = 10000
n_snps = 10

sim = Simulation(n=n)

# Genetic instruments
snps = [Variant(name=f"snp_{i}", frequency=np.random.uniform(0.1, 0.5)) for i in range(n_snps)]
sim.add_variables(snps)

# Confounder
sim.add_variable("U", np.random.normal(0, 1, size=n))

# Multivariable exposures
def generate_T1(sim):
    G = np.column_stack([sim.get_variable_data(f"snp_{i}") for i in range(n_snps)])
    U = sim.get_variable_data("U")
    return G @ np.random.normal(0.1, 0.02, n_snps) + 0.5 * U + np.random.normal(0, 1, sim.n)

def generate_T2(sim):
    G = np.column_stack([sim.get_variable_data(f"snp_{i}") for i in range(n_snps)])
    U = sim.get_variable_data("U")
    return G @ np.random.normal(-0.1, 0.03, n_snps) + 0.3 * U + np.random.normal(0, 1, sim.n)

sim.add_variable("T1", generate_T1)
sim.add_variable("T2", generate_T2)

# Outcome influenced by both T1, T2
def generate_Y(sim):
    T1 = sim.get_variable_data("T1")
    T2 = sim.get_variable_data("T2")
    U = sim.get_variable_data("U")
    return 0.6 * T1 - 0.3 * T2 + 0.7 * U + np.random.normal(0, 1, sim.n)

sim.add_variable("Y", generate_Y)

# Save data
os.makedirs("simulated_data_mv", exist_ok=True)

G = np.column_stack([sim.get_variable_data(f"snp_{i}") for i in range(n_snps)])
Z = sim.get_variable_data("U")
T1 = sim.get_variable_data("T1")
T2 = sim.get_variable_data("T2")
Y = sim.get_variable_data("Y")

pd.DataFrame(G, columns=[f"snp_{i}" for i in range(n_snps)]).to_csv("simulated_data_mv/X.csv", index=False)
pd.DataFrame({"T1": T1, "T2": T2}).to_csv("simulated_data_mv/T.csv", index=False)
pd.DataFrame({"Y": Y}).to_csv("simulated_data_mv/Y.csv", index=False)
pd.DataFrame({"U": Z}).to_csv("simulated_data_mv/Z.csv", index=False)

print("✅ Multivariable simulated data saved.")
