import numpy as np

# --- Physical Constants ---
q = 100.0        # L/min Flow rate
V = 100.0        # L Reactor Volume
rho = 1000.0     # g/L Density
Cp = 0.239       # J/(g K) Heat capacity
dH = -5.0e4      # J/mol Heat of reaction (Exothermic)
E_R = 8750.0     # K Activation Energy / R
k0 = 7.2e10      # 1/min Pre-exponential factor
UA = 5.0e4       # J/(min K) Heat transfer coeff
Caf = 1.0        # mol/L Feed Concentration
Tf = 350.0       # K Feed Temperature

# --- Precomputed terms ---
dH_term = -dH / (rho * Cp)
UA_term = UA / (V * rho * Cp)

# --- Constraints ---
# States: x[0]=Ca, x[1]=T
# Defined as numpy arrays so .size works in main.py
x_max = np.array([2.0, 550.0]) 
x_min = np.array([0.0, 250.0])

# Input: u[0]=Tc (Coolant temp)
u_max = np.array([450.0])      
u_min = np.array([250.0])

# --- Initial Conditions ---
# This is the variable your error says is missing!
x_init = np.array([0.877, 324.5]) 
u_init = np.array([300.0])

# --- Reference / Target ---
h_r = np.array([0.877, 324.5]) 
u_r = np.array([300.0])        

# --- Disturbances ---
W_low, W_up = 0, 0

# --- Penalties ---
Q = np.diag([10.0, 0.1])
R = np.diag([0.01])
Q_lqr = Q
R_lqr = R

# --- Terminal Set (Dummy values to prevent errors) ---
u_term = np.array([300.0])
x_term = np.array([0.8, 320.0])