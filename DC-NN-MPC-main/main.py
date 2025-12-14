""" Computationally tractable nonlinear robust MPC via DC programming  

Adapted for CSTR Model (Exothermic Reaction A->B)
Original Author: Martin Doff-Sotta
Adapted by: Gemini

"""

import math
import os
import sys
import time

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

from keras.layers import ReLU

import DC_decomposition as DC
# --- Import CSTR specific files ---
import param_init_cstr as param
import param_init_cstr as param_DC
from cstr_model import f, discretise, feasibility, f_full, interp_feas, get_linear_matrices
# -------------------------------------------
from control_custom import eul, dp, seed_cost

import matplotlib
import matplotlib.pyplot as plt

##########################################################################################
#################################### Initialisation ######################################
##########################################################################################

# Solver parameters
N = 30                                         # horizon
T = 3.0                                        # terminal time
delta = T/N                                    # time step
tol1 = 10e-3                                   # tolerance   
maxIter = 3                                    # max number of iterations
N_unit = 8                                     # number of units of neural network (NN)
N_layer = 1                                    # number of hidden layers of NN
batch_size = 32                                # NN training batch size
epochs = 200                                   # NN training epochs
activation = 'relu'                            # activation function ('relu' only)
N_train = 50000                                # number of training sample of NN
N_test = 10                                    # number of test points
load = False                                   # set to False if model has to be retrained
eps = np.finfo(float).eps                      # machine precision
set_param = 'elem'                             # Tube param ('elem' or 'splx')

# Variables initialisation
N_state = param.x_init.size                    # number of states (2 for CSTR)
N_input = param.u_init.size                    # number of inputs (1 for CSTR)
x = np.zeros((N_state, N+1))                   # state
x[:, 0] =  param.x_init
u = np.zeros((N_input, N))                     # control input
u_0 =np.ones((N_input,N))*param.u_init[:,None] # (feasible) guess control input                     
x_0 = np.zeros((N_state, N+1))                 # (feasible) guess trajectory
x_r = np.ones_like(x)*param.h_r[:, None]       # reference state trajectory 
u_r = np.ones_like(u)*param.u_r[:, None]       # reference trajectory
t = np.zeros(N+1)                              # time vector 
K = np.zeros((N, N_input, N_state))            # gain matrix 
real_obj = np.zeros((N, maxIter+1))            # objective value
X_0 = np.zeros((N, maxIter+1, N_state, N+1))   # store state guess trajectories
U_0 = np.zeros((N, maxIter+1, N_input, N))     # store input guess trajectories
X_low = np.zeros((N, maxIter+1, N_state, N+1)) # store perturbed state (lower bound)
X_up = np.zeros((N, maxIter+1, N_state, N+1))  # store perturbed state (upper bound)
S = np.zeros((N, maxIter+1, N_state, N+1))     # store perturbed state    
Q, R = param.Q, param.R                        # penalty matrices

# Activation
if activation == 'relu':
    sigma    = lambda x: np.maximum(x, 0)             # ReLU for numpy
    sigma_cp = lambda x: cp.maximum(x, 0)             # ReLU for cvxpy
    dsigma   = lambda x: np.diag(np.heaviside(x, 0))  # Heaviside
    layer_sigma = ReLU
else:
    print('Inconsistent activation function, abort')
    sys.exit(1)

# Tube cross section parameterisation
if set_param   == 'elem':
    from optimisation import cvx_opt_elem_fast as cvx_opt    # elementwise bounds
elif set_param == 'splx':
    from optimisation import cvx_opt_simplex_fast as cvx_opt # simplex bounds

##########################################################################################
################################### DC decomposition #####################################
##########################################################################################
N_state_DC = 2  # Reaction rate depends on BOTH Ca (x1) and T (x2)
N_input_DC = 1  # Standard format [x; u]

# Generate training samples
x_train = (param_DC.x_max-param_DC.x_min)[:,None]*np.random.rand(N_state_DC,N_train)+param_DC.x_min[:,None]
u_train = (param_DC.u_max-param_DC.u_min)[:,None]*np.random.rand(N_input_DC,N_train)+param_DC.u_min[:,None]

# Note: model.f only returns the nonlinear reaction part
y_train = f(x_train, u_train, param) 
z_train = np.vstack([x_train, u_train])  # assemble input data
avg, std = 0, 1

# Generate test samples
x_test = (param_DC.x_max-param_DC.x_min)[:,None]*np.random.rand(N_state_DC, N_test)+param_DC.x_min[:,None]
u_test = (param_DC.u_max-param_DC.u_min)[:,None]*np.random.rand(N_input_DC, N_test)+param_DC.u_min[:,None]
z_test = np.vstack([x_test, u_test])    # assemble input data
y_test = f(x_test, u_test, param)

# DC split
model_f_DC, model_g, model_h = DC.split(N_unit, N_layer, layer_sigma, 
                                        activation, batch_size, epochs, 
                                        z_train, z_test, y_train, y_test, load)

# Define functions of the decomposition from NN weights
weights_g, weights_h = model_g.get_weights(), model_h.get_weights()

# --- Get Linear Physical Dynamics ---
A_lin_phys, B_lin_phys, d_lin_phys = get_linear_matrices(param)

# --- Hybrid Functions (Physics + NN) ---
def g_hybrid(z, weights, sigma_func):
    """ Combines Linear Physics (Convex) with Neural Network Convex Part """
    # Extract x and u from z. 
    if z.ndim == 1:
        x_ = z[:N_state]
        u_ = z[N_state:]
        lin = A_lin_phys @ x_ + B_lin_phys @ u_ + d_lin_phys
    else:
        x_ = z[:N_state, :]
        u_ = z[N_state:, :]
        lin = A_lin_phys @ x_ + B_lin_phys @ u_ + d_lin_phys[:, None]
    
    nn_out = DC.weight_predict(z, sigma_func, weights)
    return lin + nn_out

# Define Lambda wrappers for use in main loop
g     = lambda z: g_hybrid(z, weights_g, sigma)
h     = lambda z: DC.weight_predict(z, sigma, weights_h)

g_cvx = lambda z: g_hybrid(z, weights_g, sigma_cp)
h_cvx = lambda z: DC.weight_predict(z, sigma_cp, weights_h)
# ---------------------------------------------

# Test fit and split
DC.check(f, lambda z: DC.weight_predict(z, sigma, weights_g), h, z_test, param)

# Sqrt
sqrt_Q = sqrtm(Q)
sqrt_R = sqrtm(R)
Q_lqr =  param.Q_lqr
R_lqr = param.R_lqr

##########################################################################################
############################# LINEARISATION HELPER #######################################
##########################################################################################

def linearise_hybrid_batch(x_traj, u_traj, wg, wh, sigma, dsigma):
    """
    Computes Jacobians for the Hybrid System using Numerical Differentiation.
    This avoids shape mismatches and complexity of manual derivatives for the NN.
    
    A1 = A_phys + d(NN_g)/dx
    B1 = B_phys + d(NN_g)/du
    A2 = d(NN_h)/dx
    B2 = d(NN_h)/du
    """
    # Ensure inputs are 2D arrays (N_dim, N_steps)
    if x_traj.ndim == 1: x_traj = x_traj[:, None]
    if u_traj.ndim == 1: u_traj = u_traj[:, None]

    N_steps = u_traj.shape[1]
    nx = x_traj.shape[0]
    nu = u_traj.shape[0]
    nz = nx + nu
    
    # Pre-allocate arrays
    A1 = np.zeros((N_steps, nx, nx))
    B1 = np.zeros((N_steps, nx, nu))
    A2 = np.zeros((N_steps, nx, nx))
    B2 = np.zeros((N_steps, nx, nu))
    
    perturb = 1e-4
    
    for k in range(N_steps):
        # Construct input vector z0 (must be column vector (nz, 1) for weight_predict)
        z0 = np.vstack([x_traj[:, k][:, None], u_traj[:, k][:, None]])
        
        # 1. Base prediction
        base_g = DC.weight_predict(z0, sigma, wg)
        base_h = DC.weight_predict(z0, sigma, wh)
        
        # 2. Numerical Differentiation
        J_g = np.zeros((nx, nz))
        J_h = np.zeros((nx, nz))
        
        for i in range(nz):
            z_p = z0.copy()
            z_p[i, 0] += perturb
            
            # Predict perturbed
            g_p = DC.weight_predict(z_p, sigma, wg)
            h_p = DC.weight_predict(z_p, sigma, wh)
            
            # Finite difference
            J_g[:, i] = ((g_p - base_g) / perturb).flatten()
            J_h[:, i] = ((h_p - base_h) / perturb).flatten()
        
        # 3. Assemble Matrices
        # A1 = A_phys + d(g_NN)/dx
        A1[k] = A_lin_phys + J_g[:, :nx]
        B1[k] = B_lin_phys + J_g[:, nx:]
        
        # A2 = d(h_NN)/dx
        A2[k] = J_h[:, :nx]
        B2[k] = J_h[:, nx:]
        
    return A1, B1, A2, B2

def discretise_matrices(A_c, B_c, delta):
    """ Simple Euler discretization for matrices """
    # x_{k+1} = (I + delta*A) x_k + (delta*B) u_k
    N_steps = A_c.shape[0]
    A_d = np.zeros_like(A_c)
    B_d = np.zeros_like(B_c)
    
    for k in range(N_steps):
        A_d[k] = np.eye(A_c.shape[1]) + delta * A_c[k]
        B_d[k] = delta * B_c[k]
        
    return A_d, B_d

##########################################################################################
############################### Terminal set computation #################################
##########################################################################################

# NOTE: We use LQR to compute the terminal cost Q_N
# 1. Linearise the full hybrid dynamics (Physics + NN) around the reference point
# We treat the reference as a "batch" of size 1
A1_ref, B1_ref, A2_ref, B2_ref = linearise_hybrid_batch(
    param.h_r[:, None], param.u_r[:, None], 
    weights_g, weights_h, sigma, dsigma
)

# Combine DC parts: A = A1 - A2, B = B1 - B2 to get total continuous Jacobian
# Take index [0] because batch size is 1, keep dim for discretise_matrices
A_cont = (A1_ref[0] - A2_ref[0])[None, :, :]
B_cont = (B1_ref[0] - B2_ref[0])[None, :, :]

# 2. Discretise the linearized system
A_d, B_d = discretise_matrices(A_cont, B_cont, delta)

# 3. Solve Discrete Algebraic Riccati Equation (DARE) using dynamic programming (dp)
P = param.Q
for _ in range(200): 
    # dp(A, B, Q, R, P_next) -> returns K, P_curr
    _, P = dp(A_d[0], B_d[0], param.Q, param.R, P)

# Set the result as the terminal cost
Q_N = P
gamma_N = 10.0 # Heuristic radius for terminal constraint
sqrt_Q_N = sqrtm(Q_N)

print("Terminal set parameters (LQR Approximation):")
print("Q_N\n", Q_N)

##########################################################################################
################################# Feasible trajectory ####################################
##########################################################################################

# Generate a feasible guess trajectory
d_feas = delta
# Simple steady state hold for CSTR guess
x_feas, u_feas, t_feas = feasibility(f_full, x[:, 0], x_r, d_feas, math.floor(T/d_feas), param)
                                                
t_0 = np.arange(N+1)*delta
x_0, u_0 = interp_feas(t_0, t_feas, x_feas, u_feas)

##########################################################################################
####################################### TMPC loop ########################################
##########################################################################################
avg_iter_time = 0
iter_count = 0

for i in range(N):

    print("Computation at time step {}/{}...".format(i+1, N)) 
    
    # Guess trajectory update
    if i > 0:
        x_0[:, :-1] = eul(f_full, u_0[:, :-1], x[:, i], delta, param) 
        
        # Linearise at the end for terminal set
        A1_hat, B1_hat, A2_hat, B2_hat = linearise_hybrid_batch(x_0[:, -2, None], param.u_r[:, None], 
                                                   weights_g, weights_h, sigma, dsigma)  
        
        A_hat, B_hat = discretise_matrices(A1_hat - A2_hat, B1_hat - B2_hat, delta) 
        
        K_hat, _ = dp(A_hat[0, :, :], B_hat[0, :, :], Q, R, Q_N)
        u_0[:, -1, None] = K_hat @ ( x_0[:,-2, None]  - x_r[:, -2, None]) + param.u_r[:, None]
        x_0[:, -1]  = x_0[:, -2] + delta*(f_full(x_0[:, -2], u_0[:, -1] , param)) 
    else:
        pass

    # Iteration
    k = 0 
    real_obj[i, 0] = 5000 
    
    print('{0: <6}'.format('iter'), '{0: <5}'.format('status'), 
          '{0: <18}'.format('time'), '{}'.format('cost'))
    
    while k < maxIter:
        
        # Linearise system at x_0, u_0 using HYBRID linearization
        A1, B1, A2, B2 = linearise_hybrid_batch(x_0[:, :-1], u_0, weights_g, weights_h, sigma, dsigma) 
        
        A, B = discretise_matrices(A1 - A2, B1 - B2, delta)
    
        # Compute K matrix
        P = Q_N
        for l in reversed(range(N)): 
            K[l, :, :], P = dp(A[l, :, :], B[l, :, :], Q_lqr, R_lqr, P)
        
        # Prediction over trajectory
        z_0 = np.vstack([x_0[:, :-1], u_0])
        g_0_val = g(z_0) # Hybrid g
        h_0_val = h(z_0) # Standard h
        
        ##################################################################################
        ############################ Optimisation problem ################################
        ##################################################################################
        
        t_start = time.time()
        # Call Optimisation
        # Ensure optimisation.py is the updated one with ECOS/CLARABEL support
        problem, X_lb, X_ub, v = cvx_opt(x[:, i], x_0, u_0, x_r, u_r, delta, param, 
                                         sqrt_Q, sqrt_R, sqrt_Q_N, gamma_N, K, A1,
                                         A2, B1, B2, param.W_low, param.W_up, g_cvx, h_cvx, g_0_val, h_0_val)

        iter_time = time.time()-t_start
        avg_iter_time += iter_time
        
        print('{0: <5}'.format(k+1), '{0: <5}'.format(problem.status), 
              '{0: <5.2f}'.format(iter_time), '{0: <5}'.format(problem.value))
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
             # Fallback strategy could go here
             break
        
        ##################################################################################
        ############################### Iteration update #################################
        ##################################################################################
        # Save variables 
        X_low[i, k, :, :] = X_lb.value.copy()
        X_up[i, k, :, :] = X_ub.value.copy()
        X_0[i, k, :, :] = x_0.copy()
        U_0[i, k, :, :] = u_0.copy()
        x_0_old = x_0.copy()
        f_x = x_0.copy()
 
        # Input and state update
        s = np.zeros((N_state, N+1))   # state perturbation
        s[:, 0] = x[:, i] - x_0[:, 0]
        Kx = np.zeros_like(v.value)
        
        for l in range(N):
            Kx[:, l] =   K[l, :, :] @ x_0[:, l]
            u_0[:, l] = v.value[:, l] + Kx[:, l]
            
            # Apply input constraints clipping for simulation safety
            u_0[:, l] = np.clip(u_0[:, l], param.u_min, param.u_max)
            
            f_x[:, l+1] = eul(f_full, u_0[:, l], x_0[:, l], delta, param)
            x_0[:, l+1] = f_x[:, l+1]
            s[:, l+1] = x_0[:, l+1]-x_0_old[:, l+1]
           
        S[i, k, :, :] = s.copy()
        k += 1
        iter_count += 1
        real_obj[i, k] = problem.value

    ######################################################################################
    #################################### System update ###################################
    ######################################################################################
    
    u[:, i] = u_0[:, 0]                                      # apply first input
    u_0[:, :-1] = u_0[:, 1:]                                 # shift
    
    # Simulate Real Plant
    x[:, i+1] = eul(f_full, u[:, i], x[:, i], delta, param)
    
    t[i+1] = t[i] + delta
    print('State (Ca, T):', x[:, i], 'Input (Tc):', u[:, i])

##########################################################################################
##################################### Plot results #######################################
##########################################################################################
print('Average time per iteration: ', avg_iter_time/iter_count)

if not os.path.isdir('plot'):
    os.mkdir('plot')
     
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Plotting specific to CSTR (2 States, 1 Input)
fig, axs = plt.subplots(3, 1, figsize=(6, 8))

# Ca (Concentration)
axs[0].plot(t, x[0,:], label=r'$C_A$')
axs[0].plot(t, x_r[0,:], '-.', label=r'$C_A^{ref}$')
axs[0].set_ylabel(r'$C_A$ (mol/L)')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# T (Temperature)
axs[1].plot(t, x[1,:], label=r'$T$')
axs[1].plot(t, x_r[1,:], '-.', label=r'$T^{ref}$')
axs[1].set_ylabel(r'$T$ (K)')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Tc (Coolant Input)
axs[2].plot(t[:-1], u[0,:], label=r'$T_c$')
axs[2].step(t[:-1], u[0,:], where='post')
axs[2].set_ylabel(r'$T_c$ (K)')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)

plt.tight_layout()
plt.savefig('plot/cstr_mpc.png')
plt.show()

## Save data
np.savez('data_CSTR.npz', t=t, x=x, u=u)