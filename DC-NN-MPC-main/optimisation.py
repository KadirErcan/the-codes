"""" Solve optimisation problem

Adapted for CSTR (2 States)
"""
import numpy as np
from scipy.linalg import block_diag
import cvxpy as cp

def cvx_opt_elem_fast(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                      gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    """ 
    Solve optimisation with elementwise bounds and linear objective
    Tailored for 2-State System (CSTR)
    """
    
    Q_N = sqrt_Q_N @ sqrt_Q_N 
    
    # Problem dimensions
    N_state = x_0.shape[0] # Should be 2
    N_input = u_0.shape[0] # Should be 1
    N = u_0.shape[1]
    
    # Number of vertices for 2 states = 2^2 = 4
    N_ver = 4 
        
    # Optimisation variables
    theta = cp.Variable(N+1)               # state cost
    chi = cp.Variable(N)                   # input cost
    v = cp.Variable((N_input, N))          # feedforward input from u = v + K x
    X_lb = cp.Variable((N_state, N+1))     # state lower bound
    X_ub = cp.Variable((N_state, N+1))     # state upper bound
    X_ = {}                                # dictionary for vertices 
    for l in range(N_ver):
        X_[l] = cp.Expression

    # Define blockdiag matrices for page-wise matrix multiplication
    K_ = block_diag(*K)
    
    A1_ = block_diag(*A1)
    A2_ = block_diag(*A2)
    B1_ = block_diag(*B1)
    B2_ = block_diag(*B2)
           
    # Objective
    objective = cp.Minimize(cp.sum(theta) + cp.sum(chi))
        
    # Constraints
    constr = []
        
    # --- Assemble vertices for 2 States ---
    # 0: [LB, LB]
    X_[0] = X_lb
    # 1: [UB, UB]
    X_[1] = X_ub
    # 2: [LB, UB]
    X_[2] = cp.vstack([X_lb[0, :], X_ub[1, :]])
    # 3: [UB, LB]
    X_[3] = cp.vstack([X_ub[0, :], X_lb[1, :]])
      
    for l in range(N_ver):
        # Define some useful variables
        X = X_[l]
        
        # Reshape for matrix multiplication
        X_r = cp.reshape(X[:, :-1], (N_state*N, 1), order='F')
        s_r = cp.reshape(X[:, :-1]-x_0[:, :-1], (N_state * N, 1), order='F')
        K_x = cp.reshape(K_ @ X_r, (N_input, N), order='F')
        
        # Control input deviation v_r
        # v is (N_input, N), K_x is (N_input, N), u_0 is (N_input, N)
        v_r = cp.reshape(v + K_x - u_0, (N_input*N, 1), order='F')
            
        A1_s = cp.reshape(A1_ @ s_r, (A1.shape[1], N), order='F')
        A2_s = cp.reshape(A2_ @ s_r, (A2.shape[1], N), order='F')
        B1_v = cp.reshape(B1_ @ v_r, (B1.shape[1], N), order='F')
        B2_v = cp.reshape(B2_ @ v_r, (B2.shape[1], N), order='F')
            
        # Cost constraints (Soft constraints / Epigraph form)
        # Input cost
        constr += [chi >= cp.norm(sqrt_R @ (v + K_x - u_r), axis=0)**2]
        
        # State cost (Running cost)
        constr += [theta[:-1] >= cp.norm(sqrt_Q @ (X[:, :-1] - x_r[:, :-1]), axis=0)**2]
        
        # Terminal cost
        constr += [theta[-1]  >= cp.quad_form(X[:, -1] - x_r[:, -1], Q_N)]
            
        # Input constraints 
        constr += [v + K_x >= param.u_min[:, None],
                   v + K_x <= param.u_max[:, None]]
            
        # --- DC Tube Dynamics Constraints ---
        # z contains [State; Input] for the Neural Network
        z = cp.vstack([X[:, :-1], v + K_x])
        
        # Upper Bound Dynamics: X_ub >= g(z) - h_linear(z)
        # We apply this to ALL states (0 and 1)
        constr += [X_ub[:, 1:] >= X[:, :-1] + delta * (g(z) - (h_0 + A2_s + B2_v))]
        
        # Lower Bound Dynamics: X_lb <= g_linear(z) - h(z)
        constr += [X_lb[:, 1:] <= X[:, :-1] + delta * (g_0 + A1_s + B1_v - h(z))]
        
    # State constraints (Box constraints)
    constr += [X_lb[:, :-1] >= param.x_min[:, None],
               X_ub[:, :-1] <= param.x_max[:, None]]
    
    # Initial condition constraints
    constr += [X_lb[:, 0] == x_p, 
               X_ub[:, 0] == x_p] 
                        
    # Terminal set constraint (Optional, simplified)
    # constr += [np.sqrt(gamma_N) >= theta[-1]] 

    # Solve problem
    problem = cp.Problem(objective, constr)
    
    # Use robust solver
    # Try CLARABEL (new default) or ECOS
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except:
            problem.solve(solver=cp.SCS, verbose=False)
    
    return problem, X_lb, X_ub, v

# ----------------------------------------------------------------------------------
# Redundant function stubs (kept to prevent import errors if main.py calls them)
# ----------------------------------------------------------------------------------

def cvx_opt_elem(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, gamma_N, 
                 K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    # Redirect to fast version
    return cvx_opt_elem_fast(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                             gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0)

def cvx_opt_simplex(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                    gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    raise NotImplementedError("Simplex constraints not implemented for CSTR. Use set_param='elem'.")

def cvx_opt_simplex_fast(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                         gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    raise NotImplementedError("Simplex constraints not implemented for CSTR. Use set_param='elem'.")