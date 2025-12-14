import numpy as np
from scipy.optimize import fsolve

def f_full(x, u, param):
    """ Full dynamics: dx/dt = f(x, u) """
    Ca = x[0]
    T  = x[1]
    Tc = u[0]
    
    k = param.k0 * np.exp(-param.E_R / T)
    r = k * Ca
    
    dCa = (param.q / param.V) * (param.Caf - Ca) - r
    dT  = (param.q / param.V) * (param.Tf - T) + param.dH_term * r + param.UA_term * (Tc - T)
    return np.array([dCa, dT])

def f(x, u, param):
    """ Nonlinear part only (Reaction Kinetics) for DC decomposition """
    Ca = x[0]
    T  = x[1]
    
    k = param.k0 * np.exp(-param.E_R / T)
    r = k * Ca
    
    # The linear parts (Flow, Heat Transfer) are handled by A_lin matrices
    # We only return the difficult nonlinear reaction terms here
    f_nl_0 = -r
    f_nl_1 = param.dH_term * r
    return np.array([f_nl_0, f_nl_1])

def get_linear_matrices(param):
    """ Linear dynamics matrices """
    a11 = -param.q / param.V
    a22 = -(param.q / param.V) - param.UA_term
    A = np.array([[a11, 0], [0, a22]])
    B = np.array([[0], [param.UA_term]])
    d = np.array([(param.q / param.V) * param.Caf, (param.q / param.V) * param.Tf])
    return A, B, d

def discretise(x, u, delta, param):
    """ RK4 Integration """
    k1 = f_full(x, u, param)
    k2 = f_full(x + 0.5 * delta * k1, u, param)
    k3 = f_full(x + 0.5 * delta * k2, u, param)
    k4 = f_full(x + delta * k3, u, param)
    return x + (delta / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def linearise(x, u, delta, param):
    """ Analytic Jacobian for the full system """
    Ca, T, Tc = x[0], x[1], u[0]
    k = param.k0 * np.exp(-param.E_R / T)
    dk_dT = k * (param.E_R / (T**2))
    
    # Jacobian of f_full
    A_c = np.array([
        [-param.q/param.V - k,                 -Ca * dk_dT],
        [param.dH_term * k,    -param.q/param.V - param.UA_term + param.dH_term * Ca * dk_dT]
    ])
    B_c = np.array([[0], [param.UA_term]])
    
    # Discretise
    return np.eye(2) + delta * A_c, delta * B_c

def feasibility(f_model, x0, xr, d_feas, steps, param):
    # Simple steady-state hold for feasibility
    x_feas = np.tile(x0[:,None], (1, steps+1))
    u_feas = np.tile(param.u_init[:,None], (1, steps))
    t_feas = np.arange(steps+1) * 0.1
    return x_feas, u_feas, t_feas

def interp_feas(t_new, t_old, x_old, u_old):
    # Simple interpolation
    x_new = np.zeros((x_old.shape[0], len(t_new)))
    u_new = np.zeros((u_old.shape[0], len(t_new)-1))
    for i in range(x_old.shape[0]):
        x_new[i,:] = np.interp(t_new, t_old, x_old[i,:])
    for i in range(u_old.shape[0]):
        u_new[i,:] = np.interp(t_new[:-1], t_old[:-1], u_old[i,:])
    return x_new, u_new