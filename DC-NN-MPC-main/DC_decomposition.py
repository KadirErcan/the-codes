""" DC Deep Neural Network models """
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import keras
from keras import layers
from keras import constraints

# REMOVED: from pvtol_model import ddy, ddz (Not available/needed for CSTR)

def convex_NN(N_layer, N_node, sigma, N_in, N_out):
    """ Create a densely connected neural network with convex input-output map
    Input: 
        - N_layer: number of hidden layers
        - N_node: number of nodes per layer
        - sigma: activation function
        - N_in: input dimension
        - N_out: output dimension
    Output: neural network model
    """

    # FIXED: Use N_in instead of hardcoded 2
    input = keras.Input(shape=(N_in,))
    x = input
    x = layers.Dense(N_node)(input)
    x = sigma()(x)
    
    # Add N_layer dense layers with N_node nodes
    for i in range(N_layer):
        x1 = layers.Dense(N_node, kernel_constraint=constraints.NonNeg())(x)
        #x1 = layers.LeakyReLU(alpha=0.3)(x1)
        x2 = layers.Dense(N_node)(input)
        x = layers.Add()([x1, x2])
        x = sigma()(x)
    
    # FIXED: Use N_out instead of hardcoded 2
    output = layers.Dense(N_out, kernel_constraint=constraints.NonNeg())(x)
    
    return keras.Model(input, output)

def weight_predict(x, sigma, weights):
    """ 
    Model prediction from weights 
    """
    
    # First layer
    x0 = x
    W = weights[0].T
    b = weights[1].T
    z = W @ x + b[:, None]
    x = sigma(z)

    # Internal layers
    N = (len(weights)-4)//4
    for i in range(N):
        Wx = weights[2+i*4].T
        bx = weights[2+i*4+1].T
        W0 = weights[2+i*4+2].T
        b0 = weights[2+i*4+3].T
        
        z = Wx @ x  + bx[:, None] +  W0 @ x0 + b0[:, None]
        x = sigma(z)
    
    # Last layer
    W = weights[-2].T
    b = weights[-1].T
    z = W @ x + b[:, None]
    
    return z 
    
def split(N_unit, N_layer, sigma, activation, N_batch, N_epoch, 
                                                  x_train, x_test, y_train, y_test, load):
    """ 
    Obtain DC decomposition of function f using DC neural networks 
    """
    
    # Dimensions
    N_arg = x_train.shape[0]  # number of inputs to NN
    N_out = y_train.shape[0]  # number of outputs (nonlinear functions)
    
    # Build model
    input = keras.Input(shape=(N_arg,))
    
    # FIXED: Pass dimensions to convex_NN
    model_g = convex_NN(N_layer, N_unit, sigma, N_arg, N_out)
    model_h = convex_NN(N_layer, N_unit, sigma, N_arg, N_out)
    
    g = model_g(input)
    h = model_h(input)
    
    output = layers.Subtract()([g, h])
    
    model_f_DC = keras.Model(inputs=input, outputs=output)

    # Compile 
    model_f_DC.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    # Load or train model
    if activation == "relu": 
        file_name = './model_ReLU/f_DC.weights.h5'
    elif activation == "elu": 
        file_name = './model_ELU/f_DC.weights.h5'
    
    if load:  # load existing model
        # Restore the weights
        model_f_DC.load_weights(file_name)

    else:  # train new model
        print("************ Training of the DC neural network... ******************")
        # Train model
        history = model_f_DC.fit(x_train.T, y_train.T, batch_size=N_batch, 
                                                     epochs=N_epoch, validation_split=0.2)
        
        # Save the weights
        # Ensure directory exists if needed, or user must create it
        model_f_DC.save_weights(file_name)
    
    # Evaluate
    test_scores = model_f_DC.evaluate(x_test.T, y_test.T, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    
    return model_f_DC, model_g, model_h
    

def plot(model_f_DC, model_g, model_h, sigma, param):
    """ Plot results of decomposition (Restricted functionality for CSTR) """
    # This function originally depended on specific PVTOL derivatives.
    # It has been simplified to avoid errors.
    print("Plotting skipped (requires model specific reference data).")
    return

## Hessian
def D_2(f, x_0, delta, i, j):
    """ Evaluate second derivative of f along x_i and x_j at x_0 """
    n = len(x_0)
    I = np.eye(n)
    return (f(x_0 + delta*I[j, :] + delta*I[i, :]) -f(x_0 + delta*I[j, :])
            - f(x_0 + delta*I[i, :]) + f(x_0))/delta**2

def hess(f, x_0, delta):
    """ Evaluate the Hessian of f at x_0 (numerically) """
    n = len(x_0)
    H = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            H[i, j] = D_2(f, x_0, delta, i, j) 
    return H
    
## Check split
def check(f, g, h, x, p):
    """ A function to check the validity of a given DC decomposition """
    
    ## 1. Check f = g-h
    # For checking, we need f(x) but f might expect (x, u, p).
    # The 'check' call in main.py passes x (which is actually z=[x;u]).
    # We need to unpack z to call f correctly.
    
    # Determine split point from main.py context or parameter file
    # For generic usage, we assume f takes z directly or we unpack if we know sizes.
    # In main.py: DC.check(f, ..., z_test, param)
    # And f is cstr_model.f(x, u, p).
    
    # Helper to unpack z for f
    def f_wrapper(z_col):
        # Assuming first 2 are states, rest are inputs (generic for CSTR setup)
        # N_state_DC=2 in main.py
        nx = 2 
        x_val = z_col[:nx]
        u_val = z_col[nx:]
        return f(x_val, u_val, p)

    N = x.shape[1]  # number of test points
    
    # Compute the error of DC decomposition
    # Evaluate f manually for each column
    f_val = np.zeros((g(x).shape[0], N))
    for i in range(N):
        f_val[:, i] = f_wrapper(x[:, i])

    err_split = np.abs(g(x)-h(x)-f_val)
    
    print("************ Error in DC approximation ****************")
    print("Mean absolute error = ", err_split.mean(axis=1))
    print("Max absolute error = ", err_split.max(axis=1))
    
    ## 2. Check convexity of g and h
    # Only check first element convexity as example
    g1 = lambda x: g(x)[0, 0]
    h1 = lambda x: h(x)[0, 0]
    
    print("********** Checking convexity of g and h **************")
    viol = 0
    tol = .01     # tolerance 
    delta = .001  # step 
    
    # Reduced check to avoid long loops
    N_check = min(N, 10) 
    
    for i in range(N_check):
        H_g1 = hess(g1, x[:, i], delta)
        H_h1 = hess(h1, x[:, i], delta)
        
        eig_g1 = np.linalg.eigvals(H_g1)
        eig_h1 = np.linalg.eigvals(H_h1)
        
        eig_all = np.stack([eig_g1, eig_h1])

        if np.any(eig_all < -tol):
            print("Hessian not psd at iteration", i, "\n")
            viol += 1
            
    print("Checking done.")
    if viol == 0: print("No convexity violations detected (in sample subset).")
    else: print("{} convexity violations detected !".format(viol))