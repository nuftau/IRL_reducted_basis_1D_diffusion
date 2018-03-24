import scipy.optimize as opti
import numpy as np

def f(x, m):
    """x is passed as a 1D-array : needed to be reshaped
        m is the number of lines in the matrix Phi
    """
    x = np.reshape(x, (m, -1)) # transformation en matrice
    x = np.ravel(x) # transformation en 1D-array
    return x[0]**2 + 2*x[1]**2 +3*x[2]**2

def grad(x, m):
    """x is passed as a 1D-array : needed to be reshaped
        m is the number of lines in the matrix Phi
    """
    return np.array([2*x[0], 4*x[1], 6*x[2], 0, 0, 0])

x0 = np.ravel(np.array([[4,3,1],[2,3,4]]))
print("avec x0=0:", f(x0, 2), grad(x0, 2))
print(opti.minimize(f, x0, args=(2), jac=grad, method='BFGS'))

