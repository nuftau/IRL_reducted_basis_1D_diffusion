import numpy as np
import scipy.linalg as la

def res_direct(u0, f, discretisation_z, dt, T, nu, nu_prime, a, b):
    """resoud l'equation du/dt + d/dz(v(z)du/dz) = f
        renvoie u(dt*n_max)
    """
    assert(len(u0) == len(discretisation_z))
    M = calculer_M(discretisation_z, nu, nu_prime, dt)
    n_z = len(discretisation_z)
    u = np.array(u0)
    discretisation_z = discretisation_z[1:-1]
    for t in np.linspace(0, T-dt, T/dt):
        second_membre = np.concatenate(([a(t+dt)], f(discretisation_z, t+dt) + u[1:-1]/dt, [b(t+dt)]))
        #print("second membre : ", second_membre) 
        #print("u : ", u)
        #print()
        u = la.solve_banded((1, 1), M, second_membre)
    return u


def calculer_M(discretisation_z, nu, nu_prime, dt):
    """ calcule la matrice K de la discretisation spatiale
        doit renvoyer une np.matrix
    """

    z = discretisation_z
    diag = [-1/(z[1] - z[0])]
    sur_diag = [0, 1/(z[1] - z[0])]
    sous_diag = []
    for j in range(1, len(discretisation_z) - 1):
        h_j = z[j+1] - z[j]
        h_j_1 = z[j] - z[j-1]
        sur_diag.append(-nu_prime[j]/(h_j+h_j_1) - \
                2*nu[j]/(h_j*(h_j+h_j_1)))
        diag.append(1/dt + 2*nu[j]/(h_j*h_j_1))
        sous_diag.append(nu_prime[j]/(h_j+h_j_1) - \
                2*nu[j]/(h_j_1*(h_j+h_j_1)))
    sous_diag.append(0)
    sous_diag.append(0)
    diag.append(1)

    return np.array((sur_diag, diag, sous_diag))

