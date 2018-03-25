import numpy as np
import scipy.linalg as la

def res_direct(u0, f, discretisation_z, dt, T, nu, nu_prime, a, b):
    """resoud l'equation du/dt + d/dz(v(z)du/dz) = f
        renvoie u(dt*n_max)
    """
    assert(len(u0) == len(discretisation_z))
    assert(len(nu) == len(nu_prime))
    assert(len(u0) == len(nu))
    A = calculer_K_plus_M_sur_dt(discretisation_z, 
            nu, nu_prime, dt)
    n_z = len(discretisation_z)
    u = np.array(u0)
    discretisation_z = discretisation_z[1:-1]
    for t in np.linspace(0, T-dt, T/dt):
        second_membre = np.concatenate(([a(t+dt)], f(discretisation_z, t+dt) + u[1:-1]/dt, [b(t+dt)]))
        u = la.solve_banded((1, 1), A, second_membre)
    return u

def calculer_M(N):
    """ renvoie une matrice de taille (N+1)x(N+1)
    qui correspond à M dans le formalisme utilisé
    (Mdu/dt + Ku = f)
    """
    return np.diag([0] + [1 for _ in range(N-1)] + [0])

def calculer_K(discretisation_z, nu, nu_prime):
    """ renvoie une matrice de taille (N+1)x(N+1)
    qui correspond à K dans le formalisme utilisé
    (Mdu/dt + Ku = f)
    """
    z = discretisation_z
    diag = [-1/(z[1] - z[0])]
    sur_diag = [1/(z[1] - z[0])]
    sous_diag = []
    for j in range(1, len(discretisation_z) - 1):
        h_j = z[j+1] - z[j]
        h_j_1 = z[j] - z[j-1]
        sur_diag.append(-nu_prime[j]/(h_j+h_j_1) - \
                2*nu[j]/(h_j*(h_j+h_j_1)))
        diag.append(2*nu[j]/(h_j*h_j_1))
        sous_diag.append(nu_prime[j]/(h_j+h_j_1) - \
                2*nu[j]/(h_j_1*(h_j+h_j_1)))
    sous_diag.append(0)
    diag.append(1)

    return np.diag(diag) + np.diag(sur_diag, k=1) + \
            np.diag(sous_diag, k=-1)

def calculer_K_plus_M_sur_dt(discretisation_z, nu, nu_prime, dt):
    """ calcule la matrice (K-M/dt) 
        dans le formalisme utilisé
        doit renvoyer une np.matrix, utilisable
        par scipy.linalg.solve_banded
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

