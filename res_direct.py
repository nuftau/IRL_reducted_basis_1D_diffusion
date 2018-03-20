import numpy as np
import scipy

def res_direct(u0, f, discretisation_z, dt, T, nu, nu_prime, a, b):
    """resoud l'equation du/dt + d/dz(v(z)du/dz) = f
        renvoie u(dt*n_max)
    """
    assert(len(u0) == len(discretisation_z))
    M = calculer_M(discretisation_z, nu, nu_prime, dt)
    n_z = len(discretisation_z)
    u = np.array(u0)
    for t in np.linspace(0, T-dt, T/dt):
        second_membre = np.concatenate(([a(t+dt)], f(discretisation_z[1:-1], t+dt) + u[1:-1]/dt, [b(t+dt)]))
        #print("u:", u)
        #print("second membre:", second_membre)
        #u = np.linalg.solve(M, second_membre)
        u = scipy.linalg.solve_banded((1, 1), M, second_membre)
    return u


def calculer_M(discretisation_z, nu, nu_prime, dt):
    """ calcule la matrice K de la discretisation spatiale
        doit renvoyer une np.matrix
    """
    h_j_moins_1 = np.array(discretisation_z[1:-1]) \
            - np.array(discretisation_z[0:-2])
    h_j = np.array(discretisation_z[2:]) \
            - np.array(discretisation_z[1:-1])
    nu = np.array(nu[1:-1])
    nu_prime = np.array(nu_prime[1:-1])

    diag =  2*nu/(h_j*h_j_moins_1) + 1/dt
    sous_diag= nu_prime/(h_j + h_j_moins_1) \
            -2*nu / (h_j_moins_1 * (h_j + h_j_moins_1))
    sur_diag= -nu_prime/(h_j + h_j_moins_1) \
            -2*nu / (h_j_moins_1 * (h_j + h_j_moins_1))

    diag = np.concatenate(([-1/h_j_moins_1[0]], diag, [1]))
    sous_diag = np.append(sous_diag, 0)
    sous_diag = np.append(sous_diag, 0)
    sur_diag = np.insert(sur_diag, 0, 1/h_j_moins_1[0])
    sur_diag = np.insert(sur_diag, 0, 1/h_j_moins_1[0])
    ab = np.array((sur_diag, diag, sous_diag))

    return ab # np.diag(diag, k=0) + np.diag(sous_diag, k=-1) + np.diag(sur_diag, k=1)

