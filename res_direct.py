import numpy as np
import scipy.linalg as la

def res_direct_one_step(K, M, u0, f, dt):
    """resoud l'equation du/dt + d/dz(v(z)du/dz) = f
        renvoie u(dt*n_max)
    """
    return la.solve(K + M/dt, f + M@u0/dt)


def res_direct_tridiagonal(K, M, u0, all_f, dt): 
    """resoud l'equation du/dt + d/dz(v(z)du/dz) = f
        renvoie u(T)
        TODO tester
    """
    u = [u0]
    M_sur_dt = M/dt
    K_plus_M_sur_dt = K + M_sur_dt
    tridiag_for_scipy = np.array((
        np.concatenate(([0], np.diag(K_plus_M_sur_dt, k=1))),
        np.diag(K_plus_M_sur_dt,k=0),
        np.concatenate((np.diag(K_plus_M_sur_dt,k=-1), [0]))))
    for f in all_f:
        u.append(la.solve_banded((1, 1),
                tridiag_for_scipy, f + M_sur_dt@u[-1]))
    return u

def res_direct(u0, f_interieur, discretisation_z, 
        dt, T, nu_plus_un_demi, a, b):
    """resoud l'equation du/dt + d/dz(v(z)du/dz) = f
        renvoie u(T)
    """
    assert(len(u0) == len(discretisation_z))
    assert(len(u0) == len(nu_plus_un_demi) + 1)
    A = calculer_K_plus_M_sur_dt(discretisation_z, 
            nu_plus_un_demi, dt)
    discretisation_z = discretisation_z[1:-1]
    u = u0
    for t in np.linspace(dt, T, T/dt):
        second_membre = np.concatenate(([a(t)], 
            f_interieur(discretisation_z, t) + u[1:-1]/dt, 
            [b(t)]))
        u = la.solve_banded((1, 1), A, second_membre)
    return u

def calculer_M(N):
    """ renvoie une matrice de taille (N+1)x(N+1)
    qui correspond à M dans le formalisme utilisé
    (Mdu/dt + Ku = f)
    """
    return np.diag([0] + [1]*(N-1) + [0])

def calculer_K(discretisation_z, nu_1_2):
    """ renvoie une matrice de taille (N+1)x(N+1)
    qui correspond à K dans le formalisme utilisé
    (Mdu/dt + Ku = f)
    """
    z = discretisation_z
    h_j = z[2:] - z[1:-1]
    h_j_1 = z[1:-1] - z[:-2]
    somme_h = h_j+h_j_1
    nu_plus_1_2 = nu_1_2[1:]
    nu_moins_1_2 = nu_1_2[:-1]

    diag = np.concatenate(([-1/(z[1] - z[0])],
        2*(h_j_1*nu_plus_1_2 + h_j*nu_moins_1_2) / \
        (h_j*h_j_1*somme_h), 
        [1]))
    sur_diag = np.concatenate(([1/(z[1] - z[0])], 
        -2*nu_plus_1_2 / (h_j * somme_h)))
    sous_diag = np.concatenate((-2*nu_moins_1_2/(h_j_1*somme_h),
        [0]))

    return np.diag(diag) + np.diag(sur_diag, k=1) + \
            np.diag(sous_diag, k=-1)

def calculer_K_plus_M_sur_dt(discretisation_z, nu_1_2, dt):
    """ calcule la matrice (K-M/dt) 
        dans le formalisme utilisé
        doit renvoyer une np.matrix, utilisable
        par scipy.linalg.solve_banded
    """

    z = discretisation_z
    h_j = z[2:] - z[1:-1]
    h_j_1 = z[1:-1] - z[:-2]
    somme_h = h_j+h_j_1
    nu_plus_1_2 = nu_1_2[1:]
    nu_moins_1_2 = nu_1_2[:-1]

    diag = np.concatenate(([-1/(z[1] - z[0])], 
        1/dt + 2*(h_j_1*nu_plus_1_2 \
            + h_j*nu_moins_1_2)/(h_j*h_j_1*somme_h), 
        [1]))
    sur_diag = np.concatenate(([0, 1/(z[1] - z[0])], 
        -2*nu_plus_1_2 / (h_j * somme_h)))
    sous_diag = np.concatenate((-2*nu_moins_1_2/(h_j_1*somme_h),
        [0,0]))

    return np.array((sur_diag, diag, sous_diag))

