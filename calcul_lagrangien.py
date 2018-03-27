import numpy as np
from res_direct import res_direct, calculer_M, calculer_K, res_direct_one_step, res_direct_tridiagonal

beta = 0.00001

def calcul_lagrangien(Phi, m, ensemble_apprentissage):
    """ renvoie le lagrangien en Phi,
    avec l'ensemble d'apprentissage
        un tuple de l'ensemble d'apprentissage est:
        (u0, K, f, dt)
        A TESTER !
        prendre 2-3 u_k et m == 1
    """
    Phi = np.reshape(Phi, (m, -1))
    x0 = beta/2 * sum([(1 - np.linalg.norm((Phi.T)[i])**2)**2 \
            for i in range(Phi.shape[1])])
    print("np.linalg.norm((Phi.T)[i]) : ", np.linalg.norm((Phi.T)[0]))
    ret = calcul_forall_k(calcul_sous_lagrangien, x0,
            Phi, ensemble_apprentissage)
    return ret


def calcul_gradient(Phi, m, all_u, K, delta_t, n_max):
    """ renvoie le gradient du lagrangien en Phi,
    avec l'ensemble d'apprentissage"""
    Phi = np.reshape(Phi, (m, -1))
    diagonale = [1 - np.linalg.norm(Phi[:][i])**2 \
            for i in range(Phi.shape[1])]
    ret = beta * Phi @ np.diag(diagonale)
    #return np.ravel(calcul_forall_k(calcul_sous_gradient, ret, 
    #        Phi, all_u, K, delta_t, n_max))

def calcul_sous_lagrangien(Phi, u_k, alpha_k, lambda_k, mu_k, 
        K, M, delta_t):
    """ renvoie (1/2)\sum_0^N {
            (u - Phi alpha)^T (u - Phi alpha)
        } + \sum_0^{N-1} {
            lambda ((Phi^T Phi / dt + Phi^T K Phi) alpha
                        - Phi^T Phi alpha /dt)
        } + mu^T (Phi^T Phi alpha_0 - Phi^T u_0)
        H n'est *PAS* pris en compte
        en fait si tout va bien L = G
        donc pas besoin de faire la boucle suivante:
        for n in range(len(lambda_k) - 1):
            ret += lambda_k[n].T @ \
                    ((Phi_T_Phi / delta_t + Phi_T_K_Phi) @ \
                    alpha_k[n+1] - Phi_T_Phi @ alpha_k[n] / delta_t)
        ret += mu_k.T @ (Phi_T_Phi @ alpha_k[0] - Phi.T @ u_k[0])
    """
    diff = np.array([u_n - Phi @ alpha_n for u_n, alpha_n \
            in zip(u_k, alpha_k)])
    return np.linalg.norm(diff)**2 / 2


def calcul_sous_gradient(Phi, u_k, alpha_k, lambda_k, mu_k, 
        K, delta_t, n_max):
    """
    renvoie \sum_{n=0}^N {
        H\Phi\alpha_n\alpha_n^T - H u_n\alpha_n^T}
    + \sum_{n=0}^{n-1} {
        \Phi (\lambda_n((\alpha_{n+1} - \alpha_n)/delta_t)^T
        +(\alpha_{n+1} - \alpha_n)/delta_t * \lambda_n^T)
        +K^T\Phi \lambda_n \alpha_{n+1}^T
        +K\Phi \alpha_{n+1}\lambda_n^T}
    - u_0 \mu^T
    """
    return 0

def calcul_forall_k_one_step(func, x0, Phi, ensemble_apprentissage):
    """ calcule func pour chaque élément de 
    l'ensemble d'apprentissage:
        un tuple de l'ensemble d'apprentissage est:
        (u0, K, f, dt)
        en conservant le formalisme utilisé dans le pdf
        on pourra par la suite ajouter H dans le tuple...
        NE FAIS QUE DES ONE_STEP
    """
    # \beta\Phi diag(...) est le seul element qui depend
    # pas de l'ensemble d'apprentissage
    for u0, K, f, dt in ensemble_apprentissage:
        M = calculer_M(len(u0) - 1)
        Phi_T_M_Phi = Phi.T@M@Phi
        Phi_T_K_T_Phi = Phi.T @ K.T @ Phi
        Phi_T_K_Phi = Phi.T @ K @ Phi

        alpha_k_0, alpha_k = calcul_alpha_one_step(Phi, K, M, u0, f, dt)
        u = res_direct_one_step(K, M, u0, f, dt)

        lambda_k = calcul_lambda_one_step(Phi, K, M, u, alpha_k, dt)
        mu_k = lambda_k
        x0 += func(Phi, u, u0, alpha_k, alpha_k_0, 
                lambda_k, mu_k, K, M, dt)
    return x0

def calcul_forall_k(func, x0, Phi, ensemble_apprentissage):
    """
        calcule func pour chaque élément de 
        l'ensemble d'apprentissage:
        un tuple de l'ensemble d'apprentissage est:
        (u0, K, all_f, dt)
        avec all_f un itérable comme dans calcul_alpha
        en conservant le formalisme utilisé dans le pdf
        on pourra par la suite ajouter H dans le tuple...

    """
    # \beta\Phi diag(...) est le seul element qui depend
    # pas de l'ensemble d'apprentissage
    for u0, K, all_f, dt in ensemble_apprentissage:
        M = calculer_M(len(u0) - 1)

        alpha = calcul_alpha(Phi, M, K, u0, all_f, dt)
        u = res_direct_tridiagonal(K, M, u0, all_f, dt)

        lambda_k = calcul_lambda(Phi, K, M, u, alpha, dt, len(all_f))
        mu_k = lambda_k[0]
        x0 += func(Phi, u, alpha, lambda_k, mu_k, K, M, dt)
    return x0

def calcul_alpha(Phi, M, K, u0, all_f, dt):
    """
        calcule les alpha associé à Phi.
        all_f doit être itérable, où le nième
        élément est le second membre de l'équation
        au temps t^n.

        renvoie: tableau de nparray, 
                    alpha^n pour chaque temps t^n,
                    de n=0 à n_max.
    """
    Phi_T_M_Phi = Phi.T@M@Phi
    if len(Phi_T_M_Phi.shape) == 0:
        zero = np.array(0).reshape(1,1)
    else:
        zero = 0
    alpha = [np.linalg.lstsq(Phi_T_M_Phi, 
        Phi.T@M@u0, rcond=None)[0]]
    # seconde equation = Phi^T M Phi \alpha0 = \Phi^T M u0
    Phi_T_M_Phi_dt = zero + Phi_T_M_Phi / dt
    for fn in all_f:
        second_membre = Phi.T @ fn + Phi_T_M_Phi_dt @ alpha[-1]
        alpha.append(np.linalg.solve(Phi_T_M_Phi_dt + \
                Phi.T@K@Phi, second_membre))
    return alpha


def calcul_lambda(Phi, K, M, u, alpha, dt, n_max):
    """attention dans le document c'est bien Phi^T K^T Phi"""
    lambda_ret = [0 for _ in range(n_max)]
    lambda_ret[n_max - 1] = np.zeros(len(alpha[n_max - 1]))
    Phi_T_M_Phi = Phi.T@M@Phi
    for n in range(n_max-1, 0, -1):
        lambda_ret[n-1] = np.linalg.solve(Phi_T_M_Phi + \
                dt * Phi.T@K.T@Phi, Phi_T_M_Phi@lambda_ret[n] + \
                dt * Phi.T @ (u - Phi @ alpha[n]))
    return lambda_ret

