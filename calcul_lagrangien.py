import numpy as np
from res_direct import res_direct, calculer_M, calculer_K, res_direct_one_step

beta = 0.1

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
    ret = calcul_forall_k(calcul_sous_lagrangien_one_step, 0, # METTRE x0 AU LIEU DE 0
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


def calcul_sous_lagrangien_one_step(Phi, u, u0, alpha_k, alpha_k_0, lambda_k, mu_k, K, M, dt):
    ret = np.linalg.norm(u - Phi@alpha_k)/ 2
    ret += lambda_k.T @ \
            ((Phi.T@M@Phi / dt + Phi.T@K@Phi) @ \
            alpha_k - Phi.T@M@Phi @ alpha_k_0 / dt)
    ret += mu_k.T @ (Phi.T@Phi @ alpha_k_0 - Phi.T @ u0)
    return ret
def calcul_sous_lagrangien(Phi, u_k, alpha_k, lambda_k, mu_k, 
        K, delta_t, n_max):
    """ renvoie (1/2)\sum_0^N {
            (u - Phi alpha)^T (u - Phi alpha)
        } + \sum_0^{N-1} {
            lambda ((Phi^T Phi / dt + Phi^T K Phi) alpha
                        - Phi^T Phi alpha /dt)
        } + mu^T (Phi^T Phi alpha_0 - Phi^T u_0)
        H n'est *PAS* pris en compte
    """
    diff = [u_n - Phi @ alpha_n for u_n, alpha_n \
            in zip(u_k, alpha_k)]
    ret = sum([np.linalg.norm(diff_n) for diff_n in diff])/2
    Phi_T_Phi = Phi.T @ Phi
    Phi_T_K_Phi = Phi.T @ K @ Phi
    for n in range(len(lambda_k) - 1):
        ret += lambda_k[n].T @ \
                ((Phi_T_Phi / delta_t + Phi_T_K_Phi) @ \
                alpha_k[n+1] - Phi_T_Phi @ alpha_k[n] / delta_t)
    ret += mu_k.T @ (Phi_T_Phi @ alpha_k[0] - Phi.T @ u_k[0])
    return ret

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

def calcul_forall_k(func, x0, Phi, ensemble_apprentissage):
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



"""def calcul_forall_k(func, x0, Phi, all_u, K, f_interieur, a, b, discretisation_z, delta_t, n_max):
     calcule func pour chaque élément de 
    l'ensemble d'apprentissage définit par les u,K,
    dt, et n_max
    u: solutions avec calcul direct
    K : matrice de discrétisation
    dt = intervalle de discretisation temporelle
    n_max : nombre d'intervalle dt
    f_interieur : definie sur l'intervalle des z, (def f_interieur(z, t):)
                    doit pouvoir prendre un tableau en z
    a : fonction de t qui donne du/dz (0)
    b : fonction de t qui donne u(H)
    NE PREND PAS REN COMPTE H
    # \beta\Phi diag(...) est le seul element qui depend
    # pas de l'ensemble d'apprentissage
    Phi_T_Phi = Phi.T @ Phi
    M = calculer_M(len(discretisation_z) - 1)
    Phi_T_M_Phi = Phi.T @ M @Phi
    Phi_T_K_T_Phi = Phi.T @ K.T @ Phi
    Phi_T_K_Phi = Phi.T @ K @ Phi
    for u_k in all_u:
        alpha_k = calcul_alpha(Phi_T_M_Phi, Phi_T_K_Phi,
                Phi.T @ M @u_k[0], delta_t, n_max, Phi,
                f_interieur, discretisation_z, a, b)
        lambda_k = calcul_lambda(Phi, Phi_T_K_T_Phi, Phi.T, 
                Phi_T_M_Phi, u_k,
                alpha_k, delta_t, n_max)
        mu_k = lambda_k[0]
        x0 += func(Phi, u_k, alpha_k, lambda_k, mu_k, K, delta_t,
                n_max)
"""


def calcul_alpha_one_step(Phi, K, M, u0, f, dt):
    alpha = np.linalg.lstsq(Phi.T@M@Phi, 
        Phi.T@M@u0, rcond=None)[0]
    # seconde equation = Phi^T M Phi \alpha0 = \Phi^T M u0
    second_membre = Phi.T @ f + Phi.T@M@Phi @ alpha / dt
    return alpha, np.linalg.solve(Phi.T@M@Phi / dt + \
                Phi.T@K@Phi, second_membre)


def calcul_alpha(Phi_T_M_Phi, Phi_T_K_Phi, Phi_T_M_u0, dt, n_max,
        Phi=None, f_interieur=None, discretisation_z=None, a=None, b=None):
    """renvoie alpha : tableau de nparray
        all_u doit être un itérable de u_k
        chaque u_k[i] est un nparray de valeurs
        au temps t_i = i*delta_t
        A TESTER : prendre des u_k,
        et verifier que \Phi @ alpha^k = \hat{u}^k ~ u^k
        le probleme c'est qu'il faut une base Phi pour ça :/
    """
    discretisation_z = discretisation_z[1:-1]
    alpha = [np.linalg.lstsq(Phi_T_M_Phi, 
        Phi_T_M_u0, rcond=None)[0]]
    # seconde equation = Phi^T M Phi \alpha0 = \Phi^T M u0
    for n in range(1, 1+int(n_max)):
        second_membre_calcul_direct = np.concatenate(([a(n*dt)], f_interieur(discretisation_z, n*dt) , [b(n*dt)]))
        # second_..._direct correspond à f dans le formalisme
        second_membre = Phi.T @ second_membre_calcul_direct
        #print("equation : ", Phi_T_M_Phi /dt + Phi_T_K_Phi, "*a_{n+1} =", second_membre,"+", Phi_T_M_Phi/dt, "*alpha_n")
        second_membre += Phi_T_M_Phi @ alpha[-1] / dt
        alpha.append(np.linalg.solve(Phi_T_M_Phi / dt + \
                Phi_T_K_Phi, second_membre))
    #    print("alpha_",n,": ", alpha[-1])
    #    print("sin(,",dt,"*",n,"): ", np.sin(n*dt))
    return alpha

def calcul_lambda_one_step(Phi, K, M, u, alpha, dt):
    return np.linalg.solve(Phi.T@M@Phi + dt * Phi.T@K@Phi, 
            dt * Phi.T @ (u - Phi @ alpha))


def calcul_lambda(Phi, Phi_T_K_T_Phi, Phi_T_H, Phi_T_M_Phi,
        u, alpha, delta_t, n_max):
    """attention dans le document c'est bien Phi^T K^T Phi"""
    lambda_ret = [0 for _ in range(n_max)]
    lambda_ret[n_max - 1] = np.zeros(len(alpha[n_max - 1]))
    for n in range(n_max-1, 0, -1):
        second_membre = Phi_T_M_Phi @ lambda_ret[n] + delta_t \
                * Phi_T_H @ (u[n] - Phi @ alpha[n])
        lambda_ret[n-1] = np.linalg.solve(Phi_T_M_Phi + \
                delta_t * Phi_T_K_Phi, second_membre)

    return lambda_ret

