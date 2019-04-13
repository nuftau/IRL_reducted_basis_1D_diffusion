import numpy as np
from itertools import repeat
import concurrent.futures
from res_direct import res_direct, calculer_M, calculer_K, res_direct_one_step, res_direct_tridiagonal

beta = 0.0000001

def calcul_error(Phi, m, ensemble_apprentissage):
    Phi = np.reshape(Phi, (m, -1))
    x0 = 0
    return calcul_forall_k(sous_calcul_erreur, x0,
            Phi, ensemble_apprentissage)

def calcul_lagrangien(Phi, m, ensemble_apprentissage):
    """ renvoie le lagrangien en Phi,
    avec l'ensemble d'apprentissage en entree
        un tuple de l'ensemble d'apprentissage est:
        (u0, K, f, dt)
        A TESTER !
        prendre 2-3 u_k et m == 1
    """
    Phi = np.reshape(Phi, (m, -1))
    x0 = beta/2 * sum([(1 - np.vdot(Phi[:,i], Phi[:,i]))**2 \
            for i in range(Phi.shape[1])])
    ret = calcul_forall_k(calcul_sous_lagrangien, x0,
            Phi, ensemble_apprentissage)
    return ret

disp = False
def calcul_gradient(Phi, m, ensemble_apprentissage, display=False):
    """ renvoie le gradient du lagrangien en Phi,
    avec l'ensemble d'apprentissage"""
    Phi = np.reshape(Phi, (m, -1))
    diagonale = [1 - np.vdot(Phi[:,i], Phi[:,i]) \
            for i in range(Phi.shape[1])]
    x0 = - 2* beta * Phi @ np.diag(diagonale)
    if display:
        disp = True
        print("x0 : ", x0)
    return np.ravel(calcul_forall_k(calcul_sous_gradient, x0, 
            Phi, ensemble_apprentissage))

def calcul_sous_lagrangien(Phi, u_k, f_k, alpha_k, lambda_k, mu_k, 
        K, M, delta_t):
    """ renvoie (1/2)*sum_0^N {
            (u - Phi alpha)^T (u - Phi alpha)
        } 
        H n'est *PAS* pris en compte
    """
    diff = [u_n - Phi @ alpha_n for u_n, alpha_n \
            in zip(u_k, alpha_k)]
    diff.pop(0)
    diff.pop(-1) # pas opti mais c'est pour le bien du test
    # on pop les valeurs extremes pour avoir un gradient exact (?)
    diff = np.array(diff)
    return np.vdot(diff,diff) / 2

def sous_calcul_erreur(Phi, u_k, f_k, alpha_k, lambda_k, mu_k, 
        K, M, delta_t):
    """ renvoie (1/2)*sum_0^N {
            (u - Phi alpha)^T (u - Phi alpha)
        } 
        H n'est *PAS* pris en compte
    """
    return max([max(abs(u_n - Phi @ alpha_n)) for u_n, alpha_n \
            in zip(u_k, alpha_k)])


def calcul_sous_gradient(Phi, u_k, f_k, alpha_k, lambda_k, mu_k, 
        K, M, dt):
    """
        renvoie la partie du gradient
        spécifique à l'échantillon k.
    """
    ret = 0
    if len(alpha_k[0].shape) == 1:
        # si on est en dimension 1, il faut redimensionner
        # les tableaux pour que les produits
        # matriciels soient valides
        for n in range(len(lambda_k)):
            alpha_k[n] = alpha_k[n].reshape(
                    (alpha_k[n].shape[0], -1))
            lambda_k[n] = lambda_k[n].reshape(
                    (lambda_k[n].shape[0], -1))
            u_k[n] = u_k[n].reshape(
                    (u_k[n].shape[0], -1))

        # la taille d'alpha est (celle de lambda) +1
        alpha_k[-1] = alpha_k[-1].reshape(
                (alpha_k[-1].shape[0], -1))
        mu_k = mu_k.reshape((mu_k.shape[0], -1))

    for n in range(len(lambda_k)):
        ret_n = Phi@alpha_k[n] @ alpha_k[n].T
        ret_n -= u_k[n] @ alpha_k[n].T

        diff = alpha_k[n+1] - alpha_k[n]

        ret_n += M@Phi@(lambda_k[n]@diff.T+ \
                diff@lambda_k[n].T)/dt
        ret_n += np.outer(K@Phi@alpha_k[n+1],lambda_k[n])
        ret_n += np.outer(K.T@Phi@lambda_k[n],alpha_k[n+1])
        ret_n -= np.outer(f_k[n],lambda_k[n])

        # ici c'est bien f_k[n], car l'utilisateur envoie
        # (normalement) pas f0. donc all_f[n] = f_{n+1}
        ret += ret_n
            
    ret += M@Phi@(alpha_k[0]@mu_k.T + mu_k@alpha_k[0].T)
    ret -= Phi@alpha_k[0] @ alpha_k[0].T
    ret += u_k[0] @ alpha_k[0].T
    
    ret -= np.outer(M@u_k[0],mu_k)
    return ret


def call_func_with_computed_data(tuple_data_func_Phi):
    """ fait le travail de chercher les alpha/lambda/mu
    pour un élément de la base d'apprentissage tuple_data.
    appelle func.
    Est faite pour etre appelee en parallel
    """
    u0, K, all_f, dt = tuple_data_func_Phi[0]
    func = tuple_data_func_Phi[1]
    Phi = tuple_data_func_Phi[2]
    M = calculer_M(len(u0) - 1)

    alpha = calcul_alpha(Phi, M, K, u0, all_f, dt)
    u = res_direct_tridiagonal(K, M, u0, all_f, dt)

    lambda_k = calcul_lambda(Phi, K, M, u, alpha, dt, len(all_f))
    mu_k = lambda_k[0]/dt
    return func(Phi, u, all_f, alpha, lambda_k, mu_k, K, M, dt)
 

def calcul_forall_k(func, x0, Phi, ensemble_apprentissage):
    """
        calcule func pour chaque élément de 
        l'ensemble d'apprentissage:
        un tuple de l'ensemble d'apprentissage est:
        (u0, K, all_f, dt)
        avec all_f un itérable comme dans calcul_alpha
        en conservant le formalisme utilisé dans le pdf
        on pourra par la suite ajouter H dans le tuple...
        On pourra aussi faire en sorte que K ne soit plus
        constant : tout ce travail est à faire dans
        call_func_with_computed_data

    """
    # \beta\Phi diag(...) est le seul element qui depend
    # pas de l'ensemble d'apprentissage
    with concurrent.futures.ProcessPoolExecutor() as executor:
        x0 += sum(executor.map(call_func_with_computed_data, 
            zip(ensemble_apprentissage, repeat(func), repeat(Phi))))
    return x0

def calcul_alpha(Phi, M, K, u0, all_f, dt):
    """
        calcule les alpha associé à Phi.
        all_f doit être itérable, où le nième
        élément est le second membre de l'équation
        au temps t^n.
        autrement dit, f_0 n'est pas dans all_f

        renvoie: tableau de nparray, 
                    alpha^n pour chaque temps t^n,
                    de n=0 à n_max.
    """
    Phi_T_M_Phi = Phi.T@M@Phi
    try:
        sec_membre = np.reshape(Phi.T@M@u0, (Phi_T_M_Phi.shape[0],))
        alpha = [np.linalg.solve(Phi_T_M_Phi, sec_membre)]
    except(np.linalg.linalg.LinAlgError):
        alpha = [np.linalg.lstsq(Phi_T_M_Phi, 
            Phi.T@M@u0, rcond=None)[0]]
        print("calcul d'alpha0 par moindres carres")
    # seconde equation = Phi^T M Phi \alpha0 = \Phi^T M u0
    Phi_T_M_Phi_dt = Phi_T_M_Phi / dt
    Phi_T_K_Phi = Phi.T@K@Phi
    #on fait zero + pour être sur d'avoir un nparray de dim2
    
    for fn in all_f:
        second_membre = Phi.T @ fn + Phi_T_M_Phi_dt @ alpha[-1]
        try:
            alpha.append(np.linalg.solve(Phi_T_M_Phi_dt + \
                    Phi_T_K_Phi, second_membre))

        except(np.linalg.linalg.LinAlgError):
            alpha.append(np.linalg.lstsq(Phi_T_M_Phi_dt + \
                    Phi.T@K@Phi, second_membre, rcond=None)[0])
            print("warning : singular matrix")
    return alpha


def calcul_lambda(Phi, K, M, u, alpha, dt, n_max):
    """attention dans le document c'est bien Phi^T K^T Phi"""
    lambda_ret = [0 for _ in range(n_max)]
    lambda_ret[n_max - 1] = np.zeros((Phi.shape[1], ))
    # lambda est de taille m (nombre de colonnes de Phi)
    Phi_T_M_Phi = Phi.T@M@Phi
    for n in range(n_max-1, 0, -1):
        try:
            lambda_ret[n-1] = np.linalg.solve(Phi_T_M_Phi + \
                    dt * Phi.T@K.T@Phi, Phi_T_M_Phi@lambda_ret[n] + \
                    dt * Phi.T @ (u[n] - Phi @ alpha[n]))
        except(np.linalg.linalg.LinAlgError):
            lambda_ret[n-1] = np.linalg.lstsq(Phi_T_M_Phi + \
                    dt * Phi.T@K.T@Phi, Phi_T_M_Phi@lambda_ret[n] + \
                    dt * Phi.T @ (u[n] - Phi @ alpha[n]), rcond=None)[0]
            print("warning : singular matrix : ")
    return lambda_ret

