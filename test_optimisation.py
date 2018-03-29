import numpy as np
import scipy.optimize as opti
from calcul_lagrangien import calcul_lagrangien,calcul_gradient
from res_direct import calculer_K
from test_calcul_direct import nu, give_nu_plus_un_demi, zero
from test_calcul_alpha import u, f

m = 51
dt=0.1
discretisation_h = np.array(np.linspace(0, 1, m))
nu_1_2 = give_nu_plus_un_demi(discretisation_h, nu)
u0 = u(discretisation_h, 0)
dis = discretisation_h[1: -1]
all_f = [np.concatenate(([0], f(dis, i*dt), [u(1,i*dt)])).reshape(len(dis)+2,1) \
        for i in range(1, 10)]
K = calculer_K(discretisation_h, nu_1_2)
#le Phi idéal est dans la direction du suivant:
# Phi = ((np.cos(discretisation_h)).T).reshape(-1, 1)
#TODO utiliser des nparrays 1D et np.outer pour ab^T (abété konasai)
Phi = np.sin(discretisation_h)
ensemble_apprentissage = [(u0, K, all_f, dt)]
res_opti = opti.minimize(calcul_lagrangien, Phi, jac=calcul_gradient, args=(m, ensemble_apprentissage), method='BFGS')
Phi = np.cos(discretisation_h)
print(res_opti.message)
print("fonctio objectif en Phi: ", calcul_lagrangien(res_opti.x, m, ensemble_apprentissage))
print("fonctio objectif en cos: ", calcul_lagrangien(Phi, m, ensemble_apprentissage))
print("TODO trouver un problème et sa matrice Phi optimale POUR DE VRAI avant de faire la comparaison avec la base trouvée par optimisation !")
"""print("avec Phi = sin:", calcul_lagrangien(Phi, m, ensemble_apprentissage))
Phi = np.cos(discretisation_h)
print("avec Phi = cos:", calcul_lagrangien(Phi, m, ensemble_apprentissage))
Phi = np.array([1 for k in discretisation_h])
print("avec Phi = 1:", calcul_lagrangien(Phi, m, ensemble_apprentissage))
Phi = np.array([2 for k in discretisation_h])
print("avec Phi = 2:", calcul_lagrangien(Phi, m, ensemble_apprentissage))
Phi = np.array([3 for k in discretisation_h])
print("avec Phi = 3:", calcul_lagrangien(Phi, m, ensemble_apprentissage))
Phi = np.array([4 for k in discretisation_h])
print("avec Phi = 4:", calcul_lagrangien(Phi, m, ensemble_apprentissage))
"""
