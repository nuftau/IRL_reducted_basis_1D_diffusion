import numpy as np
import scipy.optimize as opti
from calcul_lagrangien import calcul_lagrangien,calcul_gradient
from res_direct import calculer_K
from test_calcul_direct import nu, nu_prime, give_nu_plus_un_demi, zero
from test_calcul_alpha import u, f

"""
def u(z, t):
    return np.exp(z + t)

def f(z, t):
    return np.exp(z+t) - (nu_prime(z)+nu(z)) * np.exp(z+t)



try:
    m = 51
    dt=0.1
    z_max = 3
    discretisation_h = np.array(np.linspace(0, z_max, m))
    nu_1_2 = give_nu_plus_un_demi(discretisation_h, nu)
    u0 = u(discretisation_h, 0)
    dis = discretisation_h[1: -1]
    all_f = [np.concatenate(([0], f(dis, i*dt), [u(z_max,i*dt)])) \
            for i in range(1, 10)]
    K = calculer_K(discretisation_h, nu_1_2)
    #le Phi idéal est dans la direction du suivant:
    # Phi = ((np.cos(discretisation_h)).T).reshape(-1, 1)
    Phi = np.exp(discretisation_h)
    ensemble_apprentissage = [(u0, K, all_f, dt)]
    res_opti = opti.minimize(calcul_lagrangien, Phi, jac=calcul_gradient, args=(m, ensemble_apprentissage), method='BFGS')
    print(res_opti.message)
    print("fonctio objectif en Phi: ", calcul_lagrangien(res_opti.x, m, ensemble_apprentissage))
    print("maintenant, avec Phi0 = cos : ")
    Phi = 10*np.cos(discretisation_h)
    print("fonction objectif en Phi: ", calcul_lagrangien(Phi, m, ensemble_apprentissage))
except:
    raise
"""
#on pose le problème : 
m = 51
dt=0.1
z_max = 3
discretisation_h = np.array(np.linspace(0, z_max, m))
Phi_arbitraire = np.reshape(np.exp(discretisation_h), (m, 1))
nu_1_2 = give_nu_plus_un_demi(discretisation_h, nu)
u0 = u(discretisation_h, 0)
dis = discretisation_h[1: -1]
# all_f = [np.concatenate(([np.exp(i*dt)], f(dis, i*dt), [u(z_max,i*dt)])) \
all_f = [np.concatenate(([0], f(dis, i*dt), [u(z_max,i*dt)])) \
        for i in range(1, 10)]
K = calculer_K(discretisation_h, nu_1_2)
ensemble_apprentissage = [(u0, K, all_f, dt)]
Phi = None
try:
    for _ in range(4):
        if Phi is not None:
            Phi = np.hstack((Phi, Phi_arbitraire))
        else:
            Phi = np.copy(Phi_arbitraire)
        Phi = np.ravel(Phi)
        res_opti = opti.minimize(calcul_lagrangien, Phi, jac=calcul_gradient, args=(m, ensemble_apprentissage), method='BFGS')
        print(res_opti.message)
        print("fonction objectif en Phi: ", calcul_lagrangien(res_opti.x, m, ensemble_apprentissage))
        Phi = np.reshape(Phi, (m, -1))
    print(Phi)
except:
    raise


"""
import matplotlib.pyplot as plt
plt.plot(step_time_h, error, 'r+')
plt.title("Précision avec optimisation sur Phi")
plt.xlabel("1/dt = 1/h_max")
plt.ylabel("Erreur")
plt.yscale("log")
plt.xscale("log")
plt.show()
"""
