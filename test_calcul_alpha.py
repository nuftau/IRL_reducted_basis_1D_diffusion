import numpy as np
from calcul_lagrangien import calcul_alpha
from test_calcul_direct import discretisation, zero, nu, nu_prime, test_calcul_direct, test_func_simple
from res_direct import res_direct, calculer_K, calculer_M

def u2(z, t):
    return t+z-z#np.sin(1*t) + z - z

def f2(z, t):
    return 1+z-z#*np.cos(1*t) + z - z

def u(z, t):
    return np.cos(z) * np.sin(1*t)

def f(z, t):
    return 1 * np.cos(1*t) * np.cos(z) + \
            nu_prime(z) * np.sin(z)* np.sin(1*t) + \
            nu(z) * np.cos(z) * np.sin(1*t)
        
def b2(t):
    return u2(1, t)

def b(t):
    return u(1, t)
def phi_test(Phi, discretisation_h, dt):
    """ 
    teste le calcul des alpha avec Phi
    """
    u0 = u(discretisation_h, 0)
    t_f = 1
    
    n_max = t_f/dt
    # maintenant on calcule alpha avec Phi = Id et on vérifie
    # que Phi alpha = hat_u
    K = calculer_K(discretisation_h, nu(discretisation_h), 
            nu_prime(discretisation_h))
    M = calculer_M(len(discretisation_h) - 1)
    Phi_T_M_Phi = Phi.T @M @Phi
    Phi_T_K_Phi = Phi.T @K @Phi
    Phi_T_M_u0 = Phi.T @M @u0
    if len(Phi_T_M_Phi.shape) == 0:
        Phi_T_M_Phi = Phi_T_M_Phi.reshape(1,1)
    if len(Phi_T_K_Phi.shape) == 0:
        Phi_T_K_Phi = Phi_T_K_Phi.reshape(1,1)
    if len(Phi_T_M_u0.shape) == 0:
        Phi_T_M_u0 = Phi_T_M_u0.reshape(1)

    alpha = calcul_alpha(Phi_T_M_Phi, Phi_T_K_Phi,
            Phi_T_M_u0, dt, n_max, Phi, f, discretisation_h, zero, b)

    """u_direct = res_direct(u0, f, discretisation_h, 
                    dt, t_f, 
                    nu(discretisation_h),
                    nu_prime(discretisation_h),
                    zero, b)
    """
    #calcul de l'erreur entre hat_u et u_reel
    u_real = u(discretisation_h, t_f)
    return max(abs(Phi@alpha[-1]- u_real))
    #print("u_direct avec real : ", max(abs(u_direct - u_real)))
    #print("test calcul direct:",test_calcul_direct(u, f, zero, t_f, dt, 1/len(discretisation_h)))
dt=0.1
print("pour la fonction u = cos(z) * sin(t) : \n")
discretisation_h = np.array(np.linspace(0, 1, 51))
Phi = np.identity(len(discretisation_h))
print("avec Phi = identité : ", phi_test(Phi, discretisation_h, dt))

for i in range(6):
    discretisation_h = np.array(np.linspace(0, 1, 10**(i/2+1)))
    Phi = ((np.cos(discretisation_h)).T).reshape(-1, 1)
    print("avec Phi = cos(z_i) et dt = h =",10**(-i/2-1),": ", phi_test(Phi, discretisation_h, 10**(-1-i/2)))

for i in range(6):
    discretisation_h = np.array(np.linspace(0, 1, 10**(i/2+1)))
    Phi1 = [6*np.cos(h) if h <=0.5 else 0 for h in discretisation_h]
    Phi2 = [0.3*np.cos(h) if h >0.5 else 0 for h in discretisation_h]
    Phi3 = [np.sin(h) if h <=0.5 else 0 for h in discretisation_h]
    Phi4 = [np.cos(2*h) if h >0.5 else 0 for h in discretisation_h]
    Phi = (np.array([Phi1, Phi2, Phi3, Phi4]).T).reshape(-1, 4)
    print("avec Phi = cos(z_i) et dt = h =",10**(-i/2-1),": ", phi_test(Phi, discretisation_h, 10**(-1-i/2)))
