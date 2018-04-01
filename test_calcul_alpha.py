import numpy as np
from calcul_lagrangien import calcul_alpha
from test_calcul_direct import discretisation, zero, nu, nu_prime, test_calcul_direct, test_func_simple, give_nu_plus_un_demi
from res_direct import res_direct, calculer_K, calculer_M, res_direct_tridiagonal

def u2(z, t):
    return t+z-z#np.sin(1*t) + z - z

def f2(z, t):
    return 1+z-z#*np.cos(1*t) + z - z

def u3(z, t):
    return np.exp(z + t)

def f3(z, t):
    return np.exp(z+t) - (nu_prime(z)+nu(z)) * np.exp(z+t)

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
    u0 = u3(discretisation_h, 0)
    t_f = 3
    nu_1_2 = give_nu_plus_un_demi(discretisation_h, nu)
    K = calculer_K(discretisation_h, nu_1_2) 
    M = calculer_M(len(discretisation_h) - 1)
    second_membre = []
    for t in np.linspace(dt, t_f, t_f/dt):
        second_membre.append(np.concatenate(([np.exp(t)],
            f3(discretisation_h[1:-1], t),
            [u3(discretisation_h[-1], t)])))

    alpha = calcul_alpha(Phi,M,K,u0,second_membre,dt)
    u_direct = res_direct_tridiagonal(K, M, u0, second_membre, dt) 

    diff = [max(abs(Phi@alphan - un)) for alphan, un in zip(alpha, u_direct)]
    #calcul de l'erreur entre hat_u et u_reel
    u_real = u3(discretisation_h, t_f)
    return max(diff)
    #return max(diff)


if __name__ == "__main__":
    dt=0.1
    print("pour la fonction u = cos(z) * sin(t) : \n")
    discretisation_h = np.array(np.linspace(0, 1, 51))
    Phi = np.identity(len(discretisation_h))
    #print("avec Phi = identité : ", phi_test(Phi, discretisation_h, dt))

    error = []
    step_time_h = []
    try:
        for i in range(50):
            discretisation_h = np.array(np.linspace(0, 3, 10**(i/12+1)))
            Phi = ((np.exp(discretisation_h)).T).reshape(-1, 1)
            ret = phi_test(Phi, discretisation_h, 10**(-1-i/12))
            print(ret, "avec Phi = cos(z_i) et dt = h =",10**(-i/12-1))
            error.append(ret)
            step_time_h.append(10**(i/12 + 1))

    except:
        pass

    import matplotlib.pyplot as plt
    plt.plot(step_time_h, error, 'r+')
    plt.title("Précision du calcul des alpha")
    plt.xlabel("1/dt = 1/h_max")
    plt.ylabel("Erreur")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    """
    for i in range(0,14):
        discretisation_h = np.array(np.linspace(0, 1, 10**(i/6+1)))
        Phi1 = [6*np.cos(h) if h <=0.5 else 0 for h in discretisation_h]
        Phi2 = [0.3*np.cos(h) if h >0.5 else 0 for h in discretisation_h]
        Phi3 = [np.sin(h) if h <=0.5 else 0 for h in discretisation_h]
        Phi4 = [np.cos(2*h) if h >0.5 else 0 for h in discretisation_h]
        Phi = (np.array([Phi1, Phi2, Phi3, Phi4]).T).reshape(-1, 4)
        print(phi_test(Phi, discretisation_h, 10**(-1-i/6)), "avec Phi = cos(z_i) et dt = h =",10**(-i/6-1))
    """
