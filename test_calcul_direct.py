import numpy as np
from res_direct import res_direct_tridiagonal, calculer_K, calculer_M

def discretisation(h_max):
    ret = [0]
    actual_h= 0
    while actual_h + h_max < 1:
        actual_h += max(h_max/10, abs(np.cos(actual_h))*h_max)
        ret.append(actual_h)
    ret.append(1)
    return np.array(ret)

def zero(t):
    return 0

def nu(z):
    return (1 - z)**2

def nu_prime(z):
    return - 2 * (1 - z)

def nu_seconde(z):
    return 2

def b_test(u, h_f):
    def b(t):
        return u(h_f, t)
    return b

def test_func_indep_z(dt, h_max):
    def cst_u(z, t):
        return 1/(t+0.01) +z-z

    def cst_f(z, t):
        """should be du/dt - nu'(z) * du/dz - nu(z) d²u/dz²"""
        return -1/((t+0.01)**2) + z - z # +z - z pour np.array

    def cst_b(t):
        return cst_u(0, t)

    t_f = 1
    return test_calcul_direct(cst_u, cst_f, zero, t_f, dt, h_max)


def test_func_indep_t(dt, h_max):
    """renvoie l'erreur entre la vraie solution et la solution
        par différences finies avec la premiere fonction"""
    def u_t_independant(z,t):
        return np.cos(z)

    def f_t_independant(z, t):
        return nu(z) * np.cos(z) + nu_prime(z)*np.sin(z)

    def u_t_indep_simple(z, t):
        return z

    def f_t_indep_simple(z, t):
        return -nu_prime(z)

    def a_t_indep_simple(t):
        return 1

    t_f = 1
    return test_calcul_direct(u_t_independant, f_t_independant, 
            zero, t_f, dt, h_max)

def test_func_explode(dt, h_max):
    """renvoie l'erreur entre la vraie solution et la solution
        par différences finies avec la fonction 
        cos(z) * sin(alpha t)
    """
    discretisation_h= discretisation(h_max)
    first_h = discretisation_h[1] - discretisation_h[0]
    alpha = 1
    def u(z, t):
        return np.exp(z) * np.exp(t)

    def expanded_f(dis, t):
        diff = dis[2:] - 2 * dis[1:-1] + dis[:-2]
        z = dis[1:-1]
        return - np.exp(t + z) * ( nu(z) - 1 + nu_prime(z) \
                + diff*(nu(z) / 3 + nu_seconde(z)/4 + nu_prime(z) / 2))

    def derivee_gauche(t):
        return np.exp(t) * (1 + first_h/2)

    t_f = 3
    return test_calcul_direct_expanded_equation(u, expanded_f, derivee_gauche, t_f, dt, h_max, discretisation_h)


def test_func_simple(dt, h_max):
    """renvoie l'erreur entre la vraie solution et la solution
        par différences finies avec la fonction 
        cos(z) * sin(alpha t)
    """
    alpha = 1
    def u(z, t):
        return np.cos(z) * np.sin(alpha*t)

    def f(z, t):
        return alpha * np.cos(alpha*t) * np.cos(z) + \
                nu_prime(z) * np.sin(z)* np.sin(alpha*t) + \
                nu(z) * np.cos(z) * np.sin(alpha*t)
            

    t_f = 3
    discretisation_h = np.array(np.linspace(0, 30, 1+1/h_max))
    return test_calcul_direct(u, f, zero, t_f, dt, h_max, discretisation_h)

def give_nu_plus_un_demi(discretisation_h, nu):
    return nu((discretisation_h[:-1] + discretisation_h[1:])/2)

def test_calcul_direct_expanded_equation(u, f, a, t_f, dt, h_max, discretisation_h=None):
    """ u : fonction solution
    f : second membre
    a: (du/dz (0) )(t)
    """
    borne_z = 1
    if discretisation_h is None:
        discretisation_h= discretisation(h_max)
        # discretisation non uniforme

    def b(t):
        return u(borne_z, t)


    nu_1_2 = give_nu_plus_un_demi(discretisation_h, nu)

    u0 = u(discretisation_h, 0)
    K = calculer_K(discretisation_h, nu_1_2)
    M = calculer_M(K.shape[0] - 1)
    dis = discretisation_h#[1:-1]
    all_f = []
    for t in np.linspace(dt, t_f, t_f/dt):
        all_f.append(np.concatenate(([a(t)], f(dis,t), [b(t)])))

    hat_u = res_direct_tridiagonal(K, M, u0, all_f, dt)[-1]
    # calcul d'erreur quadratique:
    uf = u(discretisation_h, t_f)
    return max(abs(uf - hat_u))


def test_calcul_direct(u, f, a, t_f, dt, h_max, discretisation_h=None):
    """ u : fonction solution
    f : second membre
    a: (du/dz (0) )(t)
    """
    borne_z = 1
    discretisation_h = np.array(np.linspace(0, borne_z, 1+borne_z/h_max))
    if discretisation_h is None:
        discretisation_h= discretisation(h_max)
        # discretisation non uniforme

    def b(t):
        return u(borne_z, t)


    nu_1_2 = give_nu_plus_un_demi(discretisation_h, nu)

    u0 = u(discretisation_h, 0)
    K = calculer_K(discretisation_h, nu_1_2)
    M = calculer_M(K.shape[0] - 1)
    dis = discretisation_h[1:-1]
    all_f = []
    for t in np.linspace(dt, t_f, t_f/dt):
        all_f.append(np.concatenate(([a(t)], f(dis,t), [b(t)])))

    hat_u = res_direct_tridiagonal(K, M, u0, all_f, dt)[-1]

    """
    hat_u = res_direct(u0, f, discretisation_h, 
                    dt, t_f, 
                    nu_1_2,
                    a, b)
    """
    # calcul d'erreur quadratique:
    uf = u(discretisation_h, t_f)
    return max(abs(uf - hat_u))

if __name__ == "__main__":
    dt = 0.1
    h = 0.1
    """
    print("test avec df/dt = 0 : ")
    for i in range(7):
        print(test_func_indep_t(dt, 10**(-1-i/2)), "  :  10^(-", i/2+1, ")")
    print()
    print("test avec df/dz = 0 : ")
    for i in range(7):
        print(test_func_indep_z(10**(-1-i/2), h), "  :  10^(-", i/2+1, ")")
    print()
    """
    error = []
    step_time_h = []
    print("test avec u = cosz sint : ")
    dt = 0.01
    try:
        for i in range(1,80):
            step = 10**(-i/20)
            ret = test_func_explode(step*step,step)
            print(ret, "  :  10^(-", i/20+1, ")")
            error.append(ret)
            step_time_h.append(1/step)
    except:
        pass

    import matplotlib.pyplot as plt
    plt.plot(step_time_h, error, 'r+')
    plt.title("Résolution de l'équation équivalente")
    plt.xlabel("1/sqrt(dt) = 1/h_max")
    plt.ylabel("Erreur")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
