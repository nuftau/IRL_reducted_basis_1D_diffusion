import numpy as np
from res_direct import res_direct

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

def test_func_simple(dt, h_max):
    """renvoie l'erreur entre la vraie solution et la solution
        par différences finies avec la fonction 
        1000*sin(t)(sin(piz) - sin(2piz))"""
    alpha = 0.1
    def u(z, t):
        return np.cos(z) * np.sin(alpha*t)

    def f(z, t):
        return alpha * np.cos(alpha*t) * np.cos(z) + \
                nu_prime(z) * np.sin(z)* np.sin(alpha*t) + \
                nu(z) * np.cos(z) * np.sin(alpha*t)
            

    t_f = 1
    return test_calcul_direct(u, f, zero, t_f, dt, h_max)


def test_calcul_direct(u, f, a, t_f, dt, h_max):
    """ u : fonction solution
    f : second membre
    a: (du/dz (0) )(t)
    """
    def b(t):
        return u(1, t)
    # discretisation_h = np.array([h for h in np.linspace(0, 1, 1+1/h_max)])
    # ici on a h = h_max, la discretisation est uniforme

    discretisation_h= discretisation(h_max) # discretisation non uniforme

    u0 = u(discretisation_h, 0)
    n_max = t_f
    hat_u = res_direct(u0, f, discretisation_h, 
                    dt, t_f, 
                    nu(discretisation_h),
                    nu_prime(discretisation_h),
                    a, b)

    # calcul d'erreur quadratique:
    uf = u(discretisation_h, t_f)
    return max(abs(uf - hat_u))

dt = 0.1
h = 0.1
print("test avec df/dt = 0 : ")
for i in range(7):
    print(test_func_indep_t(dt, 10**(-1-i/2)), "  :  10^(-", i+1, ")")
print()
print("test avec df/dz = 0 : ")
for i in range(7):
    print(test_func_indep_z(10**(-1-i/2), h), "  :  10^(-", i+1, ")")
print()
print("test avec u = cosz sint : ")
for i in range(7):
    print(test_func_simple(10**(-1-i/2), 10**(-1-i/2)), "  :  10^(-", i+1, ")")
