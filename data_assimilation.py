import math
from netCDF4 import Dataset
from calcul_lagrangien import calcul_lagrangien,calcul_gradient, calcul_error
from res_direct import calculer_K
import numpy as np
import scipy.optimize as opti

rootgrp = Dataset("donnees.nc", "r+", format="NETCDF4")
"""
   variables(dimensions): float32 kz(time_counter,presnivs,lat,lon), float32 lat(lat), float32 lon(lon), float32 presnivs(presnivs), float32 temp(time_counter,presnivs,lat,lon), float64 time_counter(time_counter), float32 zhalf(time_counter,presnivs,lat,lon)
       groups: 

"""
t=0
dt = 600 # la base de donnee a des donnees toutes les 10min
n = 7 # arbitraire, faut un truc calculable mais 
# qui donne un peu des donnees
ensemble_apprentissage = []
ensemble_test = []
nb_h = len(rootgrp["zhalf"][0,:,0,0])
# ------- Creation de l'ensemble d'apprentissage---
for tk in range(0,int(len(rootgrp["zhalf"][:,0,0,0])/2), 50):
    all_f = []
    for t in range(tk, tk+n):
        #a est la valeur de du/dt
        a = rootgrp["temp"][t,1,0,0] - rootgrp["temp"][t,0,0,0]
        a /= rootgrp["zhalf"][t,1,0,0] - rootgrp["zhalf"][t,0,0,0]
        # b est la valeur de u en h_max
        b = rootgrp["temp"][t,-1,0,0]
        all_f += [np.concatenate(([a], [0] * (nb_h-2), [b]))] 
    discretisation_h = rootgrp["zhalf"][tk,:,0,0]
    u0 = rootgrp["temp"][tk,:,0,0]
    nu_1_2 = rootgrp["kz"][tk,1:,0,0] # on enlève la premiere valeur
    # (normalement faudrait prendre nu en les points intermédiaires
    K = calculer_K(discretisation_h, nu_1_2)
    ensemble_apprentissage += [(u0, K, all_f, dt)]

# -------------- Creation de l'ensemble test ------------
for tk in range(int(len(rootgrp["zhalf"][:,0,0,0])/2), 
        len(rootgrp["zhalf"][:,0,0,0]), 50):
    all_f = []
    for t in range(tk, tk+n):
        #a est la valeur de du/dt
        a = rootgrp["temp"][t,1,0,0] - rootgrp["temp"][t,0,0,0]
        a /= rootgrp["zhalf"][t,1,0,0] - rootgrp["zhalf"][t,0,0,0]
        # b est la valeur de u en h_max
        b = rootgrp["temp"][t,-1,0,0]
        all_f += [np.concatenate(([a], [0] * (nb_h-2), [b]))] 
    discretisation_h = rootgrp["zhalf"][tk,:,0,0]
    u0 = rootgrp["temp"][tk,:,0,0]
    nu_1_2 = rootgrp["kz"][tk,1:,0,0] # on enlève la premiere valeur
    # (normalement faudrait prendre nu en les points intermédiaires
    K = calculer_K(discretisation_h, nu_1_2)
    ensemble_test += [(u0, K, all_f, dt)]


Phi_arbitraire = np.reshape(discretisation_h[-1]-discretisation_h, (-1, 1))
Phi = None
try:
    for i in range(2,40):
        if Phi is not None:
            Phi = np.hstack((Phi, Phi_arbitraire))
        else:
            Phi = np.copy(Phi_arbitraire)
        Phi = np.ravel(Phi)
        res_opti = opti.minimize(calcul_lagrangien, Phi, jac=calcul_gradient, args=(nb_h, ensemble_apprentissage), method='BFGS')
        # print(res_opti)
        print("erreur moyenne (en norme inf.) avec l'ensemble d'apprentissage : Phi est de taille", i-1)
        print(calcul_error(Phi, nb_h, ensemble_apprentissage)/len(ensemble_apprentissage))
        print("racine de la moyenne de l'err. quadratique")
        err = calcul_lagrangien(Phi, nb_h, ensemble_apprentissage)
        err /= len(ensemble_apprentissage) * 79 * n
        print(math.sqrt(err))

        print("sur l'ensemble test, on fait les memes calcul : ")
        print(calcul_error(Phi, nb_h, ensemble_test)/len(ensemble_test))
        print("racine de la moyenne de l'err. quadratique")
        err = calcul_lagrangien(Phi, nb_h, ensemble_test)
        err /= len(ensemble_test) * 79 * n

        print(math.sqrt(err))
        # print("erreur (en norme inf.) avec l'ensemble test :")
        print("on va passer à la phase", i)
        Phi = np.reshape(res_opti.x, (nb_h, -1))
except:
    print("exception levée.")

#print(Phi)

rootgrp.close()
