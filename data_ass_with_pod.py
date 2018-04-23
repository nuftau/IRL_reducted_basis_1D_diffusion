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
n = 10 # arbitraire, faut un truc calculable mais 
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
    # (normalement faudrait prendre nu en les points intermédiaires)
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


# ------------ Calcul de la base POD pour Phi
from modred.pod import compute_POD_matrices_direct_method
vecs = rootgrp["temp"][:,:,0,0].T
# ceci etait le calcul de pod...
erreur_moyenne = []
erreur_moyenne_only_pod = []
Phi = None

try:
r   for i in range(2,80):
        Phi = compute_POD_matrices_direct_method(vecs, range(i))[0]
        Phi = np.ravel(Phi)
        erreur_moyenne_only_pod += [calcul_error(Phi, nb_h, ensemble_apprentissage)/len(ensemble_apprentissage)]
        res_opti = opti.minimize(calcul_lagrangien, Phi, jac=calcul_gradient, args=(nb_h, ensemble_apprentissage), method='BFGS', options={'disp': True})
        erreur_moyenne += [calcul_error(Phi, nb_h, ensemble_apprentissage)/len(ensemble_apprentissage)]
except:
    pass

print(erreur_moyenne)
rootgrp.close()
