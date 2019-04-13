#!/usr/bin/python3
import scipy.linalg as la
""" ce module est là pour comparer les performances 
du calcul avec base réduite, et sans"""
from itertools import repeat, cycle
import concurrent.futures
import matplotlib.pyplot as plt
import math
from netCDF4 import Dataset
from calcul_lagrangien import calcul_lagrangien,calcul_gradient, calcul_error, call_func_with_computed_data, calcul_forall_k, calcul_alpha
from res_direct import calculer_K, calculer_M, res_direct_tridiagonal
import numpy as np
from export_img import export_img
from modred.pod import compute_POD_matrices_direct_method
from time import perf_counter

rootgrp = Dataset("donnees.nc", "r+", format="NETCDF4")
"""
   variables(dimensions): float32 kz(time_counter,presnivs,lat,lon), float32 lat(lat), float32 lon(lon), float32 presnivs(presnivs), float32 temp(time_counter,presnivs,lat,lon), float64 time_counter(time_counter), float32 zhalf(time_counter,presnivs,lat,lon)
       groups: 

"""
t=0
dt = 600 # la base de donnee a des donnees toutes les 10min
# qui donne un peu des donnees
ensemble_test = []
nb_h = len(rootgrp["zhalf"][0,:,0,0])
# ------- Creation du tuple de donnees pour comparer 
# les perfs de calcul de alpha et de u
tk = 0
all_f = []
t = 0
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
M = calculer_M(len(u0) - 1)

vecs = rootgrp["temp"][:,:,0,0].T
times_calcul_alpha = []
times_calcul_u = []
nb_tests = 3




# ---------- CALCUL PERFORMANCE : MATRICE TRIDIAGONALE
nb_resolutions = 500
u = [u0]
M_sur_dt = M/dt
K_plus_M_sur_dt = K + M_sur_dt
tridiag_for_scipy = np.array((
    np.concatenate(([0], np.diag(K_plus_M_sur_dt, k=1))),
    np.diag(K_plus_M_sur_dt,k=0),
    np.concatenate((np.diag(K_plus_M_sur_dt,k=-1), [0]))))
seconde_partie_sec_membre = all_f[0] + M_sur_dt@u[-1]

dict_save_arrays = {}
dict_save_arrays['tridiag_for_scipy'] = tridiag_for_scipy
dict_save_arrays['seconde_partie_sec_membre' ] = \
        seconde_partie_sec_membre

offset = perf_counter()
for _ in range(nb_resolutions):
    la.solve_banded((1, 1), tridiag_for_scipy, seconde_partie_sec_membre)

times_calcul_u = (perf_counter() - offset) / (nb_resolutions)
print("temps utilisé : ", perf_counter() - offset)

imax = 0
# ---------- CALCUL PERFORMANCE : MATRICE PLEINE DE TAILLE i
try:
    for i in range(1,30):
        imax = i+1
        print("i =",i)
        Phi = compute_POD_matrices_direct_method(vecs, range(i))[0]
        Phi = np.reshape(Phi, (-1, i))
        # on prend un Phi "suffisamment bon" = Phi de POD
        # pour avoir un cas non pathologique 
        # (non inversibilite de Phi.T M Phi  par ex.)

        Phi_T_M_Phi = Phi.T@M@Phi
        sec_membre = np.reshape(Phi.T@M@u0, (Phi_T_M_Phi.shape[0], 1))
        dict_save_arrays['Phi_T_M_Phi' + str(i) ] = \
                Phi_T_M_Phi
        dict_save_arrays['sec_membre' + str(i) ] = \
                sec_membre

        # we print the tuple
        offset = perf_counter()
        for _ in range(nb_resolutions):
            np.linalg.solve(Phi_T_M_Phi, sec_membre)
        times_calcul_alpha.append((perf_counter() - offset)/(nb_resolutions))
        print("temps utilisé : ", perf_counter() - offset)
except:
    pass

plt.plot(list(range(1, imax)), times_calcul_alpha, 'r')
plt.plot(list(range(1, imax)), [times_calcul_u] * (imax - 1), 'v')
export_img("nombre de vecteurs dans la base réduite", "temps moyen de calcul d'un pas de temps", "figures_optimisation/performances_test_fixe.png")
with open('matrices.data', 'w+b') as outfile:
    np.savez(outfile, dict_save_arrays)
rootgrp.close()
