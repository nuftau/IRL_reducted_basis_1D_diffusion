#!/usr/bin/python3
import scipy.linalg as la
""" ce module est là pour comparer les performances 
du calcul avec base réduite, et sans"""
import matplotlib.pyplot as plt
import numpy as np
from export_img import export_img
from time import perf_counter

nb_resolutions= 10000

with open('matrices.data', 'r+b') as outfile:
    dict_save_arrays = np.load(outfile)['arr_0'].ravel((-1,))[0]

tridiag_for_scipy = dict_save_arrays['tridiag_for_scipy']
seconde_partie_sec_membre = dict_save_arrays['seconde_partie_sec_membre' ]

offset = perf_counter()
for _ in range(nb_resolutions):
    la.solve_banded((1, 1), tridiag_for_scipy, seconde_partie_sec_membre)

times_calcul_u = (perf_counter() - offset) / (nb_resolutions)
print("temps utilisé : ", perf_counter() - offset)
times_calcul_alpha = []
imax = 0
# ---------- CALCUL PERFORMANCE : MATRICE PLEINE DE TAILLE i
try:
    for i in range(1,30):
        imax = i+1
        print("i =",i)
        Phi_T_M_Phi = dict_save_arrays['Phi_T_M_Phi' + str(i) ]
        sec_membre = dict_save_arrays['sec_membre' + str(i) ]
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
