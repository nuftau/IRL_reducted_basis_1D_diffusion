from itertools import repeat
import concurrent.futures
import matplotlib.pyplot as plt
import math
from netCDF4 import Dataset
from calcul_lagrangien import calcul_lagrangien,calcul_gradient, calcul_error, call_func_with_computed_data, calcul_forall_k
from res_direct import calculer_K
import numpy as np
import scipy.optimize as opti
from export_img import export_img

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

index_drawing = 0
# -------------- Creation de l'ensemble test ------------
for tk in range(int(len(rootgrp["zhalf"][:,0,0,0])/2), 
        len(rootgrp["zhalf"][:,0,0,0]), 20):
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
    
    """
    if (tk - int(len(rootgrp["zhalf"][:,0,0,0])/2)) == 0:
        #plt.plot(nu_1_2, discretisation_h[1:], 'r')
        plt.eventplot(discretisation_h, orientation='vertical')
        index_drawing+= 1
    """
    
    # (normalement faudrait prendre nu en les points intermédiaires
    K = calculer_K(discretisation_h, nu_1_2)
    ensemble_test += [(u0, K, all_f, dt)]


#plt.yscale("log")

#export_img("...", "hauteur(m)", "figures_optimisation/discretisation_h.png")


dh = discretisation_h[1:] - discretisation_h[:-1]

# ------------ Calcul de la base POD pour Phi
from modred.pod import compute_POD_matrices_direct_method
vecs = rootgrp["temp"][:,:,0,0].T
erreur_max = []
erreur_max_residuel = []
Phi = None

try:
   for i in range(1,8,2):
        print("i =",i)
        Phi = compute_POD_matrices_direct_method(vecs, range(i))[0]
        Phi = np.ravel(Phi)
        res_opti = opti.minimize(calcul_lagrangien, Phi, jac=calcul_gradient, args=(nb_h, ensemble_apprentissage), method='BFGS', options={'disp': True})

        #print("gradient : ", calcul_gradient(res_opti.x, int(len(res_opti.x)/i), ensemble_apprentissage, True))
        plt.clf()
        color = 'r+'
        Phi = np.reshape(res_opti.x, (-1, i))
        #print(np.linalg.norm(Phi))
        def erreur_norme_inf(Phi, u_k, f_k, alpha_k, *param):
            return max([max(abs(u_n - Phi @ alpha_n)) \
                    for u_n, alpha_n in zip(u_k, alpha_k)])

        def erreur_norme_L2(Phi, u_k, f_k, alpha_k, *param):
            # on veut "intégrer" f = (u - phi@alpha)**2:
            # formule des trapezes : pour n=0...N-2,
            # /!\ ce n'est pas cette erreur qu'on minimise sur Phi !
            # int += f * dh[n] + (f[n+1] - f[n]) * dh[n] / 2
            # donc int += (f[n+1] + f[n]) * dh / 2
            # le tout divisé (à la fin) par la hauteur totale
            # on peut integrer directement la somme des f,
            # c'est ce qu'on fait ici

            f_k = [(u_n - Phi @ alpha_n)**2 for u_n, alpha_n \
                    in zip(u_k, alpha_k)]
            integrales = sum([np.sqrt(
                sum((f[1:] + f[:-1]) * dh) / 
                (2 * discretisation_h[-1])) for f in f_k])
            return integrales / len(u_k)

            """
            
            f = sum([(u_n - Phi @ alpha_n)**2 for u_n, alpha_n \
                    in zip(u_k, alpha_k)])
            integrale = sum((f[1:] + f[:-1]) * dh) / (2 * discretisation_h[-1])
            return np.sqrt(integrale) / len(u_k)
            """


        # --- calcul du pire echantillon ------
        def erreurs(ensemble_echantillons, func_erreur):
            x0 = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                x0 += executor.map(call_func_with_computed_data, 
                    zip(ensemble_echantillons,
                        repeat(func_erreur), repeat(Phi)))
            return x0
        erreurs_apprentissage = np.array(erreurs(ensemble_apprentissage, erreur_norme_L2))
        worst_elem_apprentissage = np.argmax(erreurs_apprentissage)
        erreur_max_residuel += [erreurs_apprentissage[worst_elem_apprentissage]]

        erreurs_test = np.array(erreurs(ensemble_test, erreur_norme_L2))



        worst_elem_test = np.argmax(erreurs_test)
        erreur_max += [erreurs_test[worst_elem_test]]
        print("err max : ", erreurs_test[worst_elem_test])

        def draw_one_profile(Phi, u, all_f, alpha, *param):
            #on va dessiner le PIRE profile

            erreurs = np.array([max(abs(u_n - Phi @ alpha_n)) for u_n, alpha_n in zip(u, alpha)])
            indice = np.argmax(erreurs)

            plt.plot(Phi @ alpha[indice], discretisation_h, 'r')
            plt.plot(u[indice], discretisation_h, 'g')

        call_func_with_computed_data((ensemble_test[worst_elem_test], draw_one_profile, Phi))
        export_img("rouge:approché, vert : exact(m)", "hauteur", "figures_optimisation/bons_erreur_L2_Phi"+str(i)+".png")
        # ---- creation des histogrammes ---
        """
        erreurs_k_L2 = []
        for k in range(len(ensemble_test)):
            def plot_errors(Phi, u, f, alpha, *param):
                # on commence par calculer les erreurs L2 et inf
                err_L2 = erreur_norme_L2(Phi, u, f, alpha, *param)
                err_inf= erreur_norme_inf(Phi, u, f, alpha, *param)
                #plt.plot([k], [err_L2], 'b+')
                #plt.plot([k], [err_inf], 'r+')
                return err_L2

            erreurs_k_L2 += [call_func_with_computed_data((ensemble_test[k], plot_errors, Phi))]
        plt.hist(erreurs_k_L2, 25, range=[0, 8], facecolor='green', alpha=0.75)
        plt.ylim([0,20])
        export_img("erreur (°K)", "nombre d'echantillons", "figures_optimisation/bons_histogramme_erreurs_L2_Phi"+str(i)+".png")

        #print("erreur (norme inf.): ", erreur_max[-1])
        #print("erreur residuelle (norme inf.): ", erreur_max_residuel[-1])
	"""
        
except:
    raise
    pass

print(erreur_max)

rootgrp.close()
