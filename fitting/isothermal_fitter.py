# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU GPL v2 or later.


"""
    
example_fit_data
----------------

This example demonstrates BurnMan's functionality to fit thermoelastic data to
both 2nd and 3rd orders using the EoS of the user's choice at 300 K. User's
must create a file with :math:`P, T` and :math:`V_s`. See input_minphys/ for example input
files.

requires:
- compute seismic velocities

teaches:
- averaging

"""
from __future__ import absolute_import
from __future__ import print_function
import os, sys, numpy as np, matplotlib.pyplot as plt
import scipy.optimize as opt

#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman

if __name__ == "__main__":


    def calc_volume(V_0, K_0, K_prime, mineral, pressures):

        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = K_prime

        Volumes = np.empty_like(pressures)
        for i in range(len(pressures)):
            mineral.set_state(pressures[i], 300.0) # set state with dummy temperature
            Volumes[i] = mineral.V

        return Volumes

    def error(guess, test_mineral, pressures, obs_V,V_err):
        V = calc_volume(guess[0], guess[1], guess[2], test_mineral, pressures)

        V_l2 = [(1./(pow(V_err[i],2.))) * (V[i] - obs_V[i])*(V[i] - obs_V[i]) for i in range(len(obs_V)) ]
        l2_error = sum(V_l2)

        return l2_error

    def calculate_errors(obs_V,obs_P,V_0,K_0,K_prime):

        lin_P = []
        lin_V = []
        for i in range(len(obs_P)):
            if obs_P[i] > 1.e9:
                lin_P.append(obs_P[i])
                lin_V.append(obs_V[i])
        f = np.empty_like(lin_V)
        norm_p = np.empty_like(lin_P)
        for i in range(len(lin_V)):
                f[i] = 0.5*(pow(lin_V[i]/V_0,-2./3.)-1.)
                norm_p[i] = lin_P[i]/(3.*f[i]*pow(1.+(2.*f[i]),5./2.))

        slope = 3.*K_0*(K_prime-4.)/2.
        y_int = K_0
        n = float(len(obs_V))

        in_sqrt_sum = 0.
        sum_f_squared=0.
        sum_f = 0.

        for i in range(len(f)):
            in_sqrt_sum += ((norm_p[i] - (slope*f[i]) - y_int)*(norm_p[i] - (slope*f[i]) - y_int))
            sum_f_squared += pow(f[i],2.)
            sum_f += f[i]

        S = pow(in_sqrt_sum/(n-2.),0.5)

        slope_err = S*pow((n/((n*sum_f_squared)-pow(sum_f,2.))),0.5)
        K_0_err = S*pow((sum_f_squared/((n*sum_f_squared)-pow(sum_f,2.))),0.5)
        K_prime_err = (1./K_0)*(((2./3.)*slope_err)-((K_prime-4.)*K_0_err))

        return 0,K_0_err,K_prime_err

    input_data = burnman.tools.read_table_tabs("input_minphys/Wendy_data.txt")
    obs_pressures = input_data[:,0]*1.e9
    obs_T = input_data[:,1]
    obs_V = input_data[:,2]*6.02e-7
    V_err = input_data[:,3]*6.02e-7

    pressures = np.linspace(min(obs_pressures),max(obs_pressures), 300.)
    #make the mineral to fit
    guess = [min(obs_V),200.e9, 4.5]
    dummy_mineral = burnman.Mineral()
    dummy_mineral.params['V_0'] = 24.45e-6
    dummy_mineral.params['K_0'] = 281.e9
    dummy_mineral.params['Kprime_0'] = 4.1
    dummy_mineral.params['molar_mass'] = .10

    EOS_to_fit = 'vinet'

    #first, do the second-order fit
    dummy_mineral.set_method(EOS_to_fit)
    func = lambda x : error( x, dummy_mineral, obs_pressures, obs_V,V_err)
    sol = opt.fmin(func, guess)


    class own_material(burnman.Mineral):
        def __init__(self):
            self.params = {
                'name': 'myownmineral',
                'equation_of_state': EOS_to_fit,
                'V_0': sol[0], #Molar volume [m^3/(mole molecules)]
                #at room pressure/temperature
                'K_0': sol[1], #Reference bulk modulus [Pa]
                #at room pressure/temperature
                'Kprime_0': sol[2], #pressure derivative of bulk modulus
            }
            burnman.Mineral.__init__(self)

    rock = own_material()
    rock.set_method(EOS_to_fit)
    temperature = [300. for i in pressures]

    best_fit_volume,K_s_out = (rock.evaluate(['V','K_S'],pressures,temperature))
    error_V,error_K,error_K_prime = calculate_errors(obs_V,obs_pressures,sol[0],sol[1],sol[2])

    legend_label = ("V0 = "+ "%.2f"%((sol[0]*(1.e30)/(6.02e23)))+" +/- "+"%.2f"%(error_V*(1.e30)/(6.02e23)) + " Ang^3"'\n'+ \
                    "K0 = "+"%.2f"%(sol[1]/1.e9)+" +/- "+"%.2f"%(error_K/1.e9)+" GPa "+'\n'\
                    +"K' = "+ "%.2f"%(sol[2])+" +/- "+"%.2f"%(error_K_prime))
    print(legend_label)


    #f,norm_pres = calculate_strain_P()
    figure = plt.figure( figsize = (6,12) )

    ax1 = plt.subplot2grid( (6,12), (0, 0),rowspan=3,colspan=12)
    ax2 = plt.subplot2grid((6,12), (3, 0),rowspan=3,colspan=12)
    ax1.plot(pressures/1.e9,best_fit_volume*(1.e30)/(6.02e23),color='r', linestyle='-', linewidth=2, label = legend_label)
    #ax1.errorbar(obs_pressures/1.e9, obs_V*(1.e30)/(6.02e23),yerr=V_err, fmt=None,color='k')
    ax1.scatter(obs_pressures/1.e9, obs_V*(1.e30)/(6.02e23))
    ax1.set_xlim([pressures[0]/1e9,pressures[-1]/1.e9])
    ax1.set_ylabel("volume (Ang^3)")
    ax1.legend(loc = "upper right",prop={'size':16},frameon=False)
    ax2.plot(pressures/1.e9,K_s_out/1.e9,color='b', linestyle='-', linewidth=2)
    ax2.set_xlim([pressures[0]/1e9,pressures[-1]/1.e9])
    ax2.set_ylabel("K_s (GPa)")
    ax2.set_xlabel("Pressure (GPa)")

    plt.show()
