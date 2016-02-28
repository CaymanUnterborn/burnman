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
            mineral.set_state(pressures[i], 0.0) # set state with dummy temperature
            Volumes[i] = mineral.V

        return Volumes

    def error(guess, test_mineral, pressures, obs_V):
        V = calc_volume(guess[0], guess[1], guess[2], test_mineral, pressures)

        V_l2 = [ (V[i] - obs_V[i])*(V[i] - obs_V[i]) for i in range(len(obs_V)) ]
        l2_error = sum(V_l2)

        return l2_error


    input_data = burnman.tools.read_table("input_minphys/Dewaele_Fe.txt")
    obs_pressures = input_data[:,0]*1.e9
    obs_T = input_data[:,1]
    obs_V = input_data[:,2]*1.e-6

    pressures = np.linspace(0.e9, 300.e9, 300.)

    #make the mineral to fit
    guess = [24.45e-6,200.e9, 4.0]
    mineral_test = burnman.Mineral()
    mineral_test.params['V_0'] = 24.45e-6
    mineral_test.params['K_0'] = 281.e9
    mineral_test.params['Kprime_0'] = 4.1
    mineral_test.params['molar_mass'] = .10

    EOS_to_fit = 'bm3'

    #first, do the second-order fit
    mineral_test.set_method(EOS_to_fit)
    func = lambda x : error( x, mineral_test, obs_pressures, obs_V)
    sol = opt.fmin(func, guess)

    best_fit_volume = calc_volume(sol[0], sol[1],sol[2], mineral_test, pressures)
    legend_label = ("V0 = "+ "%.2f"%((sol[0]/1.e-6)) + " cc/mol"'\n'+ "K0 = "+"%.2f"%(sol[1]/1.e9)+" GPa "+'\n'+"K' = "+ "%.2f"%(sol[2]))
    print(legend_label)
    plt.plot(pressures/1.e9,best_fit_volume/1.e-6,color='r', linestyle='-', linewidth=2, label = legend_label)
    plt.scatter(obs_pressures/1.e9, obs_V/1.e-6)
    plt.ylabel("volume (cc/mol)")
    plt.xlabel("Pressure (GPa)")
    plt.legend(loc = "lower right",prop={'size':12},frameon=False)
    #plt.savefig("output_figures/example_fit_data.png")
    plt.show()
