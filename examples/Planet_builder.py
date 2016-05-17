# BurnMan - a lower mantle toolkit
# Copyright (C) 2015, Heister, T., Unterborn, C., Rose, I. Cottaar, S., and Myhill, R.
# Released under GPL v2 or later.


'''
example_build_planet
--------------------

For Earth we have well-constrained one-dimensional density models.  This allows us to
calculate pressure as a funcion of depth.  Furthermore, petrologic data and assumptions
regarding the convective state of the planet allow us to estimate the temperature.

For planets other than Earth we have much less information, and in particular we
know almost nothing about the pressure and temperature in the interior.  Instead, we tend
to have measurements of things like mass, radius, and moment-of-inertia.  We would like
to be able to make a model of the planet's interior that is consistent with those
measurements.

However, there is a difficulty with this.  In order to know the density of the planetary
material, we need to know the pressure and temperature.  In order to know the pressure,
we need to know the gravity profile.  And in order to the the gravity profile, we need
to know the density.  This is a nonlinear problem which requires us to iterate to find
a self-consistent solution.

For self consistency between upper and lower mantle currently this code operates only between
1 < Mg/Si < 2.


*Uses:*

* :doc:`mineral_database`
* :class:`burnman.composite.Composite`
* :func:`burnman.main.velocities_from_rock`
'''

import os, sys,time
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import burnman
import burnman.tools as tools
import burnman.minerals as minerals
import burnman.tools

if __name__ == "__main__":
    output_figure_name = "../../Dropbox/compres_2015/data/profile_Fe_XXX"
    #gravitational constant
    G = 6.67e-11
    #Physical Properties of Planet
    Radius_scale = 1. #Radius of planet in Earth radii
    CMB_scale = .495 #Fraction of radius of planet that is core
    icb_scale = 0. #Fraction of core that is solid-Fe
    #Mineral Composition

    #Core
    #Set the composition of the core. Can include liquid Fe outer core, however only one EOS is available for liquid Fe
    #(minerals.other.liquid_Fe(). Solid Fe EOS's available: minerals.other.Liquid_Fe_Anderson() and minerals.other.Fe_Dewaele()
    Inner_core = burnman.Composite([minerals.other.Fe_Dewaele()],[1.])
    Outer_core = burnman.Composite([minerals.other.Liquid_Fe_Anderson()],[1.])



    #Mantle

    #Lower Mantle. Currently assumes mantle is made only of Mg-Fe-brigmanite and
    #magnesiowuestite.
    #Mole fraction of brigmanite in lower mantle. Mole fraction of magnesiowuestite then is simply 1-mol_per_pv
    mol_per_br = 0.89

    #molar fraction of Fe in brigmanite and magnesiowuestite
    amt_fe_br = 0.
    amt_fe_mw = 0.505

    #Upper Mantle. Currently assumes mantle is made only of Mg-Fe-Olivine and Mg-Fe-Enstatite
    #If you'd like to run with an upper mantle, simply set upper_mantle to True. This is the default setting.
    #Currently the Mg/Si ratio of the lower mantle is adopted in the upper mantle. This means that the amount
    #of Olivine is simply frac_olivine = (1/mol_per_br)-1, with frac_enstatite = -(1/mol_per_br)+2
    #If set to true, the transition from LM to UM is set at 25 GPa rather than a radial transition.
    #Also need to set the amount of Fe in both olivine and enstatite
    upper_mantle = True

    #currently both Mg/Si and Fe/Si cannot be the same between upper and lower mantle if Fe is included. Thus, these are
    #inputs for the user. The ratio of olivine to enstatite will still be determined as if the system is Fe free.
    #fay = Fayalite, ferro = ferrosillite
    amt_fay = 0.
    amt_ferro = 0.

    #The builder takes the number of depth slices which will
    #be used for the calculation.  More slices will generate more
    #accurate profiles, but it will take longer.
    n_slices = 5000

    #The n_iterations parameter sets how many times to iterate over density, pressure,
    #and gravity.  Empirically, seven-ish iterations is adequate.
    n_iterations = 7


    # Create a class for Planet that will do the heavy lifting.
    class Planet(object):

        def __init__(self, n_slices):

            self.trans_zone_pressure = 25.e9
            self.outer_radius = (Radius_scale*6371.e3)  # Outer radius of the planet
            self.cmb = CMB_scale*self.outer_radius #Core mantle boundary radius
            self.icb = icb_scale*self.cmb #Inner Core radius

            self.radii = np.linspace(0.e3, self.outer_radius, n_slices) # Radius list
            self.pressures = np.linspace(350.0e9, 0.0, n_slices) # initial guess at pressure profile

            #creates the geotherm, starting with just 300 K isothermal
            self.temperatures =[300. for i in self.pressures]


            brigmanite_solidsolution=minerals.SLB_2011.mg_fe_perovskite()
            periclase_solidsolution = minerals.SLB_2011.ferropericlase()

            # Set molar_fraction of mg_perovskite, fe_perovskite and al_perovskite
            brigmanite_solidsolution.set_composition([1.-amt_fe_br,amt_fe_br,0.])

           # Set molar_fraction of periclase and wuestite
            periclase_solidsolution.set_composition([1.-amt_fe_mw,amt_fe_mw])

            #create the lower mantle instance
            self.lower_mantle = burnman.Composite([brigmanite_solidsolution, periclase_solidsolution],[mol_per_br,1.-mol_per_br])

            #to conserve Mg/Si between upper and lower mantle we derive the amount of olivine from the amount of brigmanite

            self.mol_frac_ol = (1./mol_per_br)-1.
            self.mol_frac_forst = (1.-amt_fay)
            self.upper_mantle = burnman.Composite([minerals.SLB_2011.forsterite(),minerals.SLB_2011.fayalite(),\
                                                     minerals.SLB_2011.enstatite(),minerals.SLB_2011.ferrosilite()],\
                                                  [self.mol_frac_ol*self.mol_frac_forst, self.mol_frac_ol*amt_fay,\
                                                    (1.-self.mol_frac_ol)*(1.-amt_ferro),(1-self.mol_frac_ol)*amt_ferro])
            #initialize the core
            self.inner_core = Inner_core
            self.outer_core = Outer_core
            self.averaging_scheme = burnman.averaging_schemes.VoigtReussHill()


            #self.temperatures = temperature[::-1]
        def generate_profiles( self, n_iterations ):
            #Generate the density, gravity, pressure, vphi, and vs profiles for the planet.


            for i in range(n_iterations):
                print "on iteration #", str(i+1)+"/"+str(n_iterations)
                self.densities, self.bulk_sound_speed, self.shear_velocity = self._evaluate_eos(self.pressures, self.temperatures, self.radii,self.averaging_scheme)
                self.gravity = self._compute_gravity(self.densities, self.radii)
                self.pressures = self._compute_pressure(self.densities, self.gravity, self.radii)

            self.mass = self._compute_mass(self.densities, self.radii)

            self.moment_of_inertia = self._compute_moment_of_inertia(self.densities, self.radii)
            self.moment_of_inertia_factor = self.moment_of_inertia / self.mass / self.outer_radius / self.outer_radius

        def _evaluate_eos(self, pressures, temperatures, radii,averaging_scheme):
            #Evaluates the equation of state for each radius slice of the model.
            #Returns density, bulk sound speed, and shear speed.

            rho = np.empty_like(radii)
            bulk_sound_speed = np.empty_like(radii)
            shear_velocity = np.empty_like(radii)

            for i in range(len(radii)):
                if pressures[i]<self.trans_zone_pressure and radii[i] > self.cmb and upper_mantle == True:
                    density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.upper_mantle, np.array([pressures[i]]), np.array([temperatures[i]]),self.averaging_scheme)
                else:
                    if radii[i] > self.cmb:
                        density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.lower_mantle, np.array([pressures[i]]), np.array([temperatures[i]]),self.averaging_scheme)
                    else:
                        if radii[i] > self.icb:
                            density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.outer_core, np.array([pressures[i]]), np.array([temperatures[i]]),self.averaging_scheme)
                        else:
                            density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.inner_core, np.array([pressures[i]]), np.array([temperatures[i]]),self.averaging_scheme)

                rho[i] = density
                bulk_sound_speed[i] = vphi
                shear_velocity[i] = vs

            return rho, bulk_sound_speed, shear_velocity


        def _compute_gravity(self, density, radii):
            #Calculate the gravity of the planet, based on a density profile.  This integrates
            #Poisson's equation in radius, under the assumption that the planet is laterally
            #homogeneous.

            #Create a spline fit of density as a function of radius
            rhofunc = UnivariateSpline(radii, density )

            #Numerically integrate Poisson's equation
            poisson = lambda p, x : 4.0 * np.pi * G * rhofunc(x) * x * x
            grav = np.ravel(odeint( poisson, 0.0, radii ))
            grav[1:] = grav[1:]/radii[1:]/radii[1:]
            grav[0] = 0.0 #Set it to zero a the center, since radius = 0 there we cannot divide by r^2
            return grav

        def _compute_pressure(self, density, gravity, radii):
            #Calculate the pressure profile based on density and gravity.  This integrates
            #the equation for hydrostatic equilibrium  P = rho g z.

            #convert radii to depths
            depth = radii[-1]-radii

            #Make a spline fit of density as a function of depth
            rhofunc = UnivariateSpline( depth[::-1], density[::-1] )
            #Make a spline fit of gravity as a function of depth
            gfunc = UnivariateSpline( depth[::-1], gravity[::-1] )

            #integrate the hydrostatic equation
            pressure = np.ravel(odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
            return pressure[::-1]

        def _compute_mass( self, density, radii):
            #calculates the mass of the entire planet [kg]
            rhofunc = UnivariateSpline(radii, density )
            mass = quad( lambda r : 4*np.pi*rhofunc(r)*r*r,
                                     radii[0], radii[-1] )[0]
            return mass


        def _compute_moment_of_inertia( self, density, radii):
            #Returns the moment of inertia of the planet [kg m^2]

            rhofunc = UnivariateSpline(radii, density )
            moment = quad( lambda r : 8.0/3.0*np.pi*rhofunc(r)*r*r*r*r,
                                     radii[0], radii[-1] )[0]
            return moment


    def compute_shell_mass( density, radii):
        #returns a mass for a layer or sum of layers smaller than the whole planet
            rhofunc = UnivariateSpline(radii, density )
            mass = quad( lambda r : 4.*np.pi*rhofunc(r)*r*r,
                                     radii[0], radii[-1] )[0]
            return mass
    # Here we actually do the interation.  We make an instance
    # of our Plan.ry planet, then call generate_profiles.
    # Emprically, 300 slices and 5 iterations seem to do
    # a good job of converging on the correct profiles.

    Plan = Planet(n_slices)
    Plan.generate_profiles( n_iterations )

    # These are the actual observables
    # from the model, that is to say,
    # the total mass of the planet and
    # the moment of inertia factor,
    # or C/MR^2


    counter = 0

    #loop to get core radii to calculate total core mass
    for i in Plan.pressures:
        if i <= 25e9 and Plan.radii[counter] >= Plan.cmb:
            trans_radii = Plan.radii[counter]
            break
        counter +=1
    core_rad = []
    core_den=[]
    for i in range(len(Plan.radii)):
        if Plan.radii[i] <= Plan.cmb:
            core_rad.append(Plan.radii[i])
            core_den.append(Plan.densities[i])

    #calculates the core mass, note does this in FRACTION of total planet mass. Needed to calculate Si/Fe
    core_mass_fraction = compute_shell_mass(core_den, core_rad)/Plan.mass

    #Do the same for the Upper Mantle, to calculate its mass. Needed to determine Si/Fe
    UM_radius =[]
    UM_density =[]
    radius_noUM =[]
    density_noUM =[]
    for i in range(len(Plan.pressures)):
        if Plan.pressures[i] <= 25.e9 and Plan.radii[i] > Plan.cmb:
            UM_radius.append(Plan.radii[i])
            UM_density.append(Plan.densities[i])
        elif Plan.pressures[i] > 25.e9 or Plan.radii[i] <= Plan.cmb:
            radius_noUM.append(Plan.radii[i])
            density_noUM.append(Plan.densities[i])
    if upper_mantle == True:
        UM_mass_non = compute_shell_mass(density_noUM,radius_noUM)
        UM_mass_fraction = (Plan.mass - UM_mass_non)/Plan.mass
    else:
        UM_mass=0.
    print
    print "Total mass of the planet (in Earth Masses): ", "%.3f" % (Plan.mass/5.9736e24)      #PRINT MASS
    print
    print "Upper mantle mass fraction", "%.2f" % (100*UM_mass_fraction)
    print
    print "core mass % of total", "%.2f" % (100.*core_mass_fraction)
    print
    print "core radius % of total", "%.2f" % (100.*Plan.cmb/Plan.radii[-1])
    print
    print "Average density = ",str("%.2f" % (Plan.mass/((4./3.)*np.pi*pow(Plan.outer_radius,3.))/1000.))," g/cc"
    print
    tools.Stoich(Plan.mass/5.9736e24,core_mass_fraction,UM_mass_fraction,mol_per_br,amt_fe_br,amt_fe_mw,\
            Plan.mol_frac_ol,Plan.mol_frac_forst,amt_ferro)

    #save a text file
    f = open(output_figure_name+".txt", 'wb')
    f.write("#radius\tpressure\tdensity\tgravity\n")
    data = zip(Plan.radii/6371.e3, Plan.pressures/1.e9,Plan.densities,Plan.gravity)
    np.savetxt(f, data, fmt='%.10e', delimiter='\t')
    f.close()

    #Plot everything up
    import matplotlib.gridspec as gridspec

    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = '\usepackage{relsize}'
    plt.rc('font', family='sanserif')

    #Come up with axes for the final plot
    figure = plt.figure( figsize = (12,10) )
    ax1 = plt.subplot2grid( (5,3) , (0,0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid( (5,3) , (3,0), colspan=3, rowspan=1)
    ax3 = plt.subplot2grid( (5,3) , (4,0), colspan=3, rowspan=1)
         #Plot density, vphi, and vs for the planet.
    ax1.plot( Plan.radii/1.e3, Plan.densities/1.e3, label=r'$\rho$', linewidth=2.,color='k')
    #ax1.plot( Plan.radii/1.e3, Plan.bulk_sound_speed/1.e3, label=r'$V_\phi$', linewidth=2.)
    #ax1.plot( Plan.radii/1.e3, Plan.shear_velocity/1.e3, label=r'$V_S$', linewidth=2.)

    #Also plot a black line for the icb, cmb and upper mantle
    ylimits = [2., (Plan.densities[0]/1e3)+1.]
    ax1.plot( [Plan.icb/1.e3, Plan.icb/1.e3], ylimits, 'k', linewidth = 2.,color='r')
    ax1.plot( [Plan.cmb/1.e3, Plan.cmb/1.e3], ylimits, 'k', linewidth = 2.,color = 'b')
    ax1.plot( [trans_radii/1.e3, trans_radii/1.e3], ylimits, 'k', linewidth = 2.,color='g')

    ax1.set_ylabel("Density (kg/m$^3$)")

    #Make a subplot showing the calculated pressure profile
    ax2.plot( Plan.radii/1.e3, Plan.pressures/1.e9, 'k', linewidth=2.)
    ax2.set_ylabel("Pressure (GPa)")

    #Make a subplot showing the calculated gravity profile
    ax3.plot( Plan.radii/1.e3, Plan.gravity, 'k', linewidth=2.)
    ax3.set_ylabel("Gravity (m/s$^2)$")
    ax3.set_xlabel("Radius (km)")

    #plt.show()






