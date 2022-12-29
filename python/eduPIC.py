#-------------------------------------------------------------------##
#         eduPIC : educational 1d3v PIC/MCC simulation code         ##
#           version 1.0, release date: March 16, 2021               ##
#                       :) Share & enjoy :)                         ##
#-------------------------------------------------------------------##
# When you use this code, you are required to acknowledge the       ##
# authors by citing the paper:                                      ##
# Z. Donko, A. Derzsi, M. Vass, B. Horvath, S. Wilczek              ##
# B. Hartmann, P. Hartmann:                                         ##
# "eduPIC: an introductory particle based  code for radio-frequency ##
# plasma simulation"                                                ##
# Plasma Sources Science and Technology, vol XXX, pp. XXX (2021)    ##
#-------------------------------------------------------------------##
# Disclaimer: The eduPIC (educational Particle-in-Cell/Monte Carlo  ##
# Collisions simulation code), Copyright (C) 2021                   ##
# Zoltan Donko et al. is free software: you can redistribute it     ##
# and/or modify it under the terms of the GNU General Public License##
# as published by the Free Software Foundation, version 3.          ##
# This program is distributed in the hope that it will be useful,   ##
# but WITHOUT ANY WARRANTY; without even the implied warranty of    ##
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU  ##
# General Public License for more details at                        ##
# https:#www.gnu.org/licenses/gpl-3.0.html.                        ##
#-------------------------------------------------------------------##


import numpy as np
#Importing the BIT generator class to generate various random numbers
from numpy.random import default_rng
from dataclasses import dataclass
import argparse

# constants

PI = np.pi # mathematical constant pi
TWO_PI = 2.0*PI
E_CHARGE = 1.60217662e-19                # electron charge [C]
EV_TO_J        = E_CHARGE                # eV <-> Joule conversion factor
E_MASS         = 9.10938356e-31          # mass of electron [kg]
AR_MASS        = 6.63352090e-26          # mass of argon atom [kg]
MU_ARAR        = AR_MASS / 2.0           # reduced mass of two argon atoms [kg]
K_BOLTZMANN    = 1.38064852e-23          # Boltzmann's constant [J/K]
EPSILON0       = 8.85418781e-12          # permittivity of free space [F/m]

# simulation parameters

N_G            = 400                   # number of grid points
N_T            = 4000                  # time steps within an RF period
FREQUENCY      = 13.56e6               # driving frequency [Hz]
VOLTAGE        = 250.0                 # voltage amplitude [V]
L              = 0.025                 # electrode gap [m]
PRESSURE       = 10.0                  #  gas pressure [Pa]
TEMPERATURE    = 350.0                 # background gas temperature [K]
WEIGHT         = 7.0e4                 # weight of superparticles
ELECTRODE_AREA = 1.0e-4                # (fictive) electrode area [m^2]
N_INIT         = 1000                  # number of initial electrons and ions

# additional (derived) constants

PERIOD         = 1.0 / FREQUENCY                   # RF period length [s]
DT_E           = PERIOD / (N_T)                    # electron time step [s]
N_SUB          = 20        # ions move only in these cycles (subcycling)
DT_I           = N_SUB * DT_E                      # ion time step [s]
DX             = L / (N_G - 1)                     # spatial grid division [m]
INV_DX         = 1.0 / DX              # inverse of spatial grid size [1/m]
GAS_DENSITY    = PRESSURE / (K_BOLTZMANN * TEMPERATURE)    # background gas density [1/m^3]
OMEGA          = TWO_PI * FREQUENCY                # angular frequency [rad/s]

# electron and ion cross sections

N_CS           = 5             # total number of processes / cross sections
E_ELA          = 0             # process identifier: electron/elastic
E_EXC          = 1             # process identifier: electron/excitation
E_ION          = 2             # process identifier: electron/ionization
I_ISO          = 3             # process identifier: ion/elastic/isotropic
I_BACK         = 4             # process identifier: ion/elastic/backscattering
E_EXC_TH       = 11.5          # electron impact excitation threshold [eV]
E_ION_TH       = 15.8          # electron impact ionization threshold [eV]
CS_RANGES      = 1000000       # number of entries in cross section arrays
DE_CS          = 0.001         # energy division in cross section arrays [eV]

@dataclass
class cross_section:
    #Cross-section dataclass with a name, and a numpy array of size CS_RANGES
    name: str
    vals: np.ndarray = np.zeros(CS_RANGES)
#List of cross_section arrays
sigma = [cross_section(f"cs_{i}") for i in range(N_CS)]
sigma_tot_e = cross_section("sigma_tot_e") #total macroscopic cross-section of electrons
sigma_tot_i = cross_section("sigma_tot_i")   # total macroscopic cross section of ions

# particle coordinates

MAX_N_P = 1000000        # maximum number of particles (electrons / ions)
#Number of electrons, and ions
N_e, N_i = 0,0
@dataclass
class particle_vector:
    #Class for particle properties
    name: str
    vals: np.ndarray = np.zeros(MAX_N_P)
#Position, 3 velocities of electrons
x_e = particle_vector("x_e")
vx_e = particle_vector("vx_e")
vy_e = particle_vector("vy_e")
vz_e = particle_vector("vz_e")
#Position, 3 velocities of ions
x_i = particle_vector("x_i")
vx_i = particle_vector("vx_i")
vy_i = particle_vector("vy_i")
vz_i = particle_vector("vz_i")

@dataclass
class xvector:
    #Class for quantities defined at gridpoints
    name: str
    vals: np.ndarray = np.zeros(N_G)
efield = xvector("Electricfield")
pot = xvector("Potential")
e_density = xvector("electrondensity")
i_density = xvector("iondensity")
cumul_e_density = xvector("Cumulative_e_density")
cumul_i_density = xvector("Cumulative_i_density")

N_e_abs_pow  = 0    # counter for electrons absorbed at the powered electrode
N_e_abs_gnd  = 0    # counter for electrons absorbed at the grounded electrode
N_i_abs_pow  = 0    # counter for ions absorbed at the powered electrode
N_i_abs_gnd  = 0    # counter for ions absorbed at the grounded electrode

# electron energy probability function

N_EEPF  = 2000 # number of energy bins in Electron Energy Probability Function (EEPF)
DE_EEPF = 0.05     # resolution of EEPF [eV]
eepf = np.zeros(N_EEPF) # time integrated EEPF in the center of the plasma

# ion flux-energy distributions

N_IFED   = 200  # number of energy bins in Ion Flux-Energy Distributions (IFEDs)
DE_IFED  = 1.0                                 # resolution of IFEDs [eV]
#TODO: Class or just a numpy array for this IFED?
ifed_pow = np.zeros(N_IFED)# IFED at the powered electrode
ifed_gnd = np.zeros(N_IFED) # IFED at the grounded electrode


mean_i_energy_pow = 0.0         # mean ion energy at the powered electrode
mean_i_energy_gnd = 0.0         # mean ion energy at the grounded electrode

# spatio-temporal (XT) distributions

N_BIN       = 20       # number of time steps binned for the XT distributions
N_XT        = int(N_T / N_BIN)   # number of spatial bins for the XT distributions
@dataclass
class xt_distr:
    #Class with array for spatio-temporal distributions
    name: str
    vals: np.ndarray = np.zeros((N_G, N_XT))
pot_xt = xt_distr("potential")
efield_xt = xt_distr("electric_field")
ne_xt = xt_distr("electron_density")
ni_xt = xt_distr("ion_density")
ue_xt = xt_distr("mean_electron_velocity")
ui_xt = xt_distr("mean_ion_velocity")
je_xt = xt_distr("electron_current_density")
ji_xt = xt_distr("ion_current_density")
powere_xt = xt_distr("electron_power_absorption")
poweri_xt = xt_distr("ion_power_absorption")
meane_xt = xt_distr("electron_mean_energy")
meani_xt = xt_distr("ion_mean_energy")
counter_e_xt = xt_distr("electron_properties")
counter_i_xt = xt_distr("ion_properties")
ioniz_rate_xt = xt_distr("ionization_rate")


mean_energy_accu_center    = 0 # mean electron energy accumulator in the center of the gap
mean_energy_counter_center = 0 # mean electron energy counter in the center of the gap
N_e_coll                   = 0 # counter for electron collisions
N_i_coll                   = 0 # counter for ion collisions
Time = 0.0    # total simulated time (from the beginning of the simulation)

# current cycle and total cycles in the run, cycles completed
cycle = 0
no_of_cycles = 0
cycles_done  = 0 

def get_argparser():
    pr = argparse.ArgumentParser(prog="python3 eduPIC.py", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runs the eduPIC code, duh!")
    pr.add_argument("-cycles", type=int, default=1, help="Number of rf cycles")
    pr.add_argument("-measurement_mode", type=bool, default=False, help="Whether to measure some data and write it out.")
    pr.add_argument("-datafile", type=str, default="datafile.bin", help="File name used for saving data")
    return pr
#std::normal_distribution<> RMB(0.0,sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS));
rng = default_rng()


#----------------------------------------------------------------------------//
#  electron cross sections: A V Phelps & Z Lj Petrovic, PSST 8 R21 (1999)    //
#----------------------------------------------------------------------------//


def set_electron_cross_sections_ar():
    print("Setting electron cross-sections.\n")
    #TODO: the loop below is slowing things down, needs speedup
    for i in range(CS_RANGES):
        if (i==0):
            en = DE_CS
        else:
            en = DE_CS*i
        qmel = np.abs(6.0 / np.power(1.0 + (en/0.1) + np.power(en/0.6,2.0), 3.3)
                    - 1.1 * np.power(en, 1.4) / (1.0 + np.power(en/15.0, 1.2))/
                    np.sqrt(1.0 + np.power(en/5.5, 2.5) + np.power(en/60.0, 4.1)))
        + 0.05 / np.power(1.0 + en/10.0, 2.0) + 0.01 * np.power(en, 3.0) / (1.0 +
        np.power(en/12.0, 6.0))
        if (en > E_EXC_TH):
            qexc = 0.034 * np.power(en-11.5, 1.1) * (1.0 + 
            np.power(en/15.0, 28)) / (1.0 + np.power(en/23.0, 5.5))
            + 0.023 * (en-11.5) / np.power(1.0 + en/80.0, 1.9)
        else:
            qexc = 0
        if (en > E_ION_TH):
            qion = 970.0 * (en-15.8) / np.power(70.0 + en, 2.0) + 0.06 * np.power(en-15.8, 2.0) * np.exp(-en/9)
        else:
            qion = 0
        sigma[E_ELA].vals[i] = qmel * 1.0e-20;       # cross section for e- / Ar elastic collision
        sigma[E_EXC].vals[i] = qexc * 1.0e-20;       # cross section for e- / Ar excitation
        sigma[E_ION].vals[i] = qion * 1.0e-20;       # cross section for e- / Ar ionization
    print("Finished reading electron cross-sections of Ar.")
    return True


#------------------------------------------------------------------------------//
#  ion cross sections: A. V. Phelps, J. Appl. Phys. 76, 747 (1994)             //
#------------------------------------------------------------------------------//

def set_ion_cross_sections_ar():

    print("Setting ion cross-sections.\n")
    #TODO: the loop below is slowing things down, needs speedup
    for i in range(CS_RANGES):
        if (i==0):
            e_com = DE_CS
        else:
            e_com = DE_CS*i
        e_lab = 2.0 * e_com    # ion energy in the laboratory frame of reference
        qmom  = 1.15e-18 * np.power(e_lab,-0.1) * np.power(1.0 + 0.015 / e_lab, 0.6)
        qiso  = 2e-19 * np.power(e_lab,-0.5) / (1.0 + e_lab) + 3e-19 * e_lab / np.power(1.0 + e_lab / 3.0, 2.0)
        qback = (qmom-qiso) / 2.0
        sigma[I_ISO].vals[i]  = qiso # cross section for Ar+ / Ar isotropic part of elastic scattering
        sigma[I_BACK].vals[i] = qback # cross section for Ar+ / Ar backward elastic scattering
    print("Finished reading ion cross-sections of Ar.")
    return True
#----------------------------------------------------------------------//
#  calculation of total cross sections for electrons and ions          //
#----------------------------------------------------------------------//

def calc_total_cross_sections():
    #TODO: Below loop can be vectorized
    for i in range(CS_RANGES):
        sigma_tot_e.vals[i] = (sigma[E_ELA].vals[i] + sigma[E_EXC].vals[i] + sigma[E_ION].vals[i]) * GAS_DENSITY   # total macroscopic cross section of electrons
        sigma_tot_i.vals[i] = (sigma[I_ISO].vals[i] + sigma[I_BACK].vals[i]) * GAS_DENSITY                    # total macroscopic cross section of ions
    print("Total cross sections calculated.")
    return True
#----------------------------------------------------------------------//
#  test of cross sections for electrons and ions                       //
#----------------------------------------------------------------------//
def test_cross_sections():
    #TODO: Finish writing this function
    print("Finished checking cross sections.")
    return True
#void test_cross_sections(void){
#    FILE  * f;
#    int   i,j;
#    
#    f = fopen("cross_sections.dat","w");        # cross sections saved in data file: cross_sections.dat
#    for(i=0; i<CS_RANGES; i++){
#        fprint(f,"%12.4f ",i*DE_CS);
#        for(j=0; j<N_CS; j++) fprint(f,"%14e ",sigma[j][i]);
#        fprint(f,"\n");
#    }
#    fclose(f);
#}

#---------------------------------------------------------------------//
# find upper limit of collision frequencies                           //
#---------------------------------------------------------------------//

#TODO: Can the below functions be vectorized?
def max_electron_coll_freq():
    for i in range(CS_RANGES):
        e = i*DE_CS
        v  = np.sqrt(2.0 * e * EV_TO_J / E_MASS)
        nu = v * sigma_tot_e[i]
        if (nu > nu_max):
            nu_max = nu
    return nu_max

def max_ion_coll_freq():
    for i in range(CS_RANGES):
        e  = i * DE_CS
        g  = np.sqrt(2.0 * e * EV_TO_J / MU_ARAR)
        nu = g * sigma_tot_i[i]
        if (nu > nu_max):
            nu_max = nu
    return nu_max
    
#----------------------------------------------------------------------//
# initialization of the simulation by placing a given number of        //
# electrons and ions at random positions between the electrodes        //
#----------------------------------------------------------------------//

def init(nseed):
    global N_e, N_i
    for i in range(nseed):
        x_e.vals[i]  = L * rng.uniform(0,1)               # initial random position of the electron
        vx_e.vals[i] = 0
        vy_e.vals[i] = 0
        vz_e.vals[i] = 0  # initial velocity components of the electron
        x_i.vals[i]  = L * rng.uniform(0,1) # initial random position of the ion
        vx_i.vals[i] = 0
        vy_i.vals[i] = 0
        vz_i.vals[i] = 0  # initial velocity components of the ion
    N_e = nseed    # initial number of electrons
    N_i = nseed    # initial number of ions
    print("Initialised electrons and ions randomly")
    return True

#----------------------------------------------------------------------//
# e / Ar collision  (cold gas approximation)                           //
#----------------------------------------------------------------------//

def collision_electron(xe, v_e, eindex):
    F1 = E_MASS  / (E_MASS + AR_MASS)
    F2 = AR_MASS / (E_MASS + AR_MASS)
    # calculate relative velocity before collision & velocity of the centre of mass
    
    gx,gy,gz = v_e
    g  = np.sqrt(gx * gx + gy * gy + gz * gz)
    wx,wy,wz = [F1*i for i in v_e]
    
    # find Euler angles
    
    if (gx == 0):
        theta = 0.5 * PI
    else:
        theta = np.atan2(np.sqrt(gy * gy + gz * gz),gx)
    if (gy == 0):
        if (gz > 0):
            phi = 0.5 * PI
        else: 
            phi = - 0.5 * PI
    else:
        phi = np.atan2(gz, gy)
    st  = np.sin(theta)
    ct  = np.cos(theta)
    sp  = np.sin(phi)
    cp  = np.cos(phi)
    
    # choose the type of collision based on the cross sections
    # take into account energy loss in inelastic collisions
    # generate scattering and azimuth angles
    # in case of ionization handle the 'new' electron
    
    t0   =     sigma[E_ELA].val[eindex]
    t1   = t0 +sigma[E_EXC].val[eindex]
    t2   = t1 +sigma[E_ION].val[eindex]
    rnd  = rng.uniform(0,1)
    if (rnd < (t0/t2)):                              # elastic scattering
        chi = np.acos(1.0 - 2.0 * rng.uniform(0,1))  # isotropic scattering
        eta = TWO_PI * rng.uniform(0,1)                   # azimuthal angle
    elif (rnd < (t1/t2)):                       # excitation
        energy = 0.5 * E_MASS * g * g               # electron energy
        energy = np.abs(energy - E_EXC_TH * EV_TO_J)  # subtract energy loss for excitation
        g   = np.sqrt(2.0 * energy / E_MASS)           # relative velocity after energy loss
        chi = np.acos(1.0 - 2.0 * rng.uniform(0,1)) # isotropic scattering
        eta = TWO_PI * rng.uniform(0,1)         # azimuthal angle
    else:                                         # ionization
        energy = 0.5 * E_MASS * g * g               # electron energy
        energy = np.abs(energy - E_ION_TH * EV_TO_J)  # subtract energy loss of ionization
        e_ej  = 10.0 * np.tan(rng.uniform(0,1) * np.atan(energy/EV_TO_J / 20.0)) * EV_TO_J # energy of the ejected electron
        e_sc = np.abs(energy - e_ej) # energy of scattered electron after the collision
        g    = np.sqrt(2.0 * e_sc / E_MASS)            # relative velocity of scattered electron
        g2   = np.sqrt(2.0 * e_ej / E_MASS)            # relative velocity of ejected electron
        chi  = np.acos(np.sqrt(e_sc / energy))            # scattering angle for scattered electron
        chi2 = np.acos(np.sqrt(e_ej / energy))            # scattering angle for ejected electrons
        eta  = TWO_PI * rng.uniform(0,1)                 # azimuthal angle for scattered electron
        eta2 = eta + PI                             # azimuthal angle for ejected electron
        sc  = np.sin(chi2)
        cc  = np.cos(chi2)
        se  = np.sin(eta2)
        ce  = np.cos(eta2)
        gx  = g2 * (ct * cc - st * sc * ce)
        gy  = g2 * (st * cp * cc + ct * cp * sc * ce - sp * sc * se)
        gz  = g2 * (st * sp * cc + ct * sp * sc * ce + cp * sc * se)
        x_e[N_e]  = xe                              # add new electron
        vx_e[N_e] = wx + F2 * gx
        vy_e[N_e] = wy + F2 * gy
        vz_e[N_e] = wz + F2 * gz
        N_e+=1
        x_i[N_i]  = xe                              # add new ion
        vx_i[N_i],vy_i[N_i],vz_i[N_i] = rng.normal(0, np.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS), 3) # velocity is sampled from background thermal distribution
        N_i+=1
    
    # scatter the primary electron
    
    sc = np.sin(chi)
    cc = np.cos(chi)
    se = np.sin(eta)
    ce = np.cos(eta)
    
    # compute new relative velocity:
    
    gx = g * (ct * cc - st * sc * ce)
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se)
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se)
    
    # post-collision velocity of the colliding electron
    
    v_e = wx + F2 * gx, wy + F2 * gy, wz + F2 * gz
    return v_e
#----------------------------------------------------------------------//
# Ar+ / Ar collision                                                   //
#----------------------------------------------------------------------//

def collision_ion(v_1, v_2, e_index):
    gx,gy,gz = [i-j for i,j in zip(v_1, v_2)]
    g  = np.sqrt(gx * gx + gy * gy + gz * gz)
    wx,wy,wz = [0.5*(i+j) for i,j in zip(v_1, v_2)]
    
    # find Euler angles
    
    if (gx == 0):
        theta = 0.5 * PI
    else:
        theta = np.atan2(np.sqrt(gy * gy + gz * gz),gx)
    if (gy == 0):
        if (gz > 0):
            phi = 0.5 * PI
        else:
            phi = - 0.5 * PI
    else: 
        phi = np.atan2(gz, gy)
    
    # determine the type of collision based on cross sections and generate scattering angle
    
    t1  =      sigma[I_ISO].vals[e_index]
    t2  = t1 + sigma[I_BACK].vals[e_index]
    rnd = rng.uniform(0,1)
    if  (rnd < (t1 /t2)):                       # isotropic scattering
        chi = np.acos(1.0 - 2.0 * rng.uniform(0,1))     # scattering angle
    else:                                     # backward scattering
        chi = PI                                # scattering angle
    eta = TWO_PI * rng.uniform(0,1)                   # azimuthal angle
    sc  = np.sin(chi)
    cc  = np.cos(chi)
    se  = np.sin(eta)
    ce  = np.cos(eta)
    st  = np.sin(theta)
    ct  = np.cos(theta)
    sp  = np.sin(phi)
    cp  = np.cos(phi)
    
    # compute new relative velocity
    
    gx = g * (ct * cc - st * sc * ce)
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se)
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se)
    
    # post-collision velocity of the ion

    
    v_1 = wx + 0.5 * gx, wy + 0.5 * gy, wz + 0.5 * gz
    return v_1

#-----------------------------------------------------------------//
# solve Poisson equation (Thomas algorithm)                       //
#-----------------------------------------------------------------//

def solve_poisson(rho1, tt):
    A =  1.0
    B = -2.0
    C =  1.0
    S = 1.0 / (2.0 * DX)
    ALPHA = -DX * DX / EPSILON0
    g = xvector("g")
    w = xvector("w")
    f = xvector("f")
    
    # apply potential to the electrodes - boundary conditions
    
    pot[0]     = VOLTAGE * np.cos(OMEGA * tt) # potential at the powered electrode
    pot[N_G-1] = 0.0        # potential at the grounded electrode
    
    # solve Poisson equation
    
    f.vals[1:N_G-1] = ALPHA*rho1.vals[1:N_G-1]
    f.vals[1] -= pot[0]
    f.vals[N_G-2] -= pot[N_G-1]
    w.vals[1] = C/B
    g.vals[1] = f.vals[1]/B
    w.vals[2:N_G-1] =C / (B - A * w.vals[1:N_G-2]) 
    g.vals[2:N_G-1] = (f.vals[2:N_G-1] - A * g.vals[1:N_G-2]) / (B - A * w.vals[1-N_G-2])
    pot[N_G-2] = g.vals[N_G-2]
    for i in range(N_G-3,0,-1):
        pot[i] = g.vals[i] - w.vals[i] * pot[i+1]# potential at the grid points between the electrodes
    
    # compute electric field
    
    #Electric field at the points between the electrodes
    efield.vals[1:N_G-1] = (pot.vals[0:N_G-2] - pot.vals[2:N_G]) *S
    efield.vals[0] = (pot.vals[0] - pot.vals[1])* INV_DX - rho1.vals[0] * DX / (2.0 * EPSILON0);   # powered electrode
    efield.vals[N_G-1] = (pot.vals[N_G-2] - pot.vals[N_G-1]) * INV_DX + rho1.vals[N_G-1] * DX / (2.0 * EPSILON0);   # grounded electrode
    return True


#---------------------------------------------------------------------//
# simulation of one radiofrequency cycle                              //
#---------------------------------------------------------------------//


def do_one_cycle(Time):
    global N_e, N_i
    DV       = ELECTRODE_AREA * DX
    FACTOR_W = WEIGHT / DV
    FACTOR_E = DT_E / E_MASS * E_CHARGE
    FACTOR_I = DT_I / AR_MASS * E_CHARGE
    MIN_X    = 0.45 * L         # min. position for EEPF collection
    MAX_X    = 0.55 * L         # max. position for EEPF collection
    poisolve = False
    #int      k, t, p, energy_index
    #double   g, g_sqr, gx, gy, gz, vx_a, vy_a, vz_a, e_x, energy, nu, p_coll, v_sqr, velocity
    #double   mean_v, c0, c1, c2, rate
    #bool     out;
    rho = xvector("rho")

    
    for t in range(N_T):
    # the RF period is divided into N_T equal time intervals (time step DT_E)
        Time += DT_E;               # update of the total simulated time
        t_index = t / N_BIN;        # index for XT distributions
        
        # step 1: compute densities at grid points
        

        for k in range(N_e):
            c0 = x_e[k] * INV_DX
            p  = int(c0)
            e_density.vals[p]   += (p + 1 - c0) * FACTOR_W
            e_density.vals[p+1] += (c0 - p) * FACTOR_W
        e_density.vals[0]     *= 2.0
        e_density.vals[N_G-1] *= 2.0
        for p in range(N_G):
            cumul_e_density.vals[p] += e_density.vals[p]
        
        #ion density - computed in every N_SUB-th time steps (subcycling)
        if ((t % N_SUB) == 0): 
            for k in range(N_i):
                c0 = x_i.vals[k] * INV_DX
                p  = int(c0)
                i_density.vals[p]   += (p + 1 - c0) * FACTOR_W  
                i_density.vals[p+1] += (c0 - p) * FACTOR_W
            i_density.vals[0]     *= 2.0
            i_density.vals[N_G-1] *= 2.0
        for p in range(N_G):
            cumul_i_density.vals[p] += i_density.vals[p]
        
        # step 2: solve Poisson equation
        
        #Getting charge density
        rho.vals[:] = E_CHARGE*(i_density.vals[:] - e_density.vals[:])
        poisolve = solve_poisson(rho, Time) #Computing potential and electric field
        
        # steps 3 & 4: move particles according to electric field interpolated to particle positions
        
        for k in range(N_e):
        # move all electrons in every time step
            c0  = x_e.vals[k] * INV_DX
            p   = int(c0)
            c1  = p + 1.0 - c0
            c2  = c0 - p
            e_x = c1 * efield.vals[p] + c2 * efield.vals[p+1];
            
            if (args.measurement_mode):
                
                # measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'
                
                mean_v = vx_e.vals[k] - 0.5 * e_x * FACTOR_E
                counter_e_xt.vals[p][t_index]   += c1
                counter_e_xt.vals[p+1][t_index] += c2
                ue_xt.vals[p][t_index]   += c1 * mean_v
                ue_xt.vals[p+1][t_index] += c2 * mean_v
                v_sqr  = mean_v * mean_v + vy_e.vals[k] * vy_e.vals[k] + vz_e[k] * vz_e.vals[k]
                energy = 0.5 * E_MASS * v_sqr / EV_TO_J
                meane_xt.vals[p][t_index]   += c1 * energy
                meane_xt.vals[p+1][t_index] += c2 * energy
                energy_index = min( int(energy / DE_CS + 0.5), CS_RANGES-1);
                velocity = np.sqrt(v_sqr);
                rate = sigma[E_ION].vals[energy_index] * velocity * DT_E * GAS_DENSITY;
                ioniz_rate_xt.vals[p][t_index]   += c1 * rate;
                ioniz_rate_xt.vals[p+1][t_index] += c2 * rate;

                # measure EEPF in the center
                
                if ((MIN_X < x_e[k]) & (x_e[k] < MAX_X)):
                    energy_index = (int)(energy / DE_EEPF)
                    if (energy_index < N_EEPF):
                        eepf[energy_index] += 1.0
                    mean_energy_accu_center += energy
                    mean_energy_counter_center+=1
            
            # update velocity and position
            
            vx_e.vals[k] -= e_x * FACTOR_E
            x_e.vals[k]  += vx_e.vals[k] * DT_E
        
        # move all ions in every N_SUB-th time steps (subcycling)
        if ((t % N_SUB) == 0):
            for k in range(N_i):
                c0  = x_i.vals[k] * INV_DX
                p   = int(c0)
                c1  = p + 1 - c0
                c2  = c0 - p
                e_x = c1 * efield.vals[p] + c2 * efield.vals[p+1]
                
                if (args.measurement_mode):
                    
                    # measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'

                    mean_v = vx_i.vals[k] + 0.5 * e_x * FACTOR_I
                    counter_i_xt.vals[p][t_index]   += c1
                    counter_i_xt.vals[p+1][t_index] += c2
                    ui_xt.vals[p][t_index]   += c1 * mean_v
                    ui_xt.vals[p+1][t_index] += c2 * mean_v
                    v_sqr  = mean_v * mean_v + vy_i.vals[k] * vy_i.vals[k] + vz_i.vals[k] * vz_i.vals[k]
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J
                    meani_xt.vals[p][t_index]   += c1 * energy
                    meani_xt.vals[p+1][t_index] += c2 * energy
                
                # update velocity and position and accumulate absorbed energy
                
                vx_i.vals[k] += e_x * FACTOR_I
                x_i.vals[k]  += vx_i.vals[k] * DT_I
        
        # step 5: check boundaries
        
        k = 0
        # check boundaries for all electrons in every time step
        while(k < N_e):    
            out = False
            if (x_e.vals[k] < 0):
                N_e_abs_pow+=1 
                out = True    # the electron is out at the powered electrode
            if (x_e[k] > L):
                N_e_abs_gnd+=1
                out = True    # the electron is out at the grounded electrode
            if (out):  # remove the electron, if out
                x_e.vals[k] = x_e.vals[N_e-1]
                vx_e.vals[k] = vx_e.vals[N_e-1]
                vy_e.vals[k] = vy_e.vals[N_e-1]
                vz_e.vals[k] = vz_e.vals[N_e-1]
                N_e-=1
            else:
                k+=1
        
        # check boundaries for all ions in every N_SUB-th time steps (subcycling)
        if ((t % N_SUB) == 0):
            k = 0
            while(k < N_i):
                out = False
                if (x_i.vals[k] < 0):#the ion is out at the powered electrode
                    N_i_abs_pow+=1
                    out    = True
                    v_sqr  = vx_i.vals[k] * vx_i.vals[k] + vy_i.vals[k] * vy_i.vals[k] + vz_i.vals[k] * vz_i.vals[k]
                    energy = 0.5 * AR_MASS *  v_sqr/ EV_TO_J
                    energy_index = int(energy / DE_IFED)
                    if (energy_index < N_IFED):
                        ifed_pow[energy_index]+=1 # save IFED at the powered electrode
                if (x_i[k] > L): # the ion is out at the grounded electrode
                    N_i_abs_gnd+=1
                    out = True
                    v_sqr  = vx_i.vals[k] * vx_i.vals[k] + vy_i.vals[k] * vy_i.vals[k] + vz_i.vals[k] * vz_i.vals[k]
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J
                    energy_index = int(energy / DE_IFED)
                    if (energy_index < N_IFED): 
                        ifed_gnd[energy_index]+=1       # save IFED at the grounded electrode
                if (out):# delete the ion, if out
                    x_i.vals[k] = x_i.vals[N_i-1]
                    vx_i.vals[k] = vx_i.vals[N_i-1]
                    vy_i.vals[k] = vy_i.vals[N_i-1]
                    vz_i.vals[k] = vz_i.vals[N_i-1]
                    N_i-=1
                else:
                    k+=1
        
        # step 6: collisions
        
        for k in range(N_e):
        # checking for occurrence of a collision for all electrons in every time step
            v_sqr = vx_e.vals[k] * vx_e.vals[k] + vy_e.vals[k] * vy_e.vals[k] + vz_e.vals[k] * vz_e.vals[k]
            velocity = np.sqrt(v_sqr)
            energy   = 0.5 * E_MASS * v_sqr / EV_TO_J
            energy_index = min( int(energy / DE_CS + 0.5), CS_RANGES-1)
            nu = sigma_tot_e.vals[energy_index] * velocity
            p_coll = 1 - np.exp(- nu * DT_E)                  # collision probability for electrons
            if (rng.uniform(0,1) < p_coll):  # electron collision takes place
                vx_e.vals[k], vy_e.vals[k], vz_e.vals[k] = collision_electron(x_e[k], [vx_e.vals[k], vy_e.vals[k], vz_e.vals[k]], energy_index)
                N_e_coll+=1
        
# checking for occurrence of a collision for all ions in every N_SUB-th time steps (subcycling)
        if ((t % N_SUB) == 0): 
            for k in range(N_i):
                # pick velocity components of a random target gas atom
                vx_a,vy_a,vz_a = rng.normal(0, np.sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS), 3)                          
                # compute the relative velocity of the collision partners
                gx   = vx_i.vals[k] - vx_a 
                gy   = vy_i.vals[k] - vy_a
                gz   = vz_i.vals[k] - vz_a
                g_sqr = gx * gx + gy * gy + gz * gz
                g = np.sqrt(g_sqr)
                energy = 0.5 * MU_ARAR * g_sqr / EV_TO_J
                energy_index = min( int(energy / DE_CS + 0.5), CS_RANGES-1)
                nu = sigma_tot_i.vals[energy_index] * g
                p_coll = 1 - np.exp(- nu * DT_I)# collision probability for ions
                if (rng.uniform(0,1) < p_coll): # ion collision takes place
                    vx_i.vals[k], vy_i.vals[k], vz_i.vals[k] = collision_ion ([vx_i.vals[k], vy_i.vals[k], vz_i.vals[k]], [vx_a, vy_a, vz_a], energy_index)
                    N_i_coll+=1
        
        if (args.measurement_mode):
            
            # collect 'xt' data from the grid
            
            #TODO: Can this be vectorized?
            for p in range(N_G):
                pot_xt.vals[p][t_index] += pot.vals[p]
                efield_xt.vals[p][t_index] += efield.vals[p]
                ne_xt.vals[p][t_index] += e_density.vals[p]
                ni_xt.vals[p][t_index] += i_density.vals[p]
        
        if ((t % 1000) == 0):
            print(" c = %8d  t = %8d  #e = %8d  #i = %8d\n", cycle,t,N_e,N_i)

    print(datafile,"%8d  %8d  %8d\n",cycle,N_e,N_i)
    return Time

#---------------------------------------------------------------------//
# save particle coordinates                                           //
#---------------------------------------------------------------------//

#TODO: Need to convert this
#def save_particle_data():
#
#    fname = "picdata.bin"
#    f = open(fname,"wb")
#
#    fwrite(&Time,sizeof(double),1,f);
#    d = (double)(cycles_done);
#    fwrite(&d,sizeof(double),1,f);
#    d = (double)(N_e);
#    fwrite(&d,sizeof(double),1,f);
#    fwrite(x_e, sizeof(double),N_e,f);
#    fwrite(vx_e,sizeof(double),N_e,f);
#    fwrite(vy_e,sizeof(double),N_e,f);
#    fwrite(vz_e,sizeof(double),N_e,f);
#    d = (double)(N_i);
#    fwrite(&d,sizeof(double),1,f);
#    fwrite(x_i, sizeof(double),N_i,f);
#    fwrite(vx_i,sizeof(double),N_i,f);
#    fwrite(vy_i,sizeof(double),N_i,f);
#    fwrite(vz_i,sizeof(double),N_i,f);
#    fclose(f);
#    print(">> eduPIC: data saved : %d electrons %d ions, %d cycles completed, time is %e [s]\n",N_e,N_i,cycles_done,Time);
#
##---------------------------------------------------------------------//
## load particle coordinates                                           //
##---------------------------------------------------------------------//
#
##TODO: Need to convert this
#void load_particle_data(){
#    double   d;
#    FILE   * f;
#    char fname[80];
#    
#    strcpy(fname,"picdata.bin");
#    f = fopen(fname,"rb");
#    if (f==NULL) {print(">> eduPIC: ERROR: No particle data file found, try running initial cycle using argument '0'\n"); exit(0); }
#    fread(&Time,sizeof(double),1,f);
#    fread(&d,sizeof(double),1,f);
#    cycles_done = int(d);
#    fread(&d,sizeof(double),1,f);
#    N_e = int(d);
#    fread(x_e, sizeof(double),N_e,f);
#    fread(vx_e,sizeof(double),N_e,f);
#    fread(vy_e,sizeof(double),N_e,f);
#    fread(vz_e,sizeof(double),N_e,f);
#    fread(&d,sizeof(double),1,f);
#    N_i = int(d);
#    fread(x_i, sizeof(double),N_i,f);
#    fread(vx_i,sizeof(double),N_i,f);
#    fread(vy_i,sizeof(double),N_i,f);
#    fread(vz_i,sizeof(double),N_i,f);
#    fclose(f);
#    print(">> eduPIC: data loaded : %d electrons %d ions, %d cycles completed before, time is %e [s]\n",N_e,N_i,cycles_done,Time);
#}
#
##---------------------------------------------------------------------//
## save density data                                                   //
##---------------------------------------------------------------------//
#
##TODO: Need to convert this
#void save_density(void){
#    FILE *f;
#    double c;
#    int m;
#    
#    f = fopen("density.dat","w");
#    c = 1.0 / (double)(no_of_cycles) / (double)(N_T);
#    for(m=0; m<N_G; m++){
#        fprint(f,"%8.5f  %12e  %12e\n",m * DX, cumul_e_density[m] * c, cumul_i_density[m] * c);
#    }
#    fclose(f);
#}
#
##---------------------------------------------------------------------//
## save EEPF data                                                      //
##---------------------------------------------------------------------//
#
##TODO: Need to convert this
#void save_eepf(void) {
#    FILE   *f;
#    int    i;
#    double h,energy;
#    
#    h = 0.0;
#    for (i=0; i<N_EEPF; i++) {h += eepf[i];}
#    h *= DE_EEPF;
#    f = fopen("eepf.dat","w");
#    for (i=0; i<N_EEPF; i++) {
#        energy = (i + 0.5) * DE_EEPF;
#        fprint(f,"%e  %e\n", energy, eepf[i] / h / sqrt(energy));
#    }
#    fclose(f);
#}
#
##---------------------------------------------------------------------//
## save IFED data                                                      //
##---------------------------------------------------------------------//
#
##TODO: Need to convert this
#void save_ifed(void) {
#    FILE   *f;
#    int    i;
#    double h_pow,h_gnd,energy;
#    
#    h_pow = 0.0;
#    h_gnd = 0.0;
#    for (i=0; i<N_IFED; i++) {h_pow += ifed_pow[i]; h_gnd += ifed_gnd[i];}
#    h_pow *= DE_IFED;
#    h_gnd *= DE_IFED;
#    mean_i_energy_pow = 0.0;
#    mean_i_energy_gnd = 0.0;
#    f = fopen("ifed.dat","w");
#    for (i=0; i<N_IFED; i++) {
#        energy = (i + 0.5) * DE_IFED;
#        fprint(f,"%6.2f %10.6f %10.6f\n", energy, (double)(ifed_pow[i])/h_pow, (double)(ifed_gnd[i])/h_gnd);
#        mean_i_energy_pow += energy * (double)(ifed_pow[i]) / h_pow;
#        mean_i_energy_gnd += energy * (double)(ifed_gnd[i]) / h_gnd;
#    }
#    fclose(f);
#}
#
##--------------------------------------------------------------------//
## save XT data                                                       //
##--------------------------------------------------------------------//
#
##TODO: Need to convert this
#void save_xt_1(xt_distr distr, char *fname) {
#    FILE   *f;
#    int    i, j;
#    
#    f = fopen(fname,"w");
#    for (i=0; i<N_G; i++){
#        for (j=0; j<N_XT; j++){
#            fprint(f,"%e  ", distr[i][j]);
#        }
#        fprint(f,"\n");
#    }
#    fclose(f);
#}
#
##TODO: Need to convert this
#void norm_all_xt(void){
#    double f1, f2;
#    int    i, j;
#    
#    # normalize all XT data
#    
#    f1 = (double)(N_XT) / (double)(no_of_cycles * N_T);
#    f2 = WEIGHT / (ELECTRODE_AREA * DX) / (no_of_cycles * (PERIOD / (double)(N_XT)));
#    
#    for (i=0; i<N_G; i++){
#        for (j=0; j<N_XT; j++){
#            pot_xt[i][j]    *= f1;
#            efield_xt[i][j] *= f1;
#            ne_xt[i][j]     *= f1;
#            ni_xt[i][j]     *= f1;
#            if (counter_e_xt[i][j] > 0) {
#                ue_xt[i][j]     =  ue_xt[i][j] / counter_e_xt[i][j];
#                je_xt[i][j]     = -ue_xt[i][j] * ne_xt[i][j] * E_CHARGE;
#                meanee_xt[i][j] =  meanee_xt[i][j] / counter_e_xt[i][j];
#                ioniz_rate_xt[i][j] *= f2;
#             } else {
#                ue_xt[i][j]         = 0.0;
#                je_xt[i][j]         = 0.0;
#                meanee_xt[i][j]     = 0.0;
#                ioniz_rate_xt[i][j] = 0.0;
#            }
#            if (counter_i_xt[i][j] > 0) {
#                ui_xt[i][j]     = ui_xt[i][j] / counter_i_xt[i][j];
#                ji_xt[i][j]     = ui_xt[i][j] * ni_xt[i][j] * E_CHARGE;
#                meanei_xt[i][j] = meanei_xt[i][j] / counter_i_xt[i][j];
#            } else {
#                ui_xt[i][j]     = 0.0;
#                ji_xt[i][j]     = 0.0;
#                meanei_xt[i][j] = 0.0;
#            }
#            powere_xt[i][j] = je_xt[i][j] * efield_xt[i][j];
#            poweri_xt[i][j] = ji_xt[i][j] * efield_xt[i][j];
#        }
#    }
#}
#
##TODO: Need to convert this
#void save_all_xt(void){
#    char fname[80];
#    
#    strcpy(fname,"pot_xt.dat");     save_xt_1(pot_xt, fname);
#    strcpy(fname,"efield_xt.dat");  save_xt_1(efield_xt, fname);
#    strcpy(fname,"ne_xt.dat");      save_xt_1(ne_xt, fname);
#    strcpy(fname,"ni_xt.dat");      save_xt_1(ni_xt, fname);
#    strcpy(fname,"je_xt.dat");      save_xt_1(je_xt, fname);
#    strcpy(fname,"ji_xt.dat");      save_xt_1(ji_xt, fname);
#    strcpy(fname,"powere_xt.dat");  save_xt_1(powere_xt, fname);
#    strcpy(fname,"poweri_xt.dat");  save_xt_1(poweri_xt, fname);
#    strcpy(fname,"meanee_xt.dat");  save_xt_1(meanee_xt, fname);
#    strcpy(fname,"meanei_xt.dat");  save_xt_1(meanei_xt, fname);
#    strcpy(fname,"ioniz_xt.dat");   save_xt_1(ioniz_rate_xt, fname);
#}
#
##---------------------------------------------------------------------//
## simulation report including stability and accuracy conditions       //
##---------------------------------------------------------------------//
#
##TODO: Need to convert this
#void check_and_save_info(void){
#    FILE     *f;
#    double   plas_freq, meane, kT, debye_length, density, ecoll_freq, icoll_freq, sim_time, e_max, v_max, power_e, power_i, c;
#    int      i,j;
#    bool     conditions_OK;
#    
#    density    = cumul_e_density[N_G / 2] / (double)(no_of_cycles) / (double)(N_T);  # e density @ center
#    plas_freq  = E_CHARGE * sqrt(density / EPSILON0 / E_MASS);                       # e plasma frequency @ center
#    meane      = mean_energy_accu_center / (double)(mean_energy_counter_center);     # e mean energy @ center
#    kT         = 2.0 * meane * EV_TO_J / 3.0;                                        # k T_e @ center (approximate)
#    sim_time   = (double)(no_of_cycles) / FREQUENCY;                                 # simulated time
#    ecoll_freq = (double)(N_e_coll) / sim_time / (double)(N_e);                      # e collision frequency
#    icoll_freq = (double)(N_i_coll) / sim_time / (double)(N_i);                      # ion collision frequency
#    debye_length = sqrt(EPSILON0 * kT / density) / E_CHARGE;                         # e Debye length @ center
#    
#    f = fopen("info.txt","w");
#    fprint(f,"########################## eduPIC simulation report ############################\n");
#    fprint(f,"Simulation parameters:\n");
#    fprint(f,"Gap distance                          = %12.3e [m]\n",  L);
#    fprint(f,"# of grid divisions                   = %12d\n",      N_G);
#    fprint(f,"Frequency                             = %12.3e [Hz]\n", FREQUENCY);
#    fprint(f,"# of time steps / period              = %12d\n",      N_T);
#    fprint(f,"# of electron / ion time steps        = %12d\n",      N_SUB);
#    fprint(f,"Voltage amplitude                     = %12.3e [V]\n",  VOLTAGE);
#    fprint(f,"Pressure (Ar)                         = %12.3e [Pa]\n", PRESSURE);
#    fprint(f,"Temperature                           = %12.3e [K]\n",  TEMPERATURE);
#    fprint(f,"Superparticle weight                  = %12.3e\n",      WEIGHT);
#    fprint(f,"# of simulation cycles in this run    = %12d\n",      no_of_cycles);
#    fprint(f,"--------------------------------------------------------------------------------\n");
#    fprint(f,"Plasma characteristics:\n");
#    fprint(f,"Electron density @ center             = %12.3e [m^{-3}]\n", density);
#    fprint(f,"Plasma frequency @ center             = %12.3e [rad/s]\n",  plas_freq);
#    fprint(f,"Debye length @ center                 = %12.3e [m]\n",      debye_length);
#    fprint(f,"Electron collision frequency          = %12.3e [1/s]\n",    ecoll_freq);
#    fprint(f,"Ion collision frequency               = %12.3e [1/s]\n",    icoll_freq);
#    fprint(f,"--------------------------------------------------------------------------------\n");
#    fprint(f,"Stability and accuracy conditions:\n");
#    conditions_OK = true;
#    c = plas_freq * DT_E;
#    fprint(f,"Plasma frequency @ center * DT_E      = %12.3f (OK if less than 0.20)\n", c);
#    if (c > 0.2) {conditions_OK = false;}
#    c = DX / debye_length;
#    fprint(f,"DX / Debye length @ center            = %12.3f (OK if less than 1.00)\n", c);
#    if (c > 1.0) {conditions_OK = false;}
#    c = max_electron_coll_freq() * DT_E;
#    fprint(f,"Max. electron coll. frequency * DT_E  = %12.3f (OK if less than 0.05)\n", c);
#    if (c > 0.05) {conditions_OK = false;}
#    c = max_ion_coll_freq() * DT_I;
#    fprint(f,"Max. ion coll. frequency * DT_I       = %12.3f (OK if less than 0.05)\n", c);
#    if (c > 0.05) {conditions_OK = false;}
#    if (conditions_OK == false){
#        fprint(f,"--------------------------------------------------------------------------------\n");
#        fprint(f,"** STABILITY AND ACCURACY CONDITION(S) VIOLATED - REFINE SIMULATION SETTINGS! **\n");
#        fprint(f,"--------------------------------------------------------------------------------\n");
#        fclose(f);
#        print(">> eduPIC: ERROR: STABILITY AND ACCURACY CONDITION(S) VIOLATED!\n");
#        print(">> eduPIC: for details see 'info.txt' and refine simulation settings!\n");
#    }
#    else
#    {
#        # calculate maximum energy for which the Courant-Friedrichs-Levy condition holds:
#        
#        v_max = DX / DT_E;
#        e_max = 0.5 * E_MASS * v_max * v_max / EV_TO_J;
#        fprint(f,"Max e- energy for CFL condition       = %12.3f [eV]\n", e_max);
#        fprint(f,"Check EEPF to ensure that CFL is fulfilled for the majority of the electrons!\n");
#        fprint(f,"--------------------------------------------------------------------------------\n");
#        
#        # saving of the following data is done here as some of the further lines need data
#        # that are computed / normalized in these functions
#        
#        print(">> eduPIC: saving diagnostics data\n");
#        save_density();
#        save_eepf();
#        save_ifed();
#        norm_all_xt();
#        save_all_xt();
#        fprint(f,"Particle characteristics at the electrodes:\n");
#        fprint(f,"Ion flux at powered electrode         = %12.3e [m^{-2} s^{-1}]\n", N_i_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
#        fprint(f,"Ion flux at grounded electrode        = %12.3e [m^{-2} s^{-1}]\n", N_i_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
#        fprint(f,"Mean ion energy at powered electrode  = %12.3e [eV]\n", mean_i_energy_pow);
#        fprint(f,"Mean ion energy at grounded electrode = %12.3e [eV]\n", mean_i_energy_gnd);
#        fprint(f,"Electron flux at powered electrode    = %12.3e [m^{-2} s^{-1}]\n", N_e_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
#        fprint(f,"Electron flux at grounded electrode   = %12.3e [m^{-2} s^{-1}]\n", N_e_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
#        fprint(f,"--------------------------------------------------------------------------------\n");
#        
#        # calculate spatially and temporally averaged power absorption by the electrons and ions
#        
#        power_e = 0.0;
#        power_i = 0.0;
#        for (i=0; i<N_G; i++){
#            for (j=0; j<N_XT; j++){
#                power_e += powere_xt[i][j];
#                power_i += poweri_xt[i][j];
#            }
#        }
#        power_e /= (double)(N_XT * N_G);
#        power_i /= (double)(N_XT * N_G);
#        fprint(f,"Absorbed power calculated as <j*E>:\n");
#        fprint(f,"Electron power density (average)      = %12.3e [W m^{-3}]\n", power_e);
#        fprint(f,"Ion power density (average)           = %12.3e [W m^{-3}]\n", power_i);
#        fprint(f,"Total power density(average)          = %12.3e [W m^{-3}]\n", power_e + power_i);
#        fprint(f,"--------------------------------------------------------------------------------\n");
#        fclose(f);
#    }
#}

#------------------------------------------------------------------------------------------//
# main                                                                                     //
# command line arguments:                                                                  //
# [1]: number of cycles (0 for init)                                                       //
# [2]: "m" turns on data collection and saving                                             //
#------------------------------------------------------------------------------------------//
if __name__ == "__main__":
    pr = get_argparser()
    args = pr.parse_args()
    print(args)

    print(">> eduPIC: starting...\n")
    print(">> eduPIC: **************************************************************************\n")
    print(">> eduPIC: Copyright (C) 2021 Z. Donko et al.\n")
    print(">> eduPIC: This program comes with ABSOLUTELY NO WARRANTY\n")
    print(">> eduPIC: This is free software, you are welcome to use, modify and redistribute it\n")
    print(">> eduPIC: according to the GNU General Public License, https:#www.gnu.org/licenses/\n")
    print(">> eduPIC: **************************************************************************\n")

    
    if (args.measurement_mode):
        print(">> eduPIC: measurement mode: on\n")
    else:
        print(">> eduPIC: measurement mode: off\n")
    set_electron_cs = set_electron_cross_sections_ar()
    set_ion_cs = set_ion_cross_sections_ar()
    compute_total_cs = calc_total_cross_sections()
    #test_cross_sections(); return 1;
    datafile = open("conv.dat","a")
    #if (arg1 == 0) {
    #if (FILE *file = fopen("picdata.bin", "r")) { fclose(file);
    # print(">> eduPIC: Warning: Data from previous calculation are detected.\n");
    # print("           To start a new simulation from the beginning, please delete all output files before running ./eduPIC 0\n");
    # print("           To continue the existing calculation, please specify the number of cycles to run, e.g. ./eduPIC 100\n");
    # exit(0);
    #} 
    no_of_cycles = 1
    cycle = 1                                        # init cycle
    init(N_INIT)                          # seed initial electrons & ions
    print(">> eduPIC: running initializing cycle\n")
    Time = 0
    Time = do_one_cycle(Time)
    cycles_done = 1
    no_of_cycles = args.cycles # run number of cycles specified in command line
    #TODO: Function below is not yet re-written, not needed when not restarting the sim
    #load_particle_data = load_particle_data() # read previous configuration from file
    print(">> eduPIC: running %d cycle(s)\n",no_of_cycles)
    for cycle in range(cycles_done+1, cycles_done+no_of_cycles+1):
        Time = do_one_cycle(Time)

    cycles_done += no_of_cycles
    fclose(datafile) #TODO
    save_particle_data() #TODO
    if (args.measurement_mode): #TODO
        check_and_save_info()
    print(">> eduPIC: simulation of %d cycle(s) is completed.\n",no_of_cycles)
