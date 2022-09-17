#!/usr/bin/env python3

import numpy as np
import time
np.set_printoptions(precision=3, linewidth=300)
import subprocess
import argparse
import json
import sys
import matplotlib.pyplot as plt
import scipy.optimize
from datetime import datetime
import pathlib

const_e = 1.602176634e-19
const_hbar = 1.0545718176461565e-34
const_h = 6.62607015e-34

const_flux = const_h / (2 * const_e)

parser = argparse.ArgumentParser()
parser.add_argument('-Nx', help="system size (mandatory)", type=int,
                    required=True)
parser.add_argument('-Ny', help="system size (mandatory)", type=int,
                    required=True)
# parser.add_argument('-m', "--maxiter",
#                     help="number of iterations (default: 1000)",
#                     type=int, default=1000)
# parser.add_argument('-t', "--temp",
#                     help="initial temperature for annealing (default: 100e3)",
#                     type=float, default=100e3)
# parser.add_argument("--visit",
#                     help="value for visit parameter for annealing (default: 2)",
#                     type=float,
#                     default=2)
# parser.add_argument("--restart_temp_ratio",
#                     help="restart annealing when temperature has fallen below initial_temp * restart_ratio. Resets temperature to initial_temp and uses a new random start state. Depending on maxiter, this can happen multiple times. Effectively the same as calling dual annealing multiple times and selecting the best annealing result (default: 1e-9, do not restart)",
#                     type=float,
#                     default=1e-9)
parser.add_argument("-f", "--frustration", help="magnetic flux quanta per palette (default: 0)",
                    type=float,
                    default=0)
parser.add_argument('-o', "--output-dir", help="output folder (default: use date+time")
parser.add_argument('-v', '--pdf-viewer', help="open output pdf files with viewer")

parser_args = parser.parse_args()

#
# JJA has size Nx, Ny
# this is the total size, including a possible frame/edge
#
# Nx_int, Ny_int are internal size, meaning that the phi_matrix given to the optimizer has size Nx_int x Ny_int



Nx = parser_args.Nx
Ny = parser_args.Ny

    
f = parser_args.frustration

output_dir = parser_args.output_dir
pdf_viewer = parser_args.pdf_viewer



date_string = datetime.now()
date_string = date_string.strftime("%Y-%m-%d_%H-%M-%S")

data_folder = date_string + "_JJA"

for key, value in vars(parser_args).items():
    if value is not None:
        data_folder += "_" + key + "=" + str(value)
    
print("data folder: ", data_folder)
pathlib.Path(data_folder).mkdir()



run_stats = {}

# JJA Layout:

## # # # # # ##
## # # # # # ##
## # # # # # ##
## # # # # # ##
## # # # # # ##

# ^
# | y
#  --> x
#
# x is 0th-dimension, y is 1st dimension of numpy array




# φ_L is the phase in the left busbar, the phase in the right bus-bar is
# set to zero.

# print scalar field or component of vector field

def print_matrix(phi):
    phi = phi.T
    phi = np.flip(phi, axis=0)
    print(phi)
    return(phi)
    

def save_matrix_as_txt(matrix, filename):
    with open(filename, 'w') as outfile:
        np.savetxt(outfile, matrix, fmt="%.17g", delimiter="\t")
        outfile.write("\n")
    

#
# gamma and A are vectorfields with one matrix for x and y components
#
# gamma_x, A_x are (N+1) x N matrix, gamma_y, A_y are N x (N-1) matrix

#

#

#
# the real part (phi part) of gamma can get as large as +-4π, so the function of
# the free energy needs to be well-defined for these values.


def gamma_matrices(phi_matrix, phi_L, A_x_matrix, A_y_matrix):
    # diff in x-direction
    gamma_x = np.zeros((Nx + 1, Ny))
    gamma_x += A_x_matrix
                       
    # internal gammas
    gamma_x[1:-1,:] += phi_matrix[1:,:] - phi_matrix[:-1,:]

    # left busbar
    gamma_x[0,:] += phi_matrix[0,:] - phi_L

    # right busbar
    gamma_x[-1,:] += (0 - phi_matrix[-1,:])
    
    # gamma_x= phi_matrix[1:,:] - phi_matrix[:-1,:] + A_x_matrix

    # diff in y-direction
    gamma_y = phi_matrix[:,1:] - phi_matrix[:,:-1] + A_y_matrix
    return (gamma_x, gamma_y)

## x direction: I(φ) = b1x sin(φ) + b2x sin(2φ) + a1x cos(φ)
# b1x = 1
# b2x = 0.4
# a1x = 0

# b1y = 1
# b2y = 0.4
# a1y = 0

tau = 1
anisotropy = 1.5

def current_phase_relation(gamma_x_matrix, gamma_y_matrix):
    x_currents = np.sin(gamma_x_matrix)
    y_currents = np.sin(gamma_y_matrix)
    # # x_currents = b1x * np.sin(gamma_x_matrix) +\
    # #     b2x * np.sin(2 * gamma_x_matrix) +\
    # #     a1x * np.cos(gamma_x_matrix)
    
    # # y_currents =  b1y * np.sin(gamma_y_matrix) +\
    # #     b2y * np.sin(2 * gamma_y_matrix) +\
    # #     a1y * np.cos(gamma_y_matrix)

    # x_currents = np.sin(gamma_x_matrix) / np.sqrt(1 - tau * np.sin(gamma_x_matrix /2)**2)
    # y_currents = anisotropy * np.sin(gamma_y_matrix) / np.sqrt(1 - tau * np.sin(gamma_y_matrix /2)**2)
    
    return (x_currents, y_currents)

# def free_energy(gamma_x_matrix, gamma_y_matrix):
#     # f >= 0, f(γ = 0) = 0
#     free_energy = 0
#     # free_energy += np.sum(-np.sqrt(1 - tau * np.sin(gamma_x_matrix / 2)**2))
#     # free_energy += anisotropy * np.sum(-np.sqrt(1 - tau * np.sin(gamma_y_matrix
#                                                                  / 2)**2))
#     # free_energy += np.sum(- b1x * np.cos(gamma_x_matrix) - 0.5 * b2x * np.cos(2 * gamma_x_matrix)  + a1x * np.sin(gamma_x_matrix))
#     # free_energy += np.sum(- b1y * np.cos(gamma_y_matrix) - 0.5 * b2y * np.cos(2 * gamma_y_matrix)  + a1y * np.sin(gamma_y_matrix))
                          
#     return free_energy


n_calls = 0


def phi_vector_to_phi_matrix(phi_vector):
    phi = np.reshape(phi_vector[:-1], (Nx, Ny))
    return (phi, phi_vector[-1])

def phi_matrix_to_phi_vector(phi, phi_L):
    r = np.zeros((Nx * Ny + 1))
    r[:-1] = np.reshape(phi, Nx * Ny)
    r[:1] = phi_L
    return r

def cost_function(phi, phi_L, I_DC, A_x_matrix, A_y_matrix):
        gamma_x, gamma_y = gamma_matrices(phi, phi_L, A_x_matrix, A_y_matrix)
        x_current, y_current = current_phase_relation(gamma_x, gamma_y)
        # internal Kirchoff Law:
        div_I = gamma_x[:-1,:] - gamma_x[1:,:] # div I has size (Nx, Ny) 
        div_I[:,1:] += gamma_y
        div_I[:,:-1] -= gamma_y
        I_left_bar = np.sum(x_current[0,:])
      #  print("I_left_bar: ", I_left_bar)
        return np.linalg.norm(div_I) + (I_left_bar - I_DC)**2

def optimize_jja(phi_start, phi_L_start, I_DC, A_x_matrix, A_y_matrix, *args, **kwargs):
    
    bounds = np.array([[-10, 10]])
    bounds = np.repeat(bounds, Nx * Ny + 1, axis=0)
    
    def f(phi_vector):
        phi, phi_L = phi_vector_to_phi_matrix(phi_vector)
        return cost_function(phi, phi_L, I_DC, A_x_matrix, A_y_matrix)
    
    x0 = phi_matrix_to_phi_vector(phi_start, phi_L_start)
    return scipy.optimize.minimize(f, x0, method='CG',  options={'norm': 2,  'maxiter': 1000})
    
# include factor -2e / hbar in vector potential
def gen_vector_potential(Nx, Ny, f):
    # Nx x Ny JJA
    # f: f = psi / psi_0 flux per palette / magnetic_flux_quantum
    A_x = np.linspace(0, -Ny * f * const_flux, Ny)
    A_x = np.tile(A_x, (Nx+1, 1))
    A_y = np.zeros((Nx, Ny-1))
    return(-2 * const_e / const_hbar * A_x, -2 * const_e / const_hbar  * A_y)
    

A_x, A_y = gen_vector_potential(Nx, Ny, f)
t_start = time.time()
I_DC = 5
I_JJ = I_DC*0.9 / Ny
delta_phi_approx = np.arcsin(I_JJ)
print("delta phi approx = ", delta_phi_approx)
phi_start = np.linspace(-delta_phi_approx * Nx, -delta_phi_approx, Nx)
phi_start = np.tile(phi_start, (Ny, 1)).T
print(phi_start.shape)
phi_L_start = -delta_phi_approx * (Nx + 1)
print("phi start: ", phi_start)
print("phi_L start: ", phi_L_start)
#phi_start = np.zeros((Nx, Ny))
#phi_L_start = 0
print("cost function of start: ", cost_function(phi_start, phi_L_start, I_DC, A_x, A_y))

res = optimize_jja(phi_start, phi_L_start, I_DC, A_x, A_y)
print(res)
total_time = time.time() - t_start
# print(res.x)


phi_matrix, phi_L = phi_vector_to_phi_matrix(res.x)
print("phi: ", phi_matrix)
print("phi_L: ", phi_L)

#
# calculate gammas/currents
#
gamma_x, gamma_y = gamma_matrices(phi_matrix, phi_L, A_x, A_y)
print("gamma x: ", gamma_x)
print("gamma y: ", gamma_y)
x_currents, y_currents = current_phase_relation(gamma_x, gamma_y)
print("x_currents: ", x_currents)
print("y_currents: ", y_currents)

island_x_coords, island_y_coords = np.meshgrid(np.arange(Nx+2), np.arange(Ny), indexing="ij")

x_current_xcoords, x_current_ycoords = np.meshgrid(np.arange(Nx+1), np.arange(Ny), indexing="ij")

x_current_xcoords = x_current_xcoords.astype('float64')
x_current_ycoords = x_current_ycoords.astype('float64')

x_current_xcoords += 0.5


y_current_xcoords, y_current_ycoords = np.meshgrid(np.arange(Nx), np.arange(Ny-1), indexing="ij")

y_current_xcoords = y_current_xcoords.astype('float64')
y_current_ycoords = y_current_ycoords.astype('float64')

y_current_ycoords += 0.5


#
# save matrices as txt
#
print(x_currents.shape)
print(x_current_xcoords.shape)
print(x_current_ycoords.shape)

print(y_currents.shape)
print(y_current_xcoords.shape)
print(y_current_ycoords.shape)

save_matrix_as_txt(phi_matrix, data_folder + "/phi.txt")

save_matrix_as_txt(A_x, data_folder + "/A_x.txt")
save_matrix_as_txt(A_y, data_folder + "/A_y.txt")

save_matrix_as_txt(gamma_x, data_folder + "/gamma_x.txt")
save_matrix_as_txt(gamma_y, data_folder + "/gamma_y.txt")

save_matrix_as_txt(x_currents, data_folder + "/current_x.txt")
save_matrix_as_txt(y_currents, data_folder + "/current_y.txt")

save_matrix_as_txt(island_x_coords, data_folder + "/island_x.txt")
save_matrix_as_txt(island_y_coords, data_folder + "/island_y.txt")

save_matrix_as_txt(x_current_xcoords, data_folder + "/current_x_coords_x.txt")
save_matrix_as_txt(x_current_ycoords, data_folder + "/current_x_coords_y.txt")
save_matrix_as_txt(y_current_xcoords, data_folder + "/current_y_coords_x.txt")
save_matrix_as_txt(y_current_ycoords, data_folder + "/current_y_coords_y.txt")


# Save metadata

with open(data_folder + '/runtime_stats.json', 'w') as outfile:
    json.dump(run_stats, outfile)
    outfile.write("\n")

with open(data_folder + '/args.json', 'w') as outfile:
    json.dump(vars(parser_args), outfile)
    outfile.write("\n")

with open(data_folder + '/args.txt', 'w') as outfile:
    outfile.write(str(sys.argv))
    outfile.write("\n")

