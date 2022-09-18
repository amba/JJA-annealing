#!/usr/bin/env python3

import numpy as np
import numpy.random
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

# parser.add_argument("-f", "--frustration", help="magnetic flux quanta per palette (default: 0)",
#                     type=float,
#                     default=0)
# parser.add_argument('-o', "--output-dir", help="output folder (default: use date+time")
# parser.add_argument('-v', '--pdf-viewer', help="open output pdf files with viewer")

parser_args = parser.parse_args()

#
# JJA has size Nx, Ny
# this is the total size, including a possible frame/edge
#
# Nx_int, Ny_int are internal size, meaning that the phi_matrix given to the optimizer has size Nx_int x Ny_int



Nx = parser_args.Nx
Ny = parser_args.Ny

    




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

island_x_coords, island_y_coords = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")


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


def gamma_matrices(phi_matrix, A_x_matrix, A_y_matrix):
    # diff in x-direction
    gamma_x = np.zeros((Nx - 1, Ny))
    gamma_y = np.zeros((Nx - 2, Ny-1))
    gamma_x += A_x_matrix
                       
    # internal gammas
    gamma_x += phi_matrix[1:,:] - phi_matrix[:-1,:]

    
    # gamma_x= phi_matrix[1:,:] - phi_matrix[:-1,:] + A_x_matrix

    # diff in y-direction
    gamma_y += phi_matrix[1:-1,1:] - phi_matrix[1:-1,:-1] + A_y_matrix
    return (gamma_x, gamma_y)

## x direction: I(φ) = b1x sin(φ) + b2x sin(2φ) + a1x cos(φ)
# b1x = 1
# b2x = 0.4
# a1x = 0

# b1y = 1
# b2y = 0.4
# a1y = 0

tau = 0.95
anisotropy = 1.5
cos_param = 0.5


def current_phase_relation(gamma_x_matrix, gamma_y_matrix):
    x_currents = np.sin(gamma_x_matrix)
    y_currents = np.sin(gamma_y_matrix)
    # subtract cos_param to approximately zero the phi_0 effect
    # x_currents = np.sin(gamma_x_matrix - cos_param) / np.sqrt(1 - tau * np.sin((gamma_x_matrix - cos_param) / 2)**2)
    # x_currents += cos_param * np.cos(gamma_x_matrix)
    # y_currents = np.sin(gamma_y_matrix) / np.sqrt(1 - tau * np.sin(gamma_y_matrix / 2)**2)
    # x_currents = np.sin(gamma_x_matrix) / np.sqrt(1 - np.sin(gamma_x_matrix /2)**2)
    # y_currents =  np.sin(gamma_y_matrix) / np.sqrt(1 - np.sin(gamma_y_matrix /2)**2)

    
    return (x_currents, y_currents)

def free_energy(phi, A_x, A_y):
    gamma_x_matrix, gamma_y_matrix = gamma_matrices(phi, A_x, A_y)
    E = 0
    E += np.sum(1 - np.cos(gamma_x_matrix))
    E += np.sum(1 - np.cos(gamma_y_matrix))
    return E


def phi_vector_to_phi_matrix(phi_vector):
    return np.reshape(phi_vector, (Nx, Ny))

def phi_matrix_to_phi_vector(phi):
    return np.reshape(phi, Nx * Ny)

def cost_function(phi, I_DC, A_x_matrix, A_y_matrix):
        gamma_x, gamma_y = gamma_matrices(phi, A_x_matrix, A_y_matrix)
        x_current, y_current = current_phase_relation(gamma_x, gamma_y)
        # internal Kirchoff Law:
        # div I is calculated only for non-busbar islands
        # div I has size (Nx-2, Ny)
        # div I(island) = outgoing current - ingoing current
        div_I = np.zeros((Nx - 2, Ny))
        div_I += x_current[1:,:]
        div_I -= x_current[:-1,:]
        div_I[:,:-1] += y_current
        div_I[:,1:] -= y_current
        
        I_left_bar = np.sum(x_current[0,:])
        I_right_bar = np.sum(x_current[-1,:])
        
      #  print("I_left_bar: ", I_left_bar)
        return np.sum(np.abs(div_I)) + np.abs(I_left_bar - I_DC) + np.abs(I_right_bar - I_DC)


def optimize_jja(phi_start, I_DC, A_x_matrix, A_y_matrix, *args, **kwargs):
    
    bounds = np.array([[-10*np.pi, 10*np.pi]])
    bounds = np.repeat(bounds, Nx * Ny, axis=0)
    
    def f(phi_vector):
        phi = phi_vector_to_phi_matrix(phi_vector)
        return cost_function(phi, I_DC, A_x_matrix, A_y_matrix)
    
    x0 = phi_matrix_to_phi_vector(phi_start)
    return scipy.optimize.dual_annealing(f, x0=x0, maxiter=500, restart_temp_ratio=1e-12, initial_temp=1e-1, visit=2.5, bounds=bounds, no_local_search=True)
    
# include factor -2e / hbar in vector potential
def gen_vector_potential(f):
    # Nx x Ny JJA
    # f: f = psi / psi_0 flux per palette / magnetic_flux_quantum
    A_x = np.linspace(0, -Ny * f * const_flux, Ny) + Ny/2 * f * const_flux
    A_x = np.tile(A_x, (Nx-1, 1))
    A_y = np.zeros((Nx-2, Ny-1))
    return(-2 * const_e / const_hbar * A_x, -2 * const_e / const_hbar  * A_y)
    
def gen_phi0_current(I_DC, noise):
    I_JJ = I_DC / Ny
    delta_phi_approx = np.arcsin(I_JJ)
    phi_start = np.linspace(-delta_phi_approx * (Nx - 1), 0, Nx)
    phi_start = np.tile(phi_start, (Ny, 1)).T
    phi_start += noise * numpy.random.randn(Nx, Ny)
    return phi_start

def gen_phi0_vortex(x0, y0):
    phi_start = np.arctan2(island_y_coords - y0, island_x_coords - x0)
    return phi_start

def normalize_phase(phi):
    # normalize to (-np.pi, np.pi)
    phi = np.fmod(phi, 2 * np.pi)
    phi[phi > np.pi] -= 2*np.pi
    return phi

f = 1 / (Nx * Ny)
A_x, A_y = gen_vector_potential(f)
I_DC = 0

x0 = int(Nx/2) - 0.5
y0 = int(Ny/2) - 0.1

phi_start = gen_phi0_vortex(x0, y0)
phi_start += gen_phi0_current(I_DC, 0)
print("cost function of start: ", cost_function(phi_start, I_DC, A_x, A_y))
res = optimize_jja(phi_start, I_DC, A_x, A_y)
print("cost function after optimization = ", res.fun)
phi_matrix = phi_vector_to_phi_matrix(res.x)
F = free_energy(phi_matrix, A_x, A_y)
print("free energy: ", F)

# I_vals = np.linspace(0, Ny/5, 20)
# F_vals = []
# I_real_vals = []
# cost_function_vals = []

# for I_DC in I_vals:
#     phi_start = gen_phi0_current(I_DC, 0)
#     phi_start += gen_phi0_vortex(x0, y0)

#     gamma_x_start, gamma_y_start = gamma_matrices(phi_start, A_x, A_y)
#     print("cost function of start: ", cost_function(phi_start, I_DC, A_x, A_y))
#     res = optimize_jja(phi_start, I_DC, A_x, A_y)
#     print("cost function after optimization = ", res.fun)
#     cost_function_vals.append(res.fun)
    
#     phi_matrix = phi_vector_to_phi_matrix(res.x)
#     #
#     # calculate gammas/currents
#     #
#     gamma_x, gamma_y = gamma_matrices(phi_matrix, A_x, A_y)
#     x_current, y_current = current_phase_relation(gamma_x, gamma_y)
#     I_real_left = np.sum(x_current[0,:])
#     I_real_right = np.sum(x_current[-1,:])
#     I_real = (I_real_left + I_real_right) / 2
#     F = free_energy(phi_matrix, A_x, A_y)
#     F_vals.append(F)
#     I_real_vals.append(I_real)
#     print("I_DC = %g" % (I_DC))
#     print("I_left = %g, I_right = %g" % (I_real_left, I_real_right))
#     print("F(I = %g) = %g" % (I_real, F))
#     print("------------------\n")
# F_vals = np.array(F_vals)

# plt.plot(I_real_vals, F_vals, 'x', label = "one vortex")
# plt.legend()
# plt.grid()
# plt.show()

# plt.clf()
# plt.plot(I_real_vals, cost_function_vals)
# plt.show()
# exit(1)
gamma_x, gamma_y = gamma_matrices(phi_matrix, A_x, A_y)
x_currents, y_currents = current_phase_relation(gamma_x, gamma_y)


island_x_coords, island_y_coords = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

x_current_xcoords, x_current_ycoords = np.meshgrid(np.arange(Nx-1), np.arange(Ny), indexing="ij")

x_current_xcoords = x_current_xcoords.astype('float64')
x_current_ycoords = x_current_ycoords.astype('float64')

x_current_xcoords += 0.5


y_current_xcoords, y_current_ycoords = np.meshgrid(np.arange(Nx-2), np.arange(Ny-1), indexing="ij")

y_current_xcoords = y_current_xcoords.astype('float64')
y_current_ycoords = y_current_ycoords.astype('float64')

y_current_xcoords += 1
y_current_ycoords += 0.5

plt.clf()
plt.title("gamma")
plt.axes().set_aspect('equal')
plt.quiver(x_current_xcoords, x_current_ycoords,
           normalize_phase(gamma_x), np.zeros(x_currents.shape),
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))
plt.quiver(y_current_xcoords, y_current_ycoords,
           np.zeros(y_currents.shape), normalize_phase(gamma_y),
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))
plt.scatter(island_x_coords, island_y_coords, marker='s', c='b', s=5)
plt.show()

plt.clf()
plt.title("current")
plt.axes().set_aspect('equal')
plt.quiver(x_current_xcoords, x_current_ycoords,
           x_currents, np.zeros(x_currents.shape),
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))
plt.quiver(y_current_xcoords, y_current_ycoords,
           np.zeros(y_currents.shape), y_currents,
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))
plt.scatter(island_x_coords, island_y_coords, marker='s', c='b', s=5)


plt.show()

plt.show()

plt.clf()

phi_matrix = np.flip(phi_matrix, axis=1)
phi_matrix = np.swapaxes(phi_matrix, 0, 1)
plt.title("phi")
plt.imshow(phi_matrix, aspect='equal', cmap='gray')
plt.colorbar(format="%.1f", label='φ')
plt.show()
import code
code.interact()

# output_dir = parser_args.output_dir
# pdf_viewer = parser_args.pdf_viewer



# date_string = datetime.now()
# date_string = date_string.strftime("%Y-%m-%d_%H-%M-%S")

# data_folder = date_string + "_JJA"

# for key, value in vars(parser_args).items():
#     if value is not None:
#         data_folder += "_" + key + "=" + str(value)
    
# print("data folder: ", data_folder)
# pathlib.Path(data_folder).mkdir()


# #
# # save matrices as txt
# #
# print(x_currents.shape)
# print(x_current_xcoords.shape)
# print(x_current_ycoords.shape)

# print(y_currents.shape)
# print(y_current_xcoords.shape)
# print(y_current_ycoords.shape)

# save_matrix_as_txt(phi_matrix, data_folder + "/phi.txt")

# save_matrix_as_txt(A_x, data_folder + "/A_x.txt")
# save_matrix_as_txt(A_y, data_folder + "/A_y.txt")

# save_matrix_as_txt(gamma_x, data_folder + "/gamma_x.txt")
# save_matrix_as_txt(gamma_y, data_folder + "/gamma_y.txt")

# save_matrix_as_txt(x_currents, data_folder + "/current_x.txt")
# save_matrix_as_txt(y_currents, data_folder + "/current_y.txt")

# save_matrix_as_txt(island_x_coords, data_folder + "/island_x.txt")
# save_matrix_as_txt(island_y_coords, data_folder + "/island_y.txt")

# save_matrix_as_txt(x_current_xcoords, data_folder + "/current_x_coords_x.txt")
# save_matrix_as_txt(x_current_ycoords, data_folder + "/current_x_coords_y.txt")
# save_matrix_as_txt(y_current_xcoords, data_folder + "/current_y_coords_x.txt")
# save_matrix_as_txt(y_current_ycoords, data_folder + "/current_y_coords_y.txt")


# # Save metadata

# with open(data_folder + '/runtime_stats.json', 'w') as outfile:
#     json.dump(run_stats, outfile)
#     outfile.write("\n")

# with open(data_folder + '/args.json', 'w') as outfile:
#     json.dump(vars(parser_args), outfile)
#     outfile.write("\n")

# with open(data_folder + '/args.txt', 'w') as outfile:
#     outfile.write(str(sys.argv))
#     outfile.write("\n")

