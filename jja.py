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
parser.add_argument('-N', help="system size (mandatory)", type=int,
                    required=True)
parser.add_argument('-m', "--maxiter",
                    help="number of iterations (default: 1000)",
                    type=int, default=1000)
parser.add_argument('-t', "--temp",
                    help="initial temperature for annealing (default: 100e3)",
                    type=float, default=100e3)
parser.add_argument("--visit",
                    help="value for visit parameter for annealing (default: 2)",
                    type=float,
                    default=2)
parser.add_argument("--restart_temp_ratio",
                    help="restart annealing when temperature has fallen below initial_temp * restart_ratio. Resets temperature to initial_temp and uses a new random start state. Depending on maxiter, this can happen multiple times. Effectively the same as calling dual annealing multiple times and selecting the best annealing result (default: 1e-9, do not restart)",
                    type=float,
                    default=1e-9)
parser.add_argument("--ejx",
                    help="josephson energy E_J for junctions in x direction (default: 1)",
                    type=float,
                    default=1)
parser.add_argument("--ejy",
                    help="josephson energy E_J for junctions in y direction (default: 1)",
                    type=float,
                    default=1)
parser.add_argument("--phi0x",
                    help="phi-node φ_0 parameter in x-direction (rad, default: 0)",
                    type=float,
                    default=1)
parser.add_argument("-f", "--frustration", help="magnetic flux quanta per palette (default: 0)",
                    type=float,
                    default=0)
parser.add_argument('--trivial-frame', action='store_true', help="add frame islands with constant order parameter fixed to zero")
parser.add_argument('-o', "--output-dir", help="output folder (default: use date+time")
parser.add_argument('-v', '--pdf-viewer', help="open output pdf files with viewer")

parser_args = parser.parse_args()

N = parser_args.N
have_frame = parser_args.trivial_frame

if have_frame:
    N_total = N + 2
else:
    N_total = N
N_junctions = 2 * (N - 1) * N
f = parser_args.frustration
initial_temp = parser_args.temp
maxiter = parser_args.maxiter
restart_temp_ratio = parser_args.restart_temp_ratio
visit = parser_args.visit
EJx = parser_args.ejx
EJy = parser_args.ejy
phi0x = parser_args.phi0x
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


plot_pdf_current = data_folder + "/currents.pdf"
plot_pdf_magnetization = data_folder + "/magnetization.pdf"

run_stats = {}
# phi layout
#

# ^
# | y
#  --> x
#
# x is 0th-dimension, y is 1st dimension of numpy array

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
# gamma_x, A_x are (N-1) x N matrix, gamma_y, A_y are N x (N-1) matrix

#

#

#
# the real part (phi part) of gamma can get as large as +-4π, so the function of
# the free energy needs to be well-defined for these values.


def gamma_matrices(phi_matrix, A_x_matrix, A_y_matrix):
    # diff in x-direction
    gamma_x= phi_matrix[1:,:] - phi_matrix[:-1,:] + A_x_matrix
    # diff in y-direction
    gamma_y = phi_matrix[:,1:] - phi_matrix[:,:-1] + A_y_matrix
    return (gamma_x, gamma_y)


def current_phase_relation(gamma_x_matrix, gamma_y_matrix, E_J_x_matrix, E_J_y_matrix):
    x_currents = E_J_x_matrix * np.sin(gamma_x_matrix)
    y_currents = E_J_y_matrix * np.sin(gamma_y_matrix)
    return (x_currents, y_currents)

def free_energy(gamma_x_matrix, gamma_y_matrix, E_J_x_matrix, E_J_y_matrix):
    # f >= 0, f(γ = 0) = 0
    free_energy = 0
    free_energy += np.sum(E_J_x_matrix * (1 - np.cos(gamma_x_matrix)))
    free_energy += np.sum(E_J_y_matrix * (1 - np.cos(gamma_y_matrix)))
    return free_energy

# free_energy_func as free_energy function
n_calls = 0

def add_frame(inner_phi):
    Nx = inner_phi.shape[0] + 2
    Ny = inner_phi.shape[1] + 2
    phi_matrix = np.zeros((Nx, Ny))
    phi_matrix[1:-1,1:-1] = inner_phi
    return phi_matrix

def optimize_jja(E_J_x_matrix, E_J_y_matrix, A_x_matrix, A_y_matrix, *args, **kwargs):
    # Nx, Ny are number of islands including boundary islands, if present
    Nx = E_J_x_matrix.shape[0] + 1
    Ny = E_J_x_matrix.shape[1]
    
    bounds = np.array([[0, 2*np.pi]])
    if have_frame:
        bounds_length = (Nx - 2) * (Ny - 2)
    else:
        bounds_length = Nx * Ny
    bounds = np.repeat(bounds, bounds_length, axis=0)
    # f is function given to optimization function, like dual annealing
    def f(phi_vector):
        global n_calls
        n_calls += 1
        if have_frame:
            inner_phi = np.reshape(phi_vector, (Nx-2, Ny-2))
            phi_matrix = add_frame(inner_phi)
        else:
            phi_matrix = np.reshape(phi_vector, (Nx, Ny))
        
        gamma_x = phi_matrix[1:,:] - phi_matrix[:-1,:] + A_x_matrix
        gamma_y = phi_matrix[:,1:] - phi_matrix[:,:-1] + A_y_matrix
        n_calls += 1
        return free_energy(gamma_x, gamma_y, E_J_x_matrix, E_J_y_matrix)

    return scipy.optimize.dual_annealing(f, bounds=bounds, *args, **kwargs)

    
# include factor -2e / hbar in vector potential
def gen_vector_potential(N, f):
    # N x N JJA
    # f: f = psi / psi_0 flux per palette / magnetic_flux_quantum
    A_x = np.linspace(0, -N * f * const_flux, N)
    A_x = np.tile(A_x, (N-1, 1))
    A_y = np.zeros((N, N-1))
    return(-2 * const_e / const_hbar * A_x, -2 * const_e / const_hbar  * A_y)
    

A_x, A_y = gen_vector_potential(N_total, f)
A_x += phi0x

if have_frame:
    A_x[:,0] = 0
    A_x[:,-1] = 0
    A_y[0,:] = 0
    A_y[-1,:] = 0
    
E_J_x = EJx * np.ones((N_total-1, N_total))
E_J_y = EJy * np.ones((N_total, N_total-1))




t_start = time.time()
res = optimize_jja(E_J_x, E_J_y, A_x, A_y, maxiter=maxiter, initial_temp = initial_temp, visit=visit, restart_temp_ratio=restart_temp_ratio, maxfun=1e12)
total_time = time.time() - t_start

run_stats['free_energy'] = res.fun
run_stats['success'] = res.success
run_stats['message'] = res.message
run_stats['free_energy_per_junction'] = res.fun / N_junctions

run_stats['total_time'] = total_time
run_stats['maxiter'] = maxiter
run_stats['n_free_energy_calls'] = n_calls
run_stats['time_per_call'] = total_time / n_calls
run_stats['time_per_call_per_junction'] = total_time / (n_calls * N_junctions)


# comparison with trivial state
if not have_frame:
    phi0 = np.zeros((N,N))
    gamma_x0, gamma_y0 = gamma_matrices(phi0, A_x, A_y)
    free_energy = free_energy(gamma_x0, gamma_y0, E_J_x, E_J_y)

    run_stats['free_energy_of_trivial_state'] = free_energy
    run_stats['free_energy_of_trivial_state_per_junction'] = free_energy / N_junctions

print(run_stats)


phi_matrix = np.reshape(res.x, (N, N))
if have_frame:
    phi_matrix = add_frame(phi_matrix)
    

#
# calculate gammas/currents
#

gamma_x, gamma_y = gamma_matrices(phi_matrix, A_x, A_y)
x_currents, y_currents = current_phase_relation(gamma_x, gamma_y, E_J_x, E_J_y)


island_x_coords, island_y_coords = np.meshgrid(np.arange(N_total), np.arange(N_total), indexing="ij")

x_current_xcoords, x_current_ycoords = np.meshgrid(np.arange(N_total-1), np.arange(N_total), indexing="ij")

x_current_xcoords = x_current_xcoords.astype('float64')
x_current_ycoords = x_current_ycoords.astype('float64')

x_current_xcoords += 0.5


y_current_xcoords, y_current_ycoords = np.meshgrid(np.arange(N_total), np.arange(N_total-1), indexing="ij")

y_current_xcoords = y_current_xcoords.astype('float64')
y_current_ycoords = y_current_ycoords.astype('float64')

y_current_ycoords += 0.5

#
# calculate magnetization of palette: Sum of currents circling the palette
#

magnetization = x_currents[:,:-1] - x_currents[:,1:] + y_currents[1:,:] - y_currents[:-1,:]




#
# save matrices as txt
#

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
save_matrix_as_txt(magnetization, data_folder + "/magnetization.txt")


with open(data_folder + '/runtime_stats.json', 'w') as outfile:
    json.dump(run_stats, outfile)
    outfile.write("\n")

with open(data_folder + '/args.json', 'w') as outfile:
    json.dump(vars(parser_args), outfile)
    outfile.write("\n")

with open(data_folder + '/args.txt', 'w') as outfile:
    outfile.write(str(sys.argv))
    outfile.write("\n")

#
# generate pdf plot
#

plt.axes().set_aspect('equal')
plt.quiver(x_current_xcoords, x_current_ycoords,
           x_currents, np.zeros(x_currents.shape),
           pivot='mid', units='width', scale=2*N, width=1/(20*N))

plt.quiver(y_current_xcoords, y_current_ycoords,
           np.zeros(y_currents.shape), y_currents,
           pivot='mid', units='width', scale=2*N, width=1/(20*N))

plt.scatter(island_x_coords, island_y_coords, marker='s', c='b', s=5)
title = "f = Φ / a² = %.3g, EJ_x = %.1g, EJ_y = %.1g, φ_0x = %.2g\nmaxiter = %d, T_0 = %.2g, visit = %.3g" % (
    f, EJx, EJy, phi0x, maxiter, initial_temp, visit)
plt.title(title)
plt.savefig(plot_pdf_current)


#
# plot value of ring current "magnetization" for each loop
#

plt.clf()
plt.title(title)

cmap = 'gray'
# show x-axis left to right, y-axis bottom to top
magnetization = np.flip(magnetization, axis=1) # imshow plots the first axis top to bottom
magnetization = np.swapaxes(magnetization, 0, 1)

plt.imshow(magnetization, aspect='auto', cmap=cmap)
plt.colorbar(format="%.1f", label="magnetization")
plt.savefig(plot_pdf_magnetization)

if pdf_viewer:
    subprocess.run([pdf_viewer, plot_pdf_current, plot_pdf_magnetization])
