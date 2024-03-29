#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=3, linewidth=300)
import subprocess
import argparse
import json
import sys
import matplotlib.pyplot as plt
import pathlib

from os.path import join

parser = argparse.ArgumentParser()

parser.add_argument('data_folder', help="data folder with optimization output files")
parser.add_argument('-c', '--command', action='append', default=[], help=" transformation command, see below; can be provided multiple times")
parser.add_argument('--cmap', help="cmap for magnetization plot (default: gray)", default='gray')
parser.add_argument('-v', '--pdf-viewer', help="open output pdf files with viewer", default='okular')

args = parser.parse_args()


data_folder = args.data_folder
pdf_viewer = args.pdf_viewer
magnetization_cmap = args.cmap

print("opening data folder: ", data_folder)

plot_pdf_gamma = '/tmp/gamma.pdf'
plot_pdf_phi = '/tmp/phi.pdf'
plot_pdf_current = '/tmp/current.pdf'
plot_pdf_magnetization = '/tmp/magnetization.pdf'

phi_matrix = np.loadtxt(join(data_folder, 'phi.txt'))

x_currents = np.loadtxt(join(data_folder, 'current_x.txt'))
y_currents = np.loadtxt(join(data_folder, 'current_y.txt'))

gamma_x = np.loadtxt(join(data_folder, 'gamma_x.txt'))
gamma_y = np.loadtxt(join(data_folder, 'gamma_y.txt'))


gamma_x = gamma_x - 2*np.pi * np.rint(gamma_x / (2 * np.pi))
gamma_y = gamma_y - 2*np.pi * np.rint(gamma_y / (2 * np.pi))

island_x_coords = np.loadtxt(join(data_folder, 'island_x.txt'))
island_y_coords = np.loadtxt(join(data_folder, 'island_y.txt'))

Nx = island_x_coords.shape[0]
Ny = island_x_coords.shape[1]

x_current_xcoords = np.loadtxt(join(data_folder, 'current_x_coords_x.txt'))
x_current_ycoords = np.loadtxt(join(data_folder, 'current_x_coords_y.txt'))

y_current_xcoords = np.loadtxt(join(data_folder, 'current_y_coords_x.txt'))
y_current_ycoords = np.loadtxt(join(data_folder, 'current_y_coords_y.txt'))


with open(join(data_folder, 'args.json'), 'r') as infile:
    input_args = json.load(infile)

f = input_args['frustration']
# maxiter = input_args['maxiter']
# initial_temp = input_args['temp']
# visit = input_args['visit']


# def magnetization_discrete_biot_savart(x_current, y_current, x_current_xcoords, x_current_ycoords, y_current_ycoords, y_current_xcoords):
#     # Approximate magnetic field created by JJA currents
    
#     # Consider current vector in the middle of each junction
#     # Calculate magnetic field in center of each palette using a discrete
#     # Biot-Savart law

#     magnetization = np.zeros((Nx - 1, Ny -1))

#     palette_coords_x, palette_coords_y = np.meshgrid(np.arange(Nx-1), np.arange(Ny-1), indexing="ij")
#     palette_coords = np.stack([palette_coords_x, palette_coords_y], axis=-1).astype('float64')
#     palette_coords += np.array([.5, .5])
    
#     x_current_coords = np.stack([x_current_xcoords, x_current_ycoords], axis=-1)
#     y_current_coords = np.stack([y_current_xcoords, y_current_ycoords], axis=-1)
#     for i in range(Nx-1): # x index
#         for j in range(Ny-1): # y index
#             # palette center coordinates
#             pos = palette_coords[i,j]
            
#             # cross_product(e_x, v) = v_y
#             magnetization[i,j] += np.sum(
#                 x_current * (x_current_coords - pos)[:,:,1] / np.linalg.norm(x_current_coords - pos, axis=-1)**3)
#             # cross_product(e_y, v) = -v_x
#             magnetization[i,j] += np.sum(
#                 y_current * -(y_current_coords - pos)[:,:,0] / np.linalg.norm(y_current_coords - pos, axis=-1)**3)

#     return -magnetization




# magnetization = magnetization_discrete_biot_savart(x_currents, y_currents, x_current_xcoords, x_current_ycoords, y_current_ycoords, y_current_xcoords)
    


# def apply_commands(commands, z_label, magnetization):
#     for cmd in (commands):
#         magnetization, z_label = apply_command(cmd, z_label, magnetization)
#     return magnetization, z_label

# def apply_command(cmd, z_label, magnetization):
#     print("apply command ", cmd)
#     if cmd in 'abs log log10'.split():
#         magnetization = getattr(np, cmd)(magnetization)
#         z_label = cmd + '(' + z_label + ')'
#     elif cmd.startswith('min='):
#         tmp,value = cmd.split('=')
#         value = float(value)
#         magnetization = np.clip(magnetization, value, None)
#     elif cmd.startswith('max='):
#         tmp,value = cmd.split('=')
#         value = float(value)
#         magnetization = np.clip(magnetization, None, value)
#     elif cmd.startswith('add='):
#         tmp,value = cmd.split('=')
#         value = float(value)
#         magnetization = magnetization + value
#         z_label = z_label + ("%g" % value)
#     elif cmd.startswith('factor='):
#         tmp,value = cmd.split('=')
#         value = float(value)
#         magnetization = value * magnetization
#         z_label = ("%g • " % value) + z_label
#     else:
#         sys.exit("unknown command " + cmd)
#     return magnetization, z_label

# z_label = 'B-field'

# magnetization, z_label = apply_commands(args.command, z_label, magnetization)



#
# generate pdf plot
#

plt.axes().set_aspect('equal')
plt.quiver(x_current_xcoords, x_current_ycoords,
           x_currents, np.zeros(x_currents.shape),
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))

plt.quiver(y_current_xcoords, y_current_ycoords,
           np.zeros(y_currents.shape), y_currents,
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))

plt.scatter(island_x_coords, island_y_coords, marker='s', c='b', s=5)
title = "f = Φ / a² = %.3g\n" % (
    f,)
plt.title(title)
plt.savefig(plot_pdf_current)

plt.clf()
plt.axes().set_aspect('equal')
plt.quiver(x_current_xcoords, x_current_ycoords,
           gamma_x, np.zeros(gamma_x.shape),
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))

plt.quiver(y_current_xcoords, y_current_ycoords,
           np.zeros(gamma_y.shape), gamma_y,
           pivot='mid', units='width', scale=2*Nx, width=1/(20*Nx))

plt.scatter(island_x_coords, island_y_coords, marker='s', c='b', s=5)
title = "f = Φ / a² = %.3g" % (
    f,  )
plt.title(title)
plt.savefig(plot_pdf_gamma)


#
# plot value of ring current "magnetization" for each loop
#

# plt.clf()
# plt.title(title)
# # show x-axis left to right, y-axis bottom to top
# magnetization = np.flip(magnetization, axis=1) # imshow plots the first axis top to bottom
# magnetization = np.swapaxes(magnetization, 0, 1)

# plt.imshow(magnetization, aspect='equal', cmap=magnetization_cmap)
# plt.colorbar(format="%.1f", label=z_label)
# plt.savefig(plot_pdf_magnetization)

plt.clf()
plt.title(title)

phi_matrix = np.flip(phi_matrix, axis=1)
phi_matrix = np.swapaxes(phi_matrix, 0, 1)

plt.imshow(phi_matrix, aspect='equal', cmap=magnetization_cmap)
plt.colorbar(format="%.1f", label='φ')
plt.savefig(plot_pdf_phi)

if pdf_viewer:
    subprocess.run([pdf_viewer, plot_pdf_gamma, plot_pdf_current, plot_pdf_phi])
