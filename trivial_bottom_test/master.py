#!/usr/bin/env python3

import multiprocessing
import subprocess
import numpy as np
boundary = 'trivial_bottom'

def slave(input):
    Nx = input['Nx']
    Ny = input['Ny']
    maxiter = input['maxiter']
    temp = input['temp']
    frustration = "%.5g" % input['frustration']
    visit = input['visit']
    
    EJx = input['EJx']
    EJy = input['EJy']
    phi0x = "%.5g" % input['phi0x']
    
    print("input dict: ", input)
    print("EJX, EJY: %g, %g" % (EJx, EJy))
    cmd = ['../jja.py',
           # annealing algorithm args
           '--temp', str(temp), '--maxiter', str(maxiter), '--visit', str(visit),
                     
           # JJA properties
           '-Nx', str(Nx), '-Ny', str(Ny), '--frustration', str(frustration), '--ejx', str(EJx), '--ejy', str(EJy),
           '--phi0x', str(phi0x), '--boundary', boundary
    ]
    
    subprocess.call(cmd)
    


if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    print("using %d jobs in parallel" % num_cores)

    dict = {'Nx': 50, 'Ny': 50, 'temp': 2000, 'maxiter': 2000, 'visit': 1.5, 'frustration': 0, 'EJx': 1, 'EJy': 1}
    
    args = []
    for phi0x in np.linspace(0, 1, 21):
        d = dict.copy()
        d['phi0x'] = phi0x
        print(d)
        args.append(d)
    with multiprocessing.Pool(num_cores) as p:
        p.map(slave, args)

    

    
