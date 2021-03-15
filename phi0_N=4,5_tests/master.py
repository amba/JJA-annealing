#!/usr/bin/env python3

import multiprocessing
import subprocess
import numpy as np
trivial_frame = True

def slave(input):
    N = input['N']
    maxiter = input['maxiter']
    temp = input['temp']
    frustration = input['frustration']
    visit = input['visit']
    
    EJx = input['EJx']
    EJy = input['EJy']
    phi0x = input['phi0x']
    
    print("input dict: ", input)
    print("EJX, EJY: %g, %g" % (EJx, EJy))
    cmd = ['../jja.py',
           # annealing algorithm args
           '--temp', str(temp), '--maxiter', str(maxiter), '--visit', str(visit),
                     
           # JJA properties
           '-N', str(N), '--frustration', str(frustration), '--ejx', str(EJx), '--ejy', str(EJy),
           '--phi0x', str(phi0x)
    ]
    if trivial_frame:
        cmd.append('--trivial-frame')
    
    subprocess.call(cmd)
    


if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    print("using %d jobs in parallel" % num_cores)

    dict = {'temp': 10000, 'maxiter': 2000, 'visit': 2, 'frustration': 0, 'EJx': 1, 'EJy': 1}
    
    args = []
    for N in (4, 5):
        for phi0x in np.linspace(0, 1, 10):
            d = dict.copy()
            d['N'] = N
            d['phi0x'] = phi0x
            print(d)
            args.append(d)
    with multiprocessing.Pool(num_cores) as p:
        p.map(slave, args)

    

    
