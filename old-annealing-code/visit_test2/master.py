#!/usr/bin/env python3

import multiprocessing
import subprocess
import numpy as np
trivial_frame = True

N = 50
phi0x = 0.3

def slave(input):
    N = input['N']
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
           '-N', str(N), '--frustration', str(frustration), '--ejx', str(EJx), '--ejy', str(EJy),
           '--phi0x', str(phi0x)
    ]
    if trivial_frame:
        cmd.append('--trivial-frame')
    
    subprocess.call(cmd)
    


if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    print("using %d jobs in parallel" % num_cores)

    dict = {'N': N, 'frustration': 0, 'EJx': 1, 'EJy': 1, 'phi0x': phi0x}

    
    args = []
    for maxiter in (2000, 4000, 6000):
        for T_start in (1000, 2000, 5000, 10000, 20000):
            for visit in (1.75, 1.5):
                d = dict.copy()
                d['temp'] = T_start
                d['visit'] = visit
                d['maxiter'] = maxiter
                args.append(d)
                print(d)
    with multiprocessing.Pool(num_cores) as p:
        p.map(slave, args)

    

    
