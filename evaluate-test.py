#!/usr/bin/env python3

import numpy as np
import os
import argparse
import json
parser = argparse.ArgumentParser()

parser.add_argument('folder', help="top level folder with data folders")

args = parser.parse_args()

folder = args.folder
# open data folders

result_list = [] # list of run_stat dicts

data_folders = [ f.path for f in os.scandir(folder) if f.is_dir() ]


for data_folder in (data_folders):
    runtime_stats_json = os.path.join(data_folder, 'runtime_stats.json')
    args_json = os.path.join(data_folder, 'args.json')
    if os.path.isfile(runtime_stats_json) and os.path.isfile(args_json):
        with open(runtime_stats_json, 'r') as infile:
            run_stats = json.load(infile)
        with open(args_json, 'r') as infile:
            args = json.load(infile)
        result_list.append({**run_stats, **args})

result_list = sorted(result_list, key=lambda res: res['free_energy'])
for result in (result_list):
    T_start = result['temp']
    visit = result['visit']
    maxiter = result['maxiter']

    T_end = T_start * (2**(visit - 1) - 1) / (maxiter**(visit - 1) - 1)
    
    print("T0 = %8d,\tT_E = %.1f,\t\tvisit = %.3f,\t\tmaxiter = %10d:\t\t\tfun = %.3f"
          % (T_start, T_end, visit, maxiter, result['free_energy']))
    

        


