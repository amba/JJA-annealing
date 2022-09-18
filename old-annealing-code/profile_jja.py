#!/usr/bin/env python3

import subprocess
import sys
args = sys.argv
jja_args = ['./jja.py'] + args[1:]

print("starting profile")
subprocess.call(['python3', '-m', 'cProfile', '-o', '/tmp/profile_data.pyprof'] + jja_args)
print("converting profile data to kcachegrind format")
subprocess.call(['pyprof2calltree', '-i', '/tmp/profile_data.pyprof', '-k'])
