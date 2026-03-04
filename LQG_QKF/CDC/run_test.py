import os
import subprocess

info_gain_methods = ['heuristic', 'baseline', 'van_trees']
for info_gain_method in info_gain_methods:
    subprocess.run(['python', 'sensor_selection_sim-baseline.py', info_gain_method])