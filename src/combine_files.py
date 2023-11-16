import numpy as np
import os
import sys

directory = sys.argv[1]

spectra = ["tt", "ee", "te", "pp"]

spectra_arrays = {}
for xx in spectra:
    spectra_arrays[xx]=[]

for xx in spectra:
    for fname in os.listdir(directory):
        if xx in fname:
            arr = np.loadtxt(os.path.join(directory, fname))
            spectra_arrays[xx].append(arr)

final_arrays = {}

for xx in spectra:
    stacked = np.vstack(spectra_arrays[xx])
    outname = "{}_hp".format(xx)
    np.save(outname, stacked)