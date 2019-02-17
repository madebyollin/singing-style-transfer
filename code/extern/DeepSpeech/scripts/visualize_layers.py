#!/usr/bin/env python3

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

if '__main__' == __name__:
    inp_path = './'

    targets = ['cover', 'original', 'cover_pitch_normalized', 'original_pitch_normalized']
    for target in targets:
        layers = glob.glob(os.path.join(inp_path, target, '*.npy'))
        for l in layers:
            arr = np.load(l).T
            assert len(arr.shape) == 2

            plt.imshow(arr)
            plt.savefig(l + '.png')




