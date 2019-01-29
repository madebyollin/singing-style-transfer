#!/usr/bin/env python3

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

if '__main__' == __name__:
    inp_path = '/home/josephz/GoogleDrive/University/UW/2018-19/CSE481I/singing-style-transfer/src/extern/DeepSpeech/scripts'

    targets = ['cover', 'original']
    for target in targets:
        layers = glob.glob(os.path.join(inp_path, target, '*.npy'))
        for l in layers:
            arr = np.load(l)
            assert len(arr.shape) == 2

            plt.imshow(arr)
            plt.savefig(l + '.png')




