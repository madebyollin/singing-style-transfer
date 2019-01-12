#!/usr/bin/env python
import console
import conversion
import numpy as np
import sys
import cv2

# hacky, will replace with argparse later
img_file_path = sys.argv[1]
console.log("sawifying", img_file_path)

spectrogram = conversion.file_to_image(img_file_path)
# non-maximum suppression since im lazy and don't wanna interpolate
output = np.zeros(spectrogram.shape)

# slow, will replace with numpy later
for y in range(64):
    console.progressBar((y + 1) / 64)
    for t in range(spectrogram.shape[1]):
        if spectrogram[y][t] > 0.1:
            harmonic = 1
            while harmonic * (y + 1) < spectrogram.shape[0]:
                center = harmonic * y
                end = int(harmonic * (y + 1))
                for i in range(center, end):
                    output[i][t] = spectrogram[y][t] * 1 / harmonic
                harmonic += 1

conversion.image_to_file(output, img_file_path + ".saw.png")
