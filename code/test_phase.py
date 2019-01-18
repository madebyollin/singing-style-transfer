#!/usr/bin/env python
import console
import conversion
import sst
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import ipdb

from sst import extract_fundamental

test_files = ["sample/rolling_in_the_deep/reference_stylized.mp3"]

for f in test_files:
    console.log("starting", f)
    audio, sample_rate = conversion.file_to_audio(f)
    amplitude, phase = conversion.audio_to_spectrogram(audio,fft_window_size=1536)
    console.stats(phase)
    io.imsave("phase.png", plt.get_cmap("plasma")((np.clip(phase[:,:,0],-2,2) + 2) / 4))
