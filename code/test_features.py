#!/usr/bin/env python
import console
import conversion
import numpy as np
import sst
import ipdb

test_files = ["sample/rolling_in_the_deep/content.mp3", "sample/rolling_in_the_deep/reference_stylized.mp3"]

for f in test_files:
    console.time("preprocessing")
    console.log("starting", f)
    audio, sample_rate = conversion.file_to_audio(f)
    amplitude, phase = conversion.audio_to_spectrogram(audio, fft_window_size=1536)
    console.timeEnd("preprocessing")
    features = sst.compute_features(amplitude)
    conversion.image_to_file(features[:, :, np.newaxis], f + ".features.jpg")
