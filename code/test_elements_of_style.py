#!/usr/bin/env python
import console
import conversion
import numpy as np
import sst
import ipdb
from skimage.morphology import dilation

# a test of what we could get if we perfectly matched each element of style

test_content_file = "sample/rolling_in_the_deep/content.mp3"
test_style_file = "sample/rolling_in_the_deep/reference_stylized.mp3"

# Load them both as spectrograms
console.time("preprocessing")
content_audio, content_sample_rate = conversion.file_to_audio(test_content_file)
content_amplitude, content_phase = conversion.audio_to_spectrogram(content_audio, fft_window_size=1536)
style_audio, style_sample_rate = conversion.file_to_audio(test_style_file)
style_amplitude, style_phase = conversion.audio_to_spectrogram(style_audio, fft_window_size=1536)
console.timeEnd("preprocessing")

stylized_amplitude = np.zeros(content_amplitude.shape)

num_freq, num_timesteps, _ = content_amplitude.shape
num_timesteps = min(num_timesteps, style_amplitude.shape[1])

# Preprocessing - compute fundamentals and harmonics
console.time("super resolution")
content_fundamental_mask = sst.extract_fundamental(content_amplitude)
content_fundamental_freqs, content_fundamental_amps = sst.extract_fundamental_freqs_amps(content_fundamental_mask, content_amplitude)
content_sibilants = sst.get_sibilants(content_amplitude, content_fundamental_amps)
conversion.image_to_file(content_sibilants, test_content_file + ".sibilants.jpg")
console.log("finished sibilants")
content_harmonics = sst.fundamental_to_harmonics(content_fundamental_freqs, content_fundamental_amps, content_amplitude)
content_harmonics = dilation(content_harmonics)

content_sibilants *= content_amplitude.max() / content_sibilants.max()
console.stats(content_sibilants, "content sibilants")
content_harmonics *= content_amplitude.max() / content_harmonics.max()
console.stats(content_harmonics, "content harmonics")
console.timeEnd("super resolution")

console.time("frequency weighting")
# ELEMENT 1: Frequency weighting
for t in range(num_timesteps):
    content_slice = np.maximum(content_amplitude[:, t], np.maximum(content_harmonics[:,t], content_sibilants[:,t]))
    style_slice = style_amplitude[:, t, :]
    content_env = sst.spectral_envelope(content_slice)
    style_env = sst.spectral_envelope(style_slice)
    weights = np.clip(style_env / (0.001 + content_env), 0, 5)
    stylized_amplitude[:, t, :] = content_slice * weights[:, np.newaxis]
    # amplitude correction
    stylized_amplitude[:, t, :] *= np.clip(content_amplitude[:,t].max()/(stylized_amplitude[:, t, :].max() + 0.001), 0, 10)

console.timeEnd("frequency weighting")
stylized_audio = conversion.amplitude_to_audio(stylized_amplitude, fft_window_size=1536, phase_iterations=1, phase=content_phase)
conversion.image_to_file(stylized_amplitude, test_content_file + ".stylized-cheat.jpg")
conversion.audio_to_file(stylized_audio, test_content_file + ".stylized-cheat.mp3")
