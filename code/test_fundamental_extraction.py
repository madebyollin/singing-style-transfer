#!/usr/bin/env python
import console
import conversion
import sst

from sst import extract_fundamental

test_files = ["sample/rolling_in_the_deep/reference_stylized.mp3"]

for f in test_files:
    console.log("starting", f)
    audio, sample_rate = conversion.file_to_audio(f)
    amplitude, phase = conversion.audio_to_spectrogram(audio,fft_window_size=1536)
    fundamental = extract_fundamental(amplitude)
    conversion.image_to_file(fundamental, f + ".fundamental.png")
    fundamental_audio = conversion.amplitude_to_audio(fundamental, fft_window_size=1536, phase_iterations=1, phase=phase)
    conversion.audio_to_file(fundamental_audio, f + ".fundamental.mp3")
    console.log("finished", f)
