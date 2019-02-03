#!/usr/bin/env python
import console
import conversion
import numpy as np
import sst
import ipdb

<<<<<<< HEAD
test_files = ["../data/aligned/one_last_time/one_last_time_cover_aligned_30s.mp3", "../data/aligned/one_last_time/one_last_time_original_30s.mp3"]
# test_files = ["/Users/ollin/Desktop/ity.mp3"]
=======
#test_files = ["sample/rolling_in_the_deep/style.mp3"]
test_files = ["/Users/ollin/Desktop/test.mp3"]
>>>>>>> fcac82dc4fdc4518f7dcacefca7dad6c1922790b

for f in test_files:
    console.time("preprocessing")
    console.log("starting", f)
    audio, sample_rate = conversion.file_to_audio(f)
    amplitude, phase = conversion.audio_to_spectrogram(audio, fft_window_size=1536)
    console.timeEnd("preprocessing")
    console.time("extracting fundamental")
    fundamental_mask = sst.extract_fundamental(amplitude)
    console.timeEnd("extracting fundamental")
    conversion.image_to_file(fundamental_mask, f + ".fundamental.png")

    console.time("fundamental to harmonics")
    fundamental_freqs, fundamental_amps = sst.extract_fundamental_freqs_amps(
        fundamental_mask, amplitude
    )
    harmonics = sst.fundamental_to_harmonics(fundamental_freqs, fundamental_amps, amplitude)
    console.timeEnd("fundamental to harmonics")
    conversion.image_to_file(harmonics, f + ".harmonics.png")

    # pitch normalization haha
    if True:
        pitch_normalized_amp, pitch_normalized_phase = sst.normalize_pitch(
            amplitude, phase, fundamental_freqs, fundamental_amps
        )
        conversion.image_to_file(pitch_normalized_amp, f + ".pitch_normalized.png")
        console.stats(pitch_normalized_amp, "pitch_normalized_amp")
        pitch_normalized_audio = conversion.amplitude_to_audio(
            pitch_normalized_amp,
            fft_window_size=1536,
            phase_iterations=1,
            phase=pitch_normalized_phase,
        )
        conversion.audio_to_file(pitch_normalized_audio, f + ".pitch_normalized.mp3")

    fundamental_audio = conversion.amplitude_to_audio(
        fundamental_mask, fft_window_size=1536, phase_iterations=1, phase=phase
    )
    conversion.audio_to_file(fundamental_audio, f + ".fundamental.mp3")

    harmonics_audio = conversion.amplitude_to_audio(
        harmonics, fft_window_size=1536, phase_iterations=1, phase=phase
    )
    conversion.audio_to_file(harmonics_audio, f + ".harmonics.mp3")
    console.log("finished", f)
