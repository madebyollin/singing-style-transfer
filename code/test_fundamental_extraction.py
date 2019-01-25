#!/usr/bin/env python
import console
import conversion
import numpy as np
import sst
import ipdb

#test_files = ["sample/rolling_in_the_deep/reference_stylized.mp3"]
test_files = ["/Users/ollin/Desktop/ity.mp3"]

for f in test_files:
    console.time("preprocessing")
    console.log("starting", f)
    audio, sample_rate = conversion.file_to_audio(f)
    amplitude, phase = conversion.audio_to_spectrogram(audio,fft_window_size=1536)
    console.timeEnd("preprocessing")
    console.time("extracting fundamental")
    fundamental_mask = sst.extract_fundamental(amplitude)
    console.timeEnd("extracting fundamental")
    conversion.image_to_file(fundamental_mask, f + ".fundamental.png")

    console.time("fundamental to harmonics")
    fundamental_freqs, fundamental_amps = sst.extract_fundamental_freqs_amps(fundamental_mask, amplitude)
    harmonics = sst.fundamental_to_harmonics(fundamental_freqs, fundamental_amps, amplitude)
    console.timeEnd("fundamental to harmonics")
    conversion.image_to_file(harmonics, f + ".harmonics.png")

    # pitch normalization haha
    if True:
        pitch_normalized_amp = np.zeros(amplitude.shape) 
        pitch_normalized_phase = np.zeros(phase.shape)
        BASE_PITCH = 20
        console.stats(amplitude, "original_amp")
        for t in range(len(fundamental_freqs)): 
            amp_column = amplitude[:,t]
            phase_column = phase[:,t]
            if fundamental_freqs[t] != 0 and fundamental_amps[t] != 0:
                scaling_factor = BASE_PITCH / fundamental_freqs[t] 
            else:
                scaling_factor = 1
            orig_height = len(amp_column)
            scaled_height = int(orig_height * scaling_factor)
            slice_height = min(orig_height, scaled_height)
            for dst_f in range(slice_height):
                # n^2 nearest neighbor since ndimage scaling was even slower
                src_f = int(dst_f / scaling_factor)
                pitch_normalized_amp[dst_f,t,:] = amplitude[src_f,t,:]
                pitch_normalized_phase[dst_f,t,:] = phase[src_f,t,:]

            amp_column_scaled = pitch_normalized_amp[:,t]
            weights = np.clip(sst.spectral_envelope(amp_column) / (sst.spectral_envelope(amp_column_scaled) + 0.01), 0, 10)
            pitch_normalized_amp[:,t] = (amp_column_scaled.T * weights).T
            #scaled_amp_column = imresize(amp_column, (scaled_height, 1))
            #scaled_phase_column = imresize(phase_column, (scaled_height, 1))
            #pitch_normalized_amp[:slice_height,t,:] = scaled_amp_column[:slice_height]
            #pitch_normalized_phase[:slice_height,t,:] = scaled_phase_column[:slice_height]
        conversion.image_to_file(pitch_normalized_amp, f + ".pitch_normalized.png")
        console.stats(pitch_normalized_amp, "pitch_normalized_amp")
        pitch_normalized_audio = conversion.amplitude_to_audio(pitch_normalized_amp, fft_window_size=1536, phase_iterations=1, phase=phase)
        conversion.audio_to_file(pitch_normalized_audio, f + ".pitch_normalized.mp3")

    fundamental_audio = conversion.amplitude_to_audio(fundamental_mask, fft_window_size=1536, phase_iterations=1, phase=phase)
    conversion.audio_to_file(fundamental_audio, f + ".fundamental.mp3")

    harmonics_audio = conversion.amplitude_to_audio(harmonics, fft_window_size=1536, phase_iterations=1, phase=phase)
    conversion.audio_to_file(harmonics_audio, f + ".harmonics.mp3")
    console.log("finished", f)
