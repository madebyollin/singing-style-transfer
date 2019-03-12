#!/usr/bin/env python
"""
Generates spectrogram dataset for the post-processing network.

:returns
    - x: Noisy spectrogram
    - y: Reference Spectrogram

"""
import os
import numpy as np
import tensorflow as tf
# import ipdb

import console
import conversion
import skimage.io as io
import sst

from extern.DeepSpeech.util.config import initialize_globals 
from extern.DeepSpeech.util.flags import FLAGS, create_flags

# generates data to feed the post processor
def generate_data_arrs(file_path, slice_size_t=1536):
    audio, sr = conversion.file_to_audio(file_path)
    amplitude, phase = conversion.audio_to_spectrogram(audio, fft_window_size=1536)
    amplitude = amplitude[:,:2*slice_size_t]

    # clipping only the first part to minimize "easy" repeated audio
    content = amplitude[:,:slice_size_t]
    style   = amplitude[:,slice_size_t:2*slice_size_t]  
    freq_rang = [np.min(content, 1), np.max(content, 1)]

    console.log("Content shape", content.shape)
    console.log("Style shape", style.shape)
    # it's a lot of work to compute the x...
    fundamental_mask = sst.extract_fundamental(amplitude)
    #console.stats(fundamental_mask, "fundamental_mask")
    #console.stats(amplitude, "amplitude")
    fundamental_freqs, fundamental_amps = sst.extract_fundamental_freqs_amps(fundamental_mask, amplitude)
    content_fundamental_freqs = fundamental_freqs[:slice_size_t]
    content_fundamental_amps = fundamental_amps[:slice_size_t]
    style_fundamental_freqs = fundamental_freqs[slice_size_t:2*slice_size_t]
    # features are computed directly and then sliced
    features = sst.get_feature_array(file_path) / 5
    features = sst.resize(features, (2048, amplitude.shape[1]))
    content_features = features[:,:slice_size_t]
    style_features   = features[:,slice_size_t:2*slice_size_t]
    stylized = sst.audio_patch_match(
        content,
        style,
        content_fundamental_freqs,
        style_fundamental_freqs,
        content_features,
        style_features,
        iterations=10
    ) # Harmonic recovery
    content_harmonics = sst.fundamental_to_harmonics(content_fundamental_freqs, content_fundamental_amps, content)
    content_harmonics = sst.grey_dilation(content_harmonics, size=3)
    content_harmonics *= content.max() / content_harmonics.max()
    # Sibilant recovery
    content_sibilants = sst.get_sibilants(content, content_fundamental_amps)
    content_sibilants *= content.max() / content_sibilants.max()

    x_arr = np.dstack([np.mean(stylized,axis=2), np.mean(content_harmonics,axis=2), np.mean(content_sibilants, axis=2)])
    y_arr = np.mean(content, axis=2)
    style_arr = np.mean(style, axis=2)
    return 0, 0, style_arr
    # return x_arr, y_arr, style_arr

def main(_):
    initialize_globals()

    # scrape data from folder
    RAW_DATA_DIR = "../data/studio_acapellas"
    PROCESSED_DATA_DIR = "../data/processed"
    # for each one, generate the data using sst methods, and save the data
    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if file_path.endswith(".mp3"):
            processed_file_name = file_name.replace("mp3", "npy") # haha
            # todo: rewrite all this using pathlib
            processed_file_path_x = PROCESSED_DATA_DIR + "/x/" + processed_file_name
            processed_file_path_y = PROCESSED_DATA_DIR + "/y/" + processed_file_name
            console.h1("Processing", file_path)
            processed_file_path_style = PROCESSED_DATA_DIR + "/style/" + processed_file_name
            x_arr, y_arr, style_arr = generate_data_arrs(file_path)
            # for debugging just save as images
            console.stats(x_arr, "x_arr")
            console.stats(y_arr, "y_arr")
            console.stats(style_arr, "style_arr")
            #ipdb.set_trace()
            io.imsave(processed_file_path_x + ".jpg", x_arr / x_arr.max())
            io.imsave(processed_file_path_y + ".jpg", y_arr / y_arr.max())
            np.save(processed_file_path_x, x_arr) 
            np.save(processed_file_path_y, y_arr)
            np.save(processed_file_path_style, style_arr) 
        else:
            console.info("Skipping", file_path)

if __name__ == "__main__":
    create_flags()
    FLAGS.one_shot_infer = "/tmp/input.wav" 
    FLAGS.checkpoint_dir = "extern/DeepSpeech/deepspeech-0.4.1-checkpoint/" 
    FLAGS.alphabet_config_path = "extern/DeepSpeech/data/alphabet.txt"

    tf.app.run(main)
