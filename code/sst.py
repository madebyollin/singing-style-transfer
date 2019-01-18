#!/usr/bin/env python
import scipy
import numpy as np
from scipy.ndimage.morphology import grey_dilation, grey_erosion

import conversion
import console
#import ipdb

def extract_fundamental(amplitude):
    fundamental = np.zeros(amplitude.shape)
    # TODO: replace all of this with real code or at least clean it up
    for t in range(amplitude.shape[1]):
        for f in range(4, 40):
            s = amplitude[f-2:f+3,t,0]
            if np.dot(np.array([-0.5, -0.5, 2, -.5, -.5]),s) > 0 and amplitude[f][t][0] + amplitude[f][max(t-1,0)][0] + fundamental[f][max(t-1,0)][0] > 0.5 * amplitude.max():
                for i in range(f-4,f+9):
                    fundamental[i][t] = amplitude[i][t] * (1 - abs(f - i) / 13)
                break
    # remove dots
    mask = (grey_dilation(grey_erosion(fundamental, structure=np.ones((5,5,1))), structure=np.ones((6,12,1))) > 0.1)
    conversion.image_to_file(mask, "mask_" + str(amplitude.shape) + ".png")
    fundamental *= mask
    return fundamental

def global_eq_match_2(content, style):
    content_mean_freq = np.mean(content, axis=(1, 2))
    style_mean_freq = np.mean(style, axis=(1, 2))

    # apply maximum filter
    content_mean_freq = scipy.ndimage.filters.maximum_filter1d(content_mean_freq, 50)
    style_mean_freq = scipy.ndimage.filters.maximum_filter1d(style_mean_freq, 50)

    weights = style_mean_freq / content_mean_freq
    weights = np.clip(weights, 0, 2)
    # conversion.image_to_file(np.broadcast_to(weights[:,np.newaxis,np.newaxis], content.shape), "sample/weights.png")
    # weights = scipy.ndimage.filters.gaussian_filter1d(weights, 10)

    stylized = (content.T * weights).T
    assert stylized.shape == content.shape
    return stylized / stylized.max()

def global_eq_match(content, style):
    """
    :param content:
    :param style:
    """
    content_mean_freq = np.mean(content, axis=(1, 2))
    style_mean_freq = np.mean(style, axis=(1, 2))

    weights = style_mean_freq / content_mean_freq
    weights = np.clip(weights, 0, 2)
    weights = scipy.ndimage.filters.gaussian_filter1d(weights, 10)

    stylized = (content.T * weights).T
    assert stylized.shape == content.shape
    return stylized

def stylize(content, style):
    stylized = content
    stylized = global_eq_match(content, style)
    #stylized = global_eq_match_2(content, style)
    return stylized

def main():
    sample_dir = "sample"
    sample_names = ["one_more_time", "rolling_in_the_deep"]
    for sample_name in sample_names:
        console.h1("Processing %s" % sample_name)
        sample_path = sample_dir + "/" + sample_name

        style_path = sample_path + "/style.mp3"
        content_path = sample_path + "/content.mp3"
        stylized_img_path = sample_path + "/stylized.png"
        stylized_audio_path = sample_path + "/stylized.mp3"

        # Read style audio to spectrograms.
        style_audio, style_sample_rate = conversion.file_to_audio(style_path)
        style_img, style_phase = conversion.audio_to_spectrogram(style_audio, fft_window_size=1536)

        # Read content audio to spectrograms.
        content_audio, content_sample_rate = conversion.file_to_audio(content_path)
        content_img, content_phase = conversion.audio_to_spectrogram(content_audio, fft_window_size=1536)

        # Pitch fundamental extraction (currently unused)
        fundamental_mask = extract_fundamental(content_img)

        stylized_img = stylize(content_img, style_img)
        stylized_audio = conversion.amplitude_to_audio(stylized_img, fft_window_size=1536, phase_iterations=1, phase=content_phase)

        # Save stylized spectrogram and audio.
        conversion.image_to_file(stylized_img, stylized_img_path)
        conversion.audio_to_file(stylized_audio, stylized_audio_path)
        console.info("Finished processing %s; saved to %s" % (sample_name, stylized_audio_path))

if __name__ == "__main__":
    main()
