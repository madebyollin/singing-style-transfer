#!/usr/bin/env python
import scipy
import numpy as np

import conversion
import console

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
    stylized = global_eq_match(content, style)
    return stylized

def main():
    style_path = "sample/style.mp3"
    content_path = "sample/content.mp3"
    stylized_img_path = "sample/stylized.jpg"
    stylized_audio_path = "sample/stylized.mp3"

    # Read style audio to spectrograms.
    style_audio, style_sample_rate = conversion.file_to_audio(style_path)
    style_img, style_phase = conversion.audio_to_spectrogram(style_audio, fft_window_size=1536)

    # Read content audio to spectrograms.
    content_audio, content_sample_rate = conversion.file_to_audio(content_path)
    content_img, content_phase = conversion.audio_to_spectrogram(content_audio, fft_window_size=1536)

    stylized_img = stylize(content_img, style_img)
    stylized_audio = conversion.amplitude_to_audio(stylized_img, 1536)

    # Save stylized spectrogram and audio.
    conversion.image_to_file(stylized_img, stylized_img_path)
    conversion.audio_to_file(stylized_audio, stylized_audio_path)
    console.log("done! saved to sample")

if __name__ == "__main__":
    main()
