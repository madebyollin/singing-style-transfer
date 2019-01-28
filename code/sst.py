#!/usr/bin/env python
import scipy
import numpy as np
from scipy.ndimage.morphology import grey_dilation, grey_erosion

import conversion
import console
import cv2
import ipdb


def draw_harmonic_slice(spectrogram, t, f0, f1, alpha0, alpha1):
    # draw a 1px column at t corresponding to movement from f0 to f1
    # with brightness alpha
    # sigh
    width = 1
    # antialiasing function
    def sigma(distance):
        # return 1
        return np.clip(width / 2 - distance + 0.5, 0, 1)

    # make sure that f0 <= f1
    # if f1 < f0:
    #    tmp = f1
    #    f1 = f0
    #    f0 = tmp
    line_norm_squared = 1 + (f1 - f0) ** 2
    for n_p in range(int(abs(f1 - f0) + 2)):
        p = int(min(f1, f0)) + n_p
        if p >= spectrogram.shape[0]:
            continue
        dot_product = 1 + (f1 - f0) * (p - f0)
        distance_to_line = np.sqrt(abs(1 + (p - f0) ** 2 - (dot_product ** 2) / line_norm_squared))
        aa_alpha = sigma(distance_to_line)
        blending_alpha = np.clip(abs(dot_product / line_norm_squared), 0, 1)
        blended_alpha = (1 - blending_alpha) * alpha0 + blending_alpha * alpha1
        spectrogram[p, max(0, t - 1)] = (1 - aa_alpha) * blended_alpha
        spectrogram[p, t] = aa_alpha * blended_alpha


def extract_fundamental_freqs_amps(fundamental_mask, amplitude):
    # step one is to get a suitably accurate estimate of the fundamental pitch
    # the *good* way to do this is lots of subpixel sampling at each audible harmonic to extract the maximum amount of information
    # but that's a lot of work
    # so I'm doing it the lazy way for now
    # we want these two things for line drawing
    fundamental_freqs = np.zeros(fundamental_mask.shape[1])  # one per timestep
    fundamental_amps = np.zeros(fundamental_mask.shape[1])  # one per timestep

    # so lets first assume the fundamental is the amplitude-weighted average
    MAX_FREQ = 40
    coefficients = np.array(range(MAX_FREQ))
    fundamental_cropped = fundamental_mask[:MAX_FREQ, :, 0]
    denominators = np.sum(fundamental_cropped, axis=0)
    fundamental_amps = denominators[:] / np.max(denominators)
    fundamental_weighted = coefficients[:, np.newaxis] * fundamental_cropped
    console.stats(fundamental_weighted)
    # and compute it for every timestep
    for t in range(fundamental_mask.shape[1]):
        if denominators[t] == 0:
            fundamental_t = 0
        else:
            fundamental_t = np.sum(fundamental_weighted[:, t]) / denominators[t]
        # hack to make it continuous across gaps
        if t + 1 != fundamental_mask.shape[1]:
            fundamental_weighted[int(fundamental_t), t + 1] += 0.05
            denominators[t + 1] += 0.05
        for h in range(2, 20):
            # we want to extract information from the hth harmonic
            f_h_min = int((h - 0.5) * fundamental_t)
            f_h_max = int((h + 0.5) * fundamental_t)
            h_slice = amplitude[f_h_min:f_h_max, t]
            if h_slice.size == 0:
                break
            if h_slice.max() > 0.5 * fundamental_amps[t]:
                # this harmonic is actually present and reasonably loud
                # so let's use its pitch information instead
                coefficients = np.array(range(f_h_min, f_h_max))
                fundamental_h_weighted = coefficients[:, np.newaxis] * h_slice
                denominator_h = np.sum(h_slice)
                fundamental_h_t = fundamental_h_weighted.sum() / denominator_h * 1 / h
                if 0.8 * fundamental_t < fundamental_h_t < 1.2 * fundamental_t:
                    # it's recursive; the estimates get better
                    # as you increase in pitch
                    # and we weight higher frequencies more
                    fundamental_t = fundamental_h_t
        # fundamental_freqs[t] = fundamental_t
        # let's try smoothing it
        if (
            t - 1 < 0
            or fundamental_freqs[t - 1] == 0
            or abs(fundamental_t - fundamental_freqs[t - 1]) > 3
        ):
            fundamental_freqs[t] = fundamental_t
        else:
            # alpha = 0.5
            alpha = 0.2 + 0.7 * np.clip(fundamental_amps[t - 1], 0, 1)
            fundamental_freqs[t] = alpha * fundamental_freqs[t - 1] + (1 - alpha) * fundamental_t
    return fundamental_freqs, fundamental_amps


def fundamental_to_harmonics(fundamental_freqs, fundamental_amps, amplitude):
    harmonics = np.zeros(amplitude.shape)
    for t in range(len(fundamental_freqs)):
        # every python line drawing library I found
        # only works at integer coordinates
        # so here we are
        s = max(0, t - 1)
        if fundamental_amps[s] > 0 and fundamental_amps[t] > 0:
            for i in range(1, 40):
                draw_harmonic_slice(
                    harmonics,
                    t,
                    fundamental_freqs[s] * i,
                    fundamental_freqs[t] * i,
                    fundamental_amps[s],
                    fundamental_amps[t],
                )
                # draw_harmonic_slice(harmonics, t, fundamental_freqs[s]*i, fundamental_freqs[t]*i, 1,1)
    return harmonics


def normalize_pitch(amplitude, phase, fundamental_freqs, fundamental_amps, base_pitch=None):
    pitch_normalized_amp = np.zeros(amplitude.shape)
    pitch_normalized_phase = np.zeros(phase.shape)
    if base_pitch is None:
        base_pitch = np.mean(fundamental_freqs)
    for t in range(len(fundamental_freqs)):
        # TODO: factor this into a separate method
        amp_column = amplitude[:, t]
        phase_column = phase[:, t]
        if fundamental_freqs[t] != 0 and fundamental_amps[t] != 0:
            scaling_factor = base_pitch / fundamental_freqs[t]
        else:
            scaling_factor = 1
        orig_height = len(amp_column)
        scaled_height = int(orig_height * scaling_factor)
        slice_height = min(orig_height, scaled_height)
        for dst_f in range(slice_height):
            # n^2 nearest neighbor scaling since ndimage scaling was even slower
            src_f = int(dst_f / scaling_factor)
            pitch_normalized_amp[dst_f, t, :] = amplitude[src_f, t, :]
            pitch_normalized_phase[dst_f, t, :] = phase[src_f, t, :]

        amp_column_scaled = pitch_normalized_amp[:, t]
        weights = np.clip(
            spectral_envelope(amp_column) / (spectral_envelope(amp_column_scaled) + 0.01), 0, 10
        )
        pitch_normalized_amp[:, t] = (amp_column_scaled.T * weights).T
    return pitch_normalized_amp, pitch_normalized_phase


def extract_fundamental(amplitude):
    fundamental = np.zeros(amplitude.shape)
    # TODO: replace all of this with real code or at least clean it up
    # it should just be one big numpy thingy
    for t in range(amplitude.shape[1]):
        for f in range(4, 40):
            s = amplitude[f - 2 : f + 3, t, 0]
            if (
                np.dot(np.array([-0.5, -0.5, 2, -.5, -.5]), s) > 0
                and amplitude[f][t][0]
                + amplitude[f][max(t - 1, 0)][0]
                + fundamental[f][max(t - 1, 0)][0]
                > 0.5 * amplitude.max()
            ):
                for i in range(f - 4, f + 9):
                    fundamental[i][t] = amplitude[i][t] * (1 - abs(f - i) / 13)
                break
    # remove dots
    mask = (
        grey_dilation(
            grey_erosion(fundamental, structure=np.ones((5, 5, 1))), structure=np.ones((6, 12, 1))
        )
        > 0.1
    )
    conversion.image_to_file(mask, "mask_" + str(amplitude.shape) + ".png")
    fundamental *= mask
    return fundamental


def spectral_envelope(amplitude):
    axis = (1, 2)
    if amplitude.ndim == 2:  # single column
        axis = (1,)
    return scipy.ndimage.filters.gaussian_filter1d(np.mean(amplitude, axis=axis), 10)


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


def global_eq_match(content, style, harmonics):
    """
    :param content:
    :param style:
    """
    content = np.maximum(content, harmonics / harmonics.max() * content.max())  # lol
    gf = lambda x: scipy.ndimage.filters.gaussian_filter1d(x, 10)
    content_mean_freq = gf(np.mean(content, axis=(1, 2)))
    style_mean_freq = gf(np.mean(style, axis=(1, 2)))
    harmonics_mean_freq = gf(np.mean(harmonics, axis=(1, 2)))

    weights = style_mean_freq / content_mean_freq
    weights = np.clip(weights, 0, 2)
    stylized = (content.T * weights).T

    # TODO: figure out how to actually do this correctly :P
    # additive_weights = np.clip(style_mean_freq - content_mean_freq,0,10)
    # console.stats(stylized, "stylized before additive weights")
    # added_weights = (harmonics.T * additive_weights / harmonics.max()).T
    # stylized = np.maximum(stylized, 2 * added_weights * style.max())
    # console.stats(stylized, "stylized after additive weights")
    # console.stats(added_weights, "added_weights")
    assert stylized.shape == content.shape
    return stylized


def stylize(content, style):
    stylized = content
    # Pitch fundamental extraction (currently unused)
    fundamental_mask = extract_fundamental(content)
    fundamental_freqs, fundamental_amps = extract_fundamental_freqs_amps(fundamental_mask, content)
    harmonics = fundamental_to_harmonics(fundamental_freqs, fundamental_amps, content)

    stylized = global_eq_match(content, style, harmonics)
    # stylized = global_eq_match_2(content, style)
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
        content_img, content_phase = conversion.audio_to_spectrogram(
            content_audio, fft_window_size=1536
        )

        stylized_img = stylize(content_img, style_img)
        stylized_audio = conversion.amplitude_to_audio(
            stylized_img, fft_window_size=1536, phase_iterations=1, phase=content_phase
        )

        # Save stylized spectrogram and audio.
        conversion.image_to_file(stylized_img, stylized_img_path)
        conversion.audio_to_file(stylized_audio, stylized_audio_path)
        console.info("Finished processing %s; saved to %s" % (sample_name, stylized_audio_path))


if __name__ == "__main__":
    main()
