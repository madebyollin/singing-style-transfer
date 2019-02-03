#!/usr/bin/env python
import scipy
import numpy as np
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from skimage.feature import match_template
from scipy.signal import convolve2d
from skimage import measure
import conversion
import console
import cv2
import ipdb
from skimage.transform import resize
from ds_feature_extractor import get_feature_array


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


def scale_one_column(column, scaling_factor):
    output = np.zeros(column.shape)
    orig_height = len(column)
    scaled_height = int(orig_height * scaling_factor)
    slice_height = min(orig_height, scaled_height)
    for dst_f in range(slice_height):
        # n^2 nearest neighbor scaling since ndimage scaling was even slower
        src_f = int(dst_f / scaling_factor)
        output[dst_f] = column[src_f]
    return output


def formant_preserving_scale_one_column(amp_column, scaling_factor):
    amp_column_scaled = scale_one_column(amp_column, scaling_factor)
    weights = np.clip(
        spectral_envelope(amp_column) / (spectral_envelope(amp_column_scaled) + 0.01), 0, 10
    )
    return (amp_column_scaled.T * weights).T


def normalize_pitch(amplitude, phase, fundamental_freqs, fundamental_amps, base_pitch=None):
    pitch_normalized_amp = np.zeros(amplitude.shape)
    if phase is None:
        pitch_normalized_phase = None
    else:
        pitch_normalized_phase = np.zeros(phase.shape)
    if base_pitch is None:
        base_pitch = np.mean(fundamental_freqs)
    for t in range(len(fundamental_freqs)):
        # TODO: factor this into a separate method
        amp_column = amplitude[:, t]
        if phase is not None:
            phase_column = phase[:, t]
        if fundamental_freqs[t] > 5 and fundamental_amps[t] != 0:
            scaling_factor = base_pitch / fundamental_freqs[t]
        else:
            scaling_factor = 1
        if phase is not None:
            pitch_normalized_phase[:, t] = scale_one_column(phase_column, scaling_factor)
        pitch_normalized_amp[:, t] = formant_preserving_scale_one_column(amp_column, scaling_factor)
    return pitch_normalized_amp, pitch_normalized_phase


def extract_fundamental(amplitude):
    fundamental = np.zeros(amplitude.shape)
    # TODO: replace all of this with real code or at least clean it up
    # it should just be one big numpy thingy
    f_band_min = -4
    f_band_max = 8
    f_band_len = f_band_max - f_band_min
    f_band_coeffs = (
        1
        - np.concatenate(
            (np.array(range(f_band_min, 0)) / f_band_min, np.array(range(f_band_max)) / f_band_max)
        )
    )[:, np.newaxis]
    peak_finder = np.array([-0.5, -0.5, 2, -.5, -.5])[:, np.newaxis].T
    # convolved = convolve2d(np.mean(amplitude, axis=2), peak_finder, mode="same")
    console.time("big loop")
    freqs = np.argmax(np.mean(amplitude[:50], axis=2), axis=0)
    console.stats(freqs)
    for t in range(amplitude.shape[1]):
        f = freqs[t]
        # handle case where 2nd harmonic > first
        if np.mean(amplitude[f // 2, t]) > 0.4 * np.mean(amplitude[f, t]):
            f = f // 2
            freqs[t] = f
        if f > 5:
            f_min = f + f_band_min
            f_max = f + f_band_max
            fundamental[f_min:f_max, t] = f_band_coeffs * amplitude[f_min:f_max, t]

    console.timeEnd("big loop")
    console.time("remove dots")
    mask = (
        grey_dilation(
            grey_erosion(fundamental, structure=np.ones((3, 5, 1))), structure=np.ones((6, 12, 1))
        )
        > 0.1
    )
    console.timeEnd("remove dots")
    # conversion.image_to_file(mask, "mask_" + str(amplitude.shape) + ".png")
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


def compute_features(amplitude):
    # TODO: low-dimensional vector for each one
    features = measure.block_reduce(np.mean(amplitude, 2), (4, 1), np.max)
    # remove global frequency dist
    features /= np.clip(np.mean(features, axis=1)[:, np.newaxis], 0.25, 4)
    features = np.clip(features, 0, 10)
    # weight higher frequencies less
    weights = 1 - np.linspace(0, 1, num=features.shape[0])
    weights[:5] = 0  # lol
    return weights[:, np.newaxis] * features


def audio_patch_match(content, style, content_freqs, style_freqs, content_features, style_features):
    console.log("content.shape:", content.shape)
    console.log("content_features.shape:", content_features.shape)
    console.log("style.shape:", style.shape)
    console.log("style_features.shape:", style_features.shape)
    # setup
    output = np.zeros(content.shape)
    num_freqs, num_timesteps, num_channels = content.shape
    _, num_timesteps_style, _ = style.shape
    # initialization of nnf
    nnf = (np.random.uniform(low=-1, high=1, size=num_timesteps) * num_timesteps_style).astype(
        np.int32
    )

    # distance function
    def distance(content_t, style_t):
        style_t %= num_timesteps_style
        c = content_features[:, content_t]
        s = style_features[:, style_t]
        return np.sum(np.abs(c - s)) / len(c)

    w = 1024
    # TODO: iterative supersampling of priors so we don't have to do all this work each iteration
    iterations = 8
    amps = np.max(content, axis=(0, 2))
    for _ in range(iterations):
        # iteratively improve the correspondence_field
        for t in range(num_timesteps):
            amp = amps[t]
            # propagation
            # TODO: numpyify all this
            possible_offsets = nnf[t - 1 : t + 1].tolist()
            # random search around the current offset
            radius = w
            # TODO: especially this part dear lord
            while radius > 1:
                r_i = np.random.uniform(-1, 1)
                u_i = nnf[t] + int(r_i * radius)
                possible_offsets.append(u_i)
                radius = radius // 2  # alpha decreases
            # compute the best of all the options
            best_offset = None
            best_offset_dist = None
            for off in possible_offsets:
                s = t + off
                # consistency_penalty = amp * 200 * abs(nnf[t - 1] - off) / num_timesteps_style
                consistency_penalty = 0
                offset_dist = distance(t, s) + consistency_penalty
                if best_offset is None or offset_dist < best_offset_dist:
                    best_offset = off
                    best_offset_dist = offset_dist
            nnf[t] = best_offset % num_timesteps_style
    # now, we just assign the thingy wingies
    for t in range(num_timesteps):
        s = (t + nnf[t]) % num_timesteps_style
        freq_stretch = content_freqs[t] / style_freqs[s]
        if not (0.5 < freq_stretch < 5):
            freq_stretch = 1
        output[:, t] = formant_preserving_scale_one_column(style[:, s], freq_stretch)
        # TODO: why is this even necessary
        output_max = output[:, t].max()
        content_max = content[:, t].max()
        if output_max > content_max:
            scaling_factor = 0 if content_max < 0.1 else content_max / output_max
            output[:, t] *= scaling_factor
    return output


def audio_patch_match_bad(
    content, style, content_freqs, style_freqs, content_features, style_features
):
    patch_size = 8
    output = np.zeros(content.shape)
    num_identity_stretches = 0
    for t in range(content_features.shape[1] // patch_size):
        slice_start = t * patch_size
        slice_end = slice_start + patch_size
        content_patch = content_features[:, slice_start:slice_end]
        matching_scores = match_template(style_features, content_patch)
        best_patch_t = np.argmax(matching_scores)
        # ipdb.set_trace()
        best_patch = style[:, best_patch_t : best_patch_t + patch_size]
        # output[:, slice_start:slice_end] = best_patch
        for p in range(patch_size):
            freq_stretch = content_freqs[slice_start + p] / style_freqs[best_patch_t + p]
            # freq_stretch = 32 / style_freqs[best_patch_t + p]
            if not (0.5 < freq_stretch < 10):
                freq_stretch = 1
                num_identity_stretches += 1
            best_patch_scaled = formant_preserving_scale_one_column(best_patch[:, p], freq_stretch)
            output[:, slice_start + p] = best_patch_scaled
    return output


def stylize(content, style, content_path, style_path):
    stylized = content
    # Pitch fundamental extraction
    console.time("extracting fundamentals")
    content_fundamental_mask = extract_fundamental(content)
    style_fundamental_mask = extract_fundamental(style)
    console.timeEnd("extracting fundamentals")
    console.time("fundamental freqs and amps")
    content_fundamental_freqs, content_fundamental_amps = extract_fundamental_freqs_amps(
        content_fundamental_mask, content
    )
    style_fundamental_freqs, style_fundamental_amps = extract_fundamental_freqs_amps(
        style_fundamental_mask, style
    )
    console.timeEnd("fundamental freqs and amps")
    # Pitch normalization
    console.time("pitch normalization")
    content_normalized, _ = normalize_pitch(
        content, None, content_fundamental_freqs, content_fundamental_amps
    )
    style_normalized, _ = normalize_pitch(
        style, None, style_fundamental_freqs, style_fundamental_amps
    )
    console.timeEnd("pitch normalization")

    # Featurization
    '''content_features = compute_features(content_normalized)
    style_features = compute_features(style_normalized)''' 
    content_features = get_feature_array(content_path)
    content_features = resize(content_features, (2048, content.shape[1]))
    style_features = get_feature_array(style_path)
    style_features = resize(style_features, (2048, style.shape[1]))

    # Patchmatch
    console.time("patch match")
    stylized = audio_patch_match(
        content,
        style,
        content_fundamental_freqs,
        style_fundamental_freqs,
        content_features,
        style_features,
    )
    console.timeEnd("patch match")
    # stylized = global_eq_match(content, style, harmonics)
    # stylized = global_eq_match_2(content, style)
    return stylized


def main():
    sample_dir = "sample"
    sample_names = ["rolling_in_the_deep"]
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

        stylized_img = stylize(content_img, style_img, content_path, style_path)
        stylized_audio = conversion.amplitude_to_audio(
            stylized_img, fft_window_size=1536, phase_iterations=1, phase=content_phase
        )

        # Save stylized spectrogram and audio.
        conversion.image_to_file(stylized_img, stylized_img_path)
        conversion.audio_to_file(stylized_audio, stylized_audio_path)
        console.info("Finished processing %s; saved to %s" % (sample_name, stylized_audio_path))


if __name__ == "__main__":
    main()
