#!/usr/bin/env python
import numpy as np
import librosa
import scipy
import warnings
import skimage.io as io
import os.path
import argparse
import console

DEFAULT_SAMPLERATE = 22050


def file_to_audio(audio_file_path):
    """Return (audio, sample_rate). If audio is stereo, audio will be (left, right)"""
    return librosa.load(audio_file_path, mono=False)


def file_to_image(image_file_path):
    """Return image as float64 array in the range (0, 1)"""
    image = io.imread(image_file_path, as_grey=True)
    if image.max() > 1:
        image = image / image.max()
    image = np.expand_dims(image, -1)
    return image


def image_to_file(image, image_file_path):
    image = np.mean(image, -1)
    if image.max() > 1:
        image /= image.max()
    image = np.clip(image, 0, 1)
    # low-contrast image warnings are not helpful
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(image_file_path, image)


def audio_to_file(audio, audio_file_path):
    librosa.output.write_wav(audio_file_path, audio, DEFAULT_SAMPLERATE, norm=True)


def split_path(path):
    """For a path <dir>/<basename>.<extension>, return (dir, basename, extension)"""
    dir, basename_ext = os.path.split(path)
    basename, ext = os.path.splitext(basename_ext)
    return dir, basename, ext


def audio_to_spectrogram(audio, fft_window_size):
    """
    return (amplitude, phase)
    """
    amplitude = []
    phase = []
    # if audio is mono, add a channel dimension anyway for convenience
    if len(audio) != 2:
        audio = [audio]
    for channel in audio:
        spectrogram = librosa.stft(channel, fft_window_size)
        amplitude.append(np.log1p(np.abs(spectrogram)))
        phase.append(np.imag(spectrogram))
    return np.stack(amplitude, axis=-1), np.stack(phase, axis=-1)

# TODO: This needs to fix this.
def amplitude_to_audio(channel_amplitudes, fft_window_size, phase_iterations=10):
    # phase reconstruction with successive approximation
    # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
    audio = [None, None]
    # if audio is mono, add a channel dimension anyway for convenience
    if channel_amplitudes.shape[-1] != 2:
        channel_amplitudes = np.dstack([channel_amplitudes] * 2)

    reconstruction = np.random.random_sample(channel_amplitudes[:, :, 0].shape) + 1j * (
        2 * np.pi * np.random.random_sample(channel_amplitudes[:, :, 0].shape) - np.pi
    )
    for i in range(phase_iterations):
        for j in range(2):
            channel = channel_amplitudes[:, :, j]
            console.stats(channel, "channel")
            # undo log1p
            amplitude = np.exp(channel) - 1
            # combine target amplitude and current phase to get the next audio iteration
            complex_spectrogram = amplitude * np.exp(1j * np.angle(reconstruction))
            channel_audio = librosa.istft(complex_spectrogram)

            # at each step, create a new spectrogram using the current audio
            reconstruction = librosa.stft(channel_audio, fft_window_size)
            audio[j] = channel_audio
    if len(audio) != 2:
        audio *= 2  # double the list
    return np.clip(np.array(audio), -1, 1)


def audio_file_to_image_file(audio_file_path, args):
    # read, convert to spectrogram
    audio, sample_rate = file_to_audio(audio_file_path)
    amplitude, phase = audio_to_spectrogram(audio, fft_window_size=args.fft)
    # write output
    dir, basename, ext = split_path(audio_file_path)
    image_file_path = os.path.join(dir, "{}.png".format(basename))
    image_to_file(amplitude, image_file_path)


def image_file_to_audio_file(image_file_path, args):
    # read, convert to audio
    image = file_to_image(image_file_path)
    audio = amplitude_to_audio(image, fft_window_size=args.fft, phase_iterations=args.iter)
    # write output
    dir, basename, ext = split_path(image_file_path)
    audio_file_path = os.path.join(dir, "{}.wav".format(basename))
    audio_to_file(audio, audio_file_path)


def main():
    parser = argparse.ArgumentParser(description="Convert images to audio and back again")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--iter", default=10, type=int, help="Iterations for phase reconstruction")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    for f in args.files:
        if f.endswith(".mp3") or f.endswith(".wav"):
            console.log("Converting", f, "to image")
            audio_file_to_image_file(f, args)
        elif f.endswith(".png"):
            console.log("Converting", f, "to audio")
            image_file_to_audio_file(f, args)
        else:
            console.log("Ignoring", f)


if __name__ == "__main__":
    main()
