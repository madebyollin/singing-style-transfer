#!/usr/bin/env python
from __future__ import print_function

import os

import librosa
import scipy
from PIL import Image
import numpy as np
import tensorflow as tf

from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.tfutils.sessinit import SaverRestore

import data_load as data_load
import hparam as hparam
hp = hparam.hparam
from models import Net1

hp.set_hparam_yaml("convert")

CKPT_DIR = "extern/deep_voice_conversion/pretrained_model/train1"

def init_predictor(ckpt_dir):
    """ Initializes an OfflinePredictor for the 'Net1' Phoneme classifier, given a directory of tf-checkpoints.

    :param ckpt_dir: Checkpoint directory.
    :return: OfflinePredictor
    """
    ckpt1 = tf.train.latest_checkpoint(ckpt_dir)
    assert ckpt1 is not None, "Failed to load checkpoint in '{}'".format(ckpt_dir)

    net1 = Net1()
    pred_conf = PredictConfig(
        model=net1,
        input_names=['x_mfccs'],
        output_names=['net1/ppgs'],
        session_init=ChainInit([SaverRestore(ckpt1, ignore=['global_step'])])
    )
    predictor = OfflinePredictor(pred_conf)
    return predictor

def get_network_output(wav,
        ckpt_dir=CKPT_DIR,
        out_path_fmt="extern/deep_voice_conversion/outputs/test2_{:04d}_{:04d}.png"):
    """ Computes PPGs for a given loaded wav audio.
    This splits the input wav file into two-second batches and runs each through the Phoneme classifier.
    For each batch, this outputs each ppg to the given output_path.

    :param wav: Loaded Wav audio.
    :param ckpt_dir: Pretrained 'Net1' weights.
    :param out_path_fmt: Output path format template to write each of the network outputs.
    """
    assert os.path.isdir(ckpt_dir)

    # Make output directory.
    out_dir = os.path.dirname(out_path_fmt)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Initialize Offline Predictor.
    predictor = init_predictor(ckpt_dir)

    # Split wav into 2-second clips.
    length = hp.default.sr * hp.default.duration
    # num_splits = int(wav.shape[0] / length)
    splits = list(range(0, wav.shape[0], length))
    num_splits = len(splits)
    wavs = np.array_split(wav, splits, axis=0)
    print("Original wav length is", len(wav), "with sample rate", hp.default.sr)
    print("Length of wavs is ", [len(x) for x in wavs])

    # mfcc_batch: [b=num_splits, time/lenghth, feats]
    mfcc_batch = np.array([data_load.get_mfccs_and_spectrogram(wav=wav_)[0] for wav_ in wavs])
    print("Length of mfccs is ", [len(x) for x in mfcc_batch])

    # inp.shape: [b=num_splits, time/length, feats]
    # ppgs: (N, T, V); 
    # you would think this would work, but
    # preds = predictor(mfcc_batch)
    # lets do this instead
    ppgs = []
    for mfcc in mfcc_batch:
        mfcc_minibatch = np.array([mfcc])
        # print("Running on mfcc of shape", mfcc_minibatch.shape)
        ppg = predictor(mfcc_minibatch)[0][0]
        # print("Got ppg of shape", ppg.shape)
        ppgs.append(ppg)
    print("Length of ppgs is ", [len(x) for x in ppgs])

    # Output each ppg.
    heatmaps = []
    for i, heatmap in enumerate(ppgs):
        assert 0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0
        out_path = out_path_fmt.format(i, num_splits-1)
        print("Writing heatmap '{}' of shape '{}' to '{}'".format(i, heatmap.shape, out_path))

        # REVIEW josephz: This really should be changed to the PIL Image.fromarray(...) -> convert -> save.
        #   as scipy.misc.toimage is deprecated. But how the fuck are they doing the conversion?
        #   I guess asserting the range and then manually converting will just have to do.
        #   I think the scipy.misc.toimage method was deprecated as they realized supporting generalized
        #   image conversion is actually really hard and the user should know how to do it themselves.
        #   In fact we have the most information available to best approach convert it exactly as we need.
        # import scipy.misc
        # scipy.misc.toimage(heatmap).save(out_path)

        # Convert [0, 1] heatmap to [0, 255].
        heatmap = 255 * heatmap
        im = Image.fromarray(heatmap.astype(np.uint8))

        # Save the image to disk.
        im.save(out_path)

        # Accumulate heatmap.
        heatmaps.append(heatmap)
    return heatmaps

def get_heatmap(wav):
    # Get non-stitched and un-upsampled heatmaps.
    # [n, [t, v]]
    heatmaps = get_network_output(wav)

    # Stitch and upsample heatmaps into one final heatmap.
    stitched_heatmap = np.concatenate(heatmaps, axis=0)
    print("stitched heatmap shape", stitched_heatmap.shape)
    stitched_heatmap = np.transpose(stitched_heatmap)
    return stitched_heatmap


if __name__ == "__main__":
    # REVIEW josephz: This should really be a cmd_line parameter.
    wav_file = "extern/deep_voice_conversion/test_data/reference_stylized.wav"
    assert os.path.isfile(wav_file)
    wav, _ = librosa.load(wav_file, sr=hp.default.sr)

    stitched_heatmap = get_heatmap(wav)
    im = Image.fromarray(stitched_heatmap.astype(np.uint8))
    im.save("extern/deep_voice_conversion/outputs/heatmap.png")
