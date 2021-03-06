#!/usr/bin/env python

import os
import argparse
import numpy as np
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    UpSampling2D,
    Concatenate,
    Dropout,
    Lambda,
)
from keras.models import Model
from keras.optimizers import Adam

import console

import conversion
from post_tiled_data_gen import DataGenerator

class PostProcessor:
    def __init__(self, name=None):
        self.build_network()
        console.log("PostProcessor has", self.model.count_params(), "params")
        self.compile_network()
        self.name = name or "PostProcessor"

    def compile_network(self):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
        self.model.compile(loss="mean_absolute_error", optimizer=optimizer)

    def train(self, train, epochs, validation_data=None, test_file=None):
        losses = []
        cumulative_epochs = 0
        while epochs > 0:
            for epoch in range(epochs):
                console.info("epoch", cumulative_epochs)
                loss = self.train_epoch(train, validation_data, test_file)
                losses.append(loss)
                cumulative_epochs += 1
            while True:
                try:
                    epochs = int(console.prompt("How many more epochs should we train for? "))
                    break
                except ValueError:
                    console.warn("Oops, number parse failed. Try again, I guess?")

    def train_epoch(self, train, validation_data=None, test_file=None):
        history = self.model.fit_generator(
            generator=train, validation_data=validation_data, use_multiprocessing=True, workers=6
        )
        if test_file:
            self.denoise_from_file(test_file)
        self.save_weights("checkpoint_" + self.name + ".h5")
        return history.history["loss"]

    def build_network(self):
        num_filters_dict = {
            "d0": 256,
            "d1": 256,
            "pool_freq": 16,
            "pool_time": 16,
            "u0": 256,
            "u1": 256,
            "u2": 16,
            "output": 1, # Don't touch this, this is the number of output channels.
        }

        def base_conv(num_filters, kernel_shape, strides=1):
            return Conv2D(
                num_filters, kernel_shape, strides=strides, activation="relu", padding="same"
            )

        def static_conv(num_filters=32):
            return base_conv(num_filters, 3)

        def downsample_conv(num_filters=32):
            return base_conv(num_filters, 4, strides=2)

        def frequency_conv(num_filters=8):
            """Aggregates over the frequency axis (0) by downsampling the time axis (1)"""
            return base_conv(num_filters, (1, 4), strides=(1, 2))

        def time_conv(num_filters=8):
            """Aggregates over the time axis (1) by downsampling the frequency axis (0)"""
            return base_conv(num_filters, (4, 1), strides=(2, 1))

        # input
        noisy = Input(shape=(None, None, 3), name="noisy")
        conv = BatchNormalization()(noisy) # lazy

        # downsampling
        skip_a = conv
        conv = static_conv(num_filters_dict["d0"])(conv)
        # conv = Dropout(0.75)(conv)
        conv = static_conv(num_filters_dict["d0"])(conv)
        # conv = Dropout(0.75)(conv)
        conv = BatchNormalization()(downsample_conv(num_filters_dict["d0"])(conv))

        skip_b = conv
        conv = static_conv(num_filters_dict["d1"])(conv)
        # conv = Dropout(0.5)(conv)
        conv = static_conv(num_filters_dict["d1"])(conv)
        # conv = Dropout(0.5)(conv)
        conv = BatchNormalization()(downsample_conv(num_filters_dict["d1"])(conv))

        # processing at 1/4x resolution
        conv = static_conv(num_filters_dict["u0"])(conv)
        conv = static_conv(num_filters_dict["u0"])(conv)
        conv = BatchNormalization()(static_conv(num_filters_dict["u0"])(conv))

        # upsampling
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, skip_b])
        conv = static_conv(num_filters_dict["u1"])(conv)
        conv = static_conv(num_filters_dict["u1"])(conv)
        conv = BatchNormalization()(static_conv(num_filters_dict["u1"])(conv))
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, skip_a])
        conv = static_conv(num_filters_dict["u2"])(conv)
        conv = static_conv(num_filters_dict["u2"])(conv)
        conv = static_conv(num_filters_dict["output"])(conv)

        output = conv

        self.model = Model(inputs=noisy, outputs=output)

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def save_weights(self, weight_file_path):
        self.model.save_weights(weight_file_path, overwrite=True)

    def denoise_from_file(self, file_path):
        noisy = np.load(file_path)
        denoised = self.predict(noisy)
        conversion.image_to_file(denoised, file_path + ".denoised.png")

    def predict_unstacked(self, amplitude, harmonics, sibilants):
        return self.predict(np.dstack([amplitude, harmonics, sibilants]))

    def predict(self, x):
        padded_x = preprocess_x(x)
        x_with_batch = np.expand_dims(padded_x, 0)
        predicted_y_with_batch = self.model.predict(x_with_batch)
        predicted_y = predicted_y_with_batch[0]
        console.log("predicted_y produced output of shape", predicted_y.shape)
        y = postprocess_y(predicted_y, x.shape)
        return y

def preprocess_x(x):
    freq_size = 768
    num_x_slices = int(np.ceil(x.shape[1] / 64))
    time_size = num_x_slices * 64
    padded = np.zeros((freq_size, time_size, x.shape[2]))
    padded[0:, : x.shape[1], :] = x[:freq_size, :, :]
    return padded


def postprocess_y(y, target_shape):
    assert len(y.shape) == 3, "y shape of {} is wrong".format(y.shape)
    assert y.shape[-1] == 1, "y should have two channels"
    # clip negatives
    y = np.clip(y, 0, 10)
    # fix shape
    output = np.zeros(target_shape)
    output[: y.shape[0], :, :] = y[:, : target_shape[1], :]
    return output[:,:,:1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None, type=str, help="path containing training data")
    parser.add_argument("--valid", default=None, type=str, help="path containing validation data")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train")
    parser.add_argument("--name", default=None, type=str, help="name of experiment")
    parser.add_argument(
        "--weights", default="weights.h5", type=str, help="h5 file to read/write weights to"
    )
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument(
        "--load", action="store_true", help="Load previous weights file before starting"
    )
    parser.add_argument("--test", default=None, type=str, help="Test file to infer on every epoch")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()
    post_processor = PostProcessor(args.name)

    if len(args.files) == 0 and args.train:
        console.h1("preparing to train on {}".format(args.train))
        if args.load:
            console.log("loading weights from {}".format(args.weights))
            post_processor.load_weights(args.weights)
        console.log("loading data")
        train = DataGenerator(args.train, args.batch_size)
        valid = DataGenerator(args.valid, args.batch_size) if args.valid else None
        console.h1("training")
        post_processor.train(train, args.epochs, validation_data=valid, test_file=args.test)
        post_processor.save_weights(args.weights)
    elif len(args.files) > 0:
        console.h1("preparing to process", args.files, "...")
        post_processor.load_weights(args.weights)
        for f in args.files:
            post_processor.denoise_from_file(f)
    else:
        console.error("please provide data to train on (--train) or files to process")


if __name__ == "__main__":
    main()
