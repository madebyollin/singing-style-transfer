#!/usr/bin/env python
import os
import numpy as np
import keras

import console

class DataGenerator(keras.utils.Sequence):
    def __init__(self, in_path="/Users/ollin/Desktop/train", batch_size=16, tile_size=64):
        self.in_path = in_path
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.pairs = []
        self.load()

    def load(self):
        tile_size = self.tile_size
        for file_name in os.listdir(self.in_path + "/x/"):
            if file_name.endswith("npy"):
                source = np.load(self.in_path + "/x/" + file_name)
                target = np.load(self.in_path + "/y/" + file_name)
                style_input = np.load(self.in_path + "/style/" + file_name)
                coord_conv_slice = np.linspace(0, 1, source.shape[0])[:,np.newaxis]
                coord_conv_channel = np.repeat(coord_conv_slice, source.shape[1], axis=1)
                source = np.dstack([source, coord_conv_channel])
                num_freqs, num_timesteps, num_channels = source.shape
                for f_tile in range(num_freqs // tile_size):
                    for t_tile in range(num_timesteps // tile_size):
                        s = np.s_[f_tile * tile_size:f_tile * tile_size + tile_size, 
                                  t_tile * tile_size:t_tile * tile_size + tile_size]
                        x = source[s]
                        y = target[s][:,:,np.newaxis]
                        self.pairs.append([x, y])
        np.random.shuffle(self.pairs)
        console.log("Loaded", len(self.pairs), "pairs")
        console.log("Shape of first pair is", self.pairs[0][0].shape, self.pairs[0][1].shape)
    
    def on_epoch_end(self):
        np.random.shuffle(self.pairs)

    def __len__(self):
        return 64

    def __getitem__(self, index):
        max_index = int(np.floor(len(self.pairs) / self.batch_size))
        index %= max_index
        x = []
        y = []
        for b in range(self.batch_size):
            x_i, y_i = self.pairs[index * self.batch_size + b]
            x.append(x_i)
            y.append(y_i)
        return np.array(x), np.array(y)

if __name__ == "__main__":
    console.time("loading all data")
    d = DataGenerator()
    console.timeEnd("loading all data")
