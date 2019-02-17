#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

# import src.extern.DeepSpeech.util.audio as _audio

from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS

GRAPH_PB_PATH = '/home/josephz/GoogleDrive/University/UW/2018-19/CSE481I/singing-style-transfer/src/extern/models/models/output_graph.pb'


def load_graph(frozen_graph_filename=GRAPH_PB_PATH):
  # We load the protobuf file from the disk and parse it to retrieve the
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  return graph_def

if __name__ == '__main__':
  # Initialize Globals.
  create_flags()
  initialize_globals()

  # Load pretrained model.
  graph_def = load_graph()
  with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="prefix")

    # Open tf.Session.
    with tf.Session(graph=graph) as sess:

      # Extract graph node names.
      tf.import_graph_def(graph_def, name='')
      graph_nodes = [n for n in graph_def.node]
      names = []
      for i, t in enumerate(graph_nodes):
        names.append(t.name)
        print("graph_node: '{:03d}' -- '{}'".format(i, t.name))

      # Prepare audio input data.
      input_file_path = '/home/josephz/GoogleDrive/University/UW/2018-19/CSE481I/singing-style-transfer' \
                        '/src/data/aligned/one_last_time/one_last_time_original_30s.wav'
      features = audiofile_to_input_vector(input_file_path, Config.n_input, Config.n_context)
      num_strides = len(features) - (Config.n_context * 2)
      # Create a view into the array with overlapping strides of size
      # numcontext (past) + 1 (present) + numcontext (future)
      window_size = 2 * Config.n_context + 1
      features = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, Config.n_input),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

      # Prepare graph nodes for inference.
      # Prepare input nodes.
      # initialize_state = graph.get_tensor_by_name('initialize_state:0')
      input_node = graph.get_tensor_by_name('input_node:0')
      input_lengths = graph.get_tensor_by_name('input_lengths:0')

      name_mapping = {
        'layer_1': 'Minimum:0',
        'layer_2': 'Minimum_1:0',
        'layer_3': 'Minimum_2:0',
        'layer_5': 'Minimum_3:0',
        'layer_6': 'Add_4:0'
      }

      # Run graph.
      # sess.run(initialize_state)
      for l, node_name in name_mapping.items():
        print("Layer '{}' -- '{}'".format(l, node_name))
        y = graph.get_tensor_by_name(node_name)
        import pdb; pdb.set_trace()
        output_shit = sess.run(y, feed_dict={
          input_node: [features[:15]],
          input_lengths: [15],
        })
        output_path = '/home/josephz/GoogleDrive/University/UW/2018-19/CSE481I/singing-style-transfer/src/extern/DeepSpeech/scripts/tmp'
        print("\toutputting thing of shape '{}' to '{}'".format(y.shape, os.path.basename(output_path)))
        np.save(os.path.join(output_path, l), y)
