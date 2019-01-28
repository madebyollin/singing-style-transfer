#!/usr/bin/env python3
import os
import tensorflow as tf

GRAPH_PB_PATH = '/home/josephz/GoogleDrive/University/UW/2018-19/CSE481I/singing-style-transfer/src/extern/models/models/output_graph.pb'

with tf.Session() as sess:
  with tf.gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  sess.graph.as_default()
  tf.import_graph_def(graph_def, name='')
  graph_nodes = [n for n in graph_def.node]
  names = []
  for i, t in enumerate(graph_nodes):
    names.append(t.name)
    print("graph_node: '{:03d}' -- '{}'".format(i, t.name))

  # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
  # y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
  #
  # sess.run()

