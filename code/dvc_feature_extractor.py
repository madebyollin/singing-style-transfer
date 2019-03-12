import subprocess
import numpy as np
import console

def dvc_call(fname, output_fname):
    subprocess.run(["python2", "extern/deep_voice_conversion/get_network_output.py", fname, output_fname])

def get_feature_array(file_path):
    npy_file_path = "/tmp/features.npy"
    dvc_call(file_path, npy_file_path)

    network_output = np.load(npy_file_path)
    console.stats(network_output, "network output")
    return network_output
