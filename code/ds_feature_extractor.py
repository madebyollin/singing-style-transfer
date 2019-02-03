import subprocess
import numpy as np

def get_feature_array(file_path):
    subprocess.run(["./DeepSpeechCommand.sh", file_path, "/tmp"])
    return np.load('/tmp/layer_5.npy').T
