import subprocess
import numpy as np

from extern.DeepSpeech.util.flags import FLAGS
from extern.DeepSpeech.DeepSpeech_inference_hack import do_single_file_inference

def sox_call(fname):
    subprocess.run(["sox", fname, "/tmp/input.wav", "remix", "1,2"])

def get_feature_array(file_path):
    """ 
    0. Do necessary input preprocessing
    1. Setup Flags and Configs for DeepSpeech
    2. Run DeepSpeech
    """
    sox_call(file_path)

    thingy = do_single_file_inference(FLAGS.one_shot_infer)
    return thingy["layer_5"].T
    # return np.load('/tmp/layer_5.npy').T
