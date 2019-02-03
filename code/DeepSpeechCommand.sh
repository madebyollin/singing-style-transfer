#!/usr/bin/env bash
sox ${1} /tmp/input.wav remix 1,2
python ../extern/DeepSpeech/DeepSpeech-inference-hack.py --one_shot_infer="/tmp/input.wav" --checkpoint_dir="../extern/DeepSpeech/deepspeech-0.4.1-checkpoint/" --np_output_path="${2}" --alphabet_config_path="../extern/DeepSpeech/data/alphabet.txt"
