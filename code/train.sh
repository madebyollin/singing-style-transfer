#!/usr/bin/env bash
train="$HOME/Desktop/train"
valid="$HOME/Desktop/valid"

python post_processor.py --train "$train" --valid "$valid" --test "sample/post_process/x.npy"
