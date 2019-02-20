#!/usr/bin/env bash
train="/cse/web/homes/ollin/post_processor/train"
valid="/cse/web/homes/ollin/post_processor/valid"

python post_processor.py --train "$train" --valid "$valid" --test "sample/post_process/x.npy" --batch_size 128 "$@"
