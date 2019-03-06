# Deep Voice Conversion

Voice Conversion with Non-Parallel Data

## Overview

- Code: see Github [repo link](https://github.com/joseph-zhong/deep-voice-conversion) for complete code base
- Model: You must download one of the pre-trained models into `./pretrained_model/train1`
  - 70% Accuracy:
    - [gdrive link](https://drive.google.com/open?id=1ExlBIpZO0mxBhK4WEoW2WhahG1dZwdKW)
  - 93% Accuracy:
    - [gdrive link](https://drive.google.com/file/d/1yC3G3V03X3s8mKJ1J6bMkOqDT8r-TBb8/view?usp=sharing)

## Workflow

- Simply run `./get_network_output.py` to produce the stitched heatmap at `outputs/heatmap.png`
- `get_network_output.get_heatmap(wav)` returns the stitched heatmap from 2-second clips

## TODOs

- [x] Variable Input
- [ ] Variable Output
- [x] Automatic Output stitching
- [ ] and upsampling (This should be handled by the client?)

### Dependencies

Package Requirements:

**Important**: Tested only in Python2.7

(It is to-be-determined whether this is actually a hard-requirement)

- `tensorflow-gpu >= 1.8`
- `numpy >= 1.11.1`
- `librosa == 0.6.3`
- `joblib >= 0.12.0`
- `tensorpack >= 0.8.6`
- `pyyaml`
- `soundfile`
- `pydub`
- `tqdm`
- `git+https://github.com/wookayin/tensorflow-plot.git@master`


