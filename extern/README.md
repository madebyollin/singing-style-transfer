# extern

External repositories bootstrapped for our use

- DeepSpeech [[github link]](https://github.com/mozilla/DeepSpeech)

## DeepSpeech

### Workflow

0. Download models

  ```bash
  wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-models.tar.gz
  tar xvfz deepspeech-0.4.1-models.tar.gz
  ```
1. Install requirements (tested for Python3, Ubuntu 16.04)
  - This primarily uses `tensorflow-gpu=1.12` and an obscure `ds_ctcdecoder`
  ```bash
  pip install -r requirements
  ```
  - You can read more about
    [`ds_ctcdecoder`](https://tools.taskcluster.net/index/project.deepspeech.deepspeech.native_client.master/cpu-ctc)
    - Python 2.7 [whl](https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.master.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a1-cp27-cp27m-manylinux1_x86_64.whl)
    - Python 3.5 [whl](https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.master.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a1-cp35-cp35m-manylinux1_x86_64.whl)
2. Run `DeepSpeech/DeepSpeech.py` 

  ```bash
  python3 DeepSpeech/DeepSpeech.py --one_shot_infer=<wav file>
  ```

