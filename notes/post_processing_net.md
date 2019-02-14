## Spectrogram Post-Processing

**Inputs:**

- Stylized amplitude spectrogram
- Content super-res'd vowel harmonics
- Content super-res'd sibilants

So the input shape will be `[frequency_bins, num_timesteps, 3]`.

**Note:** We could also add the style audio (tiled/cropped) as an additional channel, more on this later.

**Outputs:**

- Stylized amplitude spectrogram

**Note:** we could also try predicting the phase.

**Data Generation Approach 1:**

- Take an input acapella file, convert to spectrogram.
- Split it in half, use the first half as the content and the second as the style, and stylize it. The original first half is our target; the stylized first half is our source. The network should learn to smooth irregularities and remove noise in the input.

**Data Generation Approach 2:**

- Take an input acapella file, convert to spectrogram.
- Stylize it with a different acapella.
- Stylize that again with the original. Use the double-stylized as the source and the original as the target.

**Network Structure 1:**

- A fully convolutional image-to-image network (similar to AcapellaBot). This should be capable of basic denoising and smoothing.

**Network Structure 2:**

- Some sort of fancy Conv-LSTM that reads over both the style audio and the stylized audio channels. This might be able to do something  smarter (like resolve ambiguities in a way that is characteristic of similar regions in the style audio).

## Sample-level Post Processing

Seems difficult, do not want.

## Tasks

- [ ] Decide on framework (Keras? PyTorch?)
- [ ] Prepare a bunch of training data using at least one of the approaches.
- [ ] Write an MVP network architecture
- [ ] Figure out why the MVP is bad and fix it.