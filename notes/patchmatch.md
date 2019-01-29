There's a good thread about the MSR neural patchmatch paper on twitter here:

https://twitter.com/alexjc/status/861154823688916992

I also found http://vis.berkeley.edu/papers/audioanalogies/aa.pdf which is a million years old but potentially a useful reference

Anyway, the idea is to apply the same algorithm in 1d to the pitch-normalized spectrograms, then un-normalize them (this should really be done in one scaling pass rather than 2, but will switch to that if results are usable). the similarity vectors should be neural, but we can test it with just spectral envelopes for now.

---

So, after implementing it, some notes:

* Better features will definitely help, but
* Vanilla Patchmatch is not a 100% natural fit. In particular: we may need to generate sounds that are *not present* in the style audio, which requires stronger priors than patchmatch has.

