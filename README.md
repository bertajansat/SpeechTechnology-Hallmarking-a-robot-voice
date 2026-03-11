# Hallmarking a robot voice

DT2112 Speech Technology Project: Hallmarking a robot voice

**Authors**: BERTA JANSAT BALLARÍN, FREDRIKA LUNDQVIST ÅBRINK, DAVID MARZBAN, FELIX ÖLANDER

## Setup

1. Create conda environment:
```
conda create -n xcodec2 python=3.10.19
conda activate xcodec2

```
2. Install requirements on "requirements.txt"

## Folders and files

* `Llasa.py` : File for TTS generation using llasa-3b-tts (Ref: https://huggingface.co/blog/srinivasbilla/llasa-tts) 
* `post_processing.py` : File for applying post-processing on the generated audios, using librosa (https://librosa.org/doc/latest/index.html) and pedalboard (https://github.com/spotify/pedalboard). Some of the post-processing functions include:
   * `Flanger`: Audio effect produced by mixing two identical signals together, one signal delayed by a small and gradually changing period.
   * `Clipping` : Abruptly flattening a signal’s amplitude when it exceeds a given threshold.
   * `Bitcrush`: Produce distortion by reducing the resolution or bandwidth of the audio data.
   * `Distortion` : Apply a non-linear (hyperbolic tangent tanh) waveshaping function to apply harmonically pleasing distortion to a signal.
   * `Pitch shift`: Changing the pitch of an audio signal
* `save_dataset.py` : File for downloading selected files from AbstractTTS/PODCAST Dataset (https://huggingface.co/datasets/AbstractTTS/PODCAST) 
* `Generated_audios/` : Folder containing the audio files generated using Llasa TTS with a given speech prompt from the PODCAST Dataset and a given text prompt. The final reference audio used is `reference_audio.wav`. In the folder `Flanger + Pitch`, the three different modified versions of the reference audio can be found.


