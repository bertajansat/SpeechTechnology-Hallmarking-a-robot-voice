# Speech Technology

DT2112 Speech Technology Project: Hallmarking a robot voice

1. Create conda environment:
```
conda create -n xcodec2 python=3.10.19
conda activate xcodec2

```
2. Install requirements on "requirements.txt"

## Structure of the repository

<Llasa.py> : File for TTS generation using llasa-3b-tts (Ref: https://huggingface.co/blog/srinivasbilla/llasa-tts) 
<post_processing.py> : File for applying post-processing on the generated audios, using librosa (https://librosa.org/doc/latest/index.html) and pedalboard (https://github.com/spotify/pedalboard)
