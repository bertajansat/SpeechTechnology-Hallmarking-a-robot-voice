import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Gain

def distortion(audio,sr):  
    board = Pedalboard([
    Distortion(),
    Gain(1.0)
    ])
    effected = board(audio, sr)
    return effected.T


# Load audio
audio_name="gen_2.wav"
y, sr = librosa.load("Generated/"+audio_name, sr=None)

# Shift formants
y_dist = distortion(y,sr)
# Save generated audios
sf.write("Generated/Post-processing/distorted_"+audio_name, y_dist, sr)

