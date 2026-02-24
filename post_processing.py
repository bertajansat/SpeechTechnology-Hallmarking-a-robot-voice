import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Gain, Bitcrush, Clipping

def distortion(audio,sr, drive_db=20):  # drive_db= Amplification of input signal before being processed by the distortion (intensity of the effect)
    # Distortion: Applies a non-linear (hyperbolic tangent tanh) waveshaping function to apply harmonically pleasing distortion to a signal
    board = Pedalboard([
    Distortion(drive_db=drive_db),
    Gain(1.0)
    ])
    effected = board(audio, sr)
    return effected.T

def bitcrush(audio,sr, bit_depth=8):  # Bit_depth: 0-32 (32 Maximum resolution, 0 min)
    
    # Bitcrush: Produces distortion by reducing the resolution or bandwidth of the audio data
    
    board = Pedalboard([
    Bitcrush(bit_depth=bit_depth),
    Gain(1.0)
    ])
    effected = board(audio, sr)
    return effected.T

def clipping(audio,sr, offset_db=7):  # Offset_db: 0 (no clipping), clipping increases as Offset_db decreases
    
    # Clipping: Produces distortion by limiting a signal once it exceeds a dB threshold. 

    # Measure average amplitude in dB, so clipping threshold depends on the amplitude of the signal:
    rms_db = librosa.amplitude_to_db(np.sqrt(np.mean(audio**2)),ref=1.0)
    print(rms_db)
    th = rms_db - offset_db

    board = Pedalboard([
    Clipping(threshold_db=th),
    Gain(1.0)
    ])
    effected = board(audio, sr)
    return effected.T 

# Load audio
audio_name="gen_2.wav"
y, sr = librosa.load("Generated/"+audio_name, sr=None)

# Shift formants
y_dist = distortion(y,sr)
y_bitcrush = bitcrush(y,sr)
y_clip = clipping(y,sr)
# Save generated audios
sf.write("Generated/Post-processing/distorted_"+audio_name, y_dist, sr)
sf.write("Generated/Post-processing/bitcrush"+audio_name, y_bitcrush, sr)
sf.write("Generated/Post-processing/clipping"+audio_name, y_clip, sr)
