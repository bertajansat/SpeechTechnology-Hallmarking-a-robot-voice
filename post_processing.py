import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Gain, Bitcrush, Clipping, MP3Compressor
import math
import pyrubberband


def pitch_time_shift(audio,sr, delay=0):
   
    depth = 0
    mix = 0
    n=len(y)
    delay_samples = math.trunc(delay*sr)
    t = np.arange(n) / sr
    
    y_high = pitch_switch(audio,sr,n_steps=2,bins_octave=12)
    y_low = pitch_switch(audio,sr,n_steps=-2,bins_octave=12)

    y_pitch = y_high + y_low
    
    return y_pitch

def compression_artifacts(audio,sr): # --> Not very notable changes applied
    compressor = MP3Compressor(vbr_quality=9.99)  # 0.0 = better quality, 9.99 = worse
    
    processed = compressor.process(
        audio,
        sample_rate=sr,
        buffer_size=64
    )
    return processed.T

def speaking_vel(audio, sr, v_rate, quality="low"): 
    effected = np.copy(audio)
    print(quality)
    if quality=="high":
        print("hola")
        return pyrubberband.pyrb.time_stretch(effected, sr, rate=v_rate)
    elif quality=="low":
        return librosa.effects.time_stretch(effected,rate=v_rate)
    else:
        return None
def random_n_steps(min_steps,max_steps,zero_prob):
    if np.random.rand() < zero_prob:
        return 0
    else:
        while True:
            n = np.random.uniform(min_steps, max_steps)
            if n != 0.0:
                return n

def pitch_switch(audio,sr,n_steps=4,bins_octave=12,type='soxr_lq'):
    effected = np.copy(audio)  # ratio = 2 ** (n_steps / bins_per_octave)
    return librosa.effects.pitch_shift(effected,sr=int(sr),n_steps=n_steps,bins_per_octave=bins_octave,res_type=type)

def pitch_contour(audio,sr,num_pitches=50,zero_prob=0.2, bins_octave=24):
    f0, voicing, voicing_probability = librosa.pyin(y=audio, sr=sr, fmin=50, fmax=300)
    effected = np.copy(audio)
    n=math.trunc(len(audio)/num_pitches)
    for i in range(num_pitches):
        start = n * i
        end = n * (i + 1)
        # Pitch shift del segment
        n_steps = random_n_steps(0, 1, zero_prob=zero_prob)
        effected[start:end] = librosa.effects.pitch_shift(
            effected[start:end],
            sr=sr,
            n_steps=n_steps,
            bins_per_octave=bins_octave,
            res_type='soxr_vhq' # Can be changed: 'soxr_vhq' (very high-quality FFT bandlimited), 'soxr_hq' (high-quality FFT bandlimited, default), 'soxr_mq' (medium-quality FFT bandlimited), 
                                #'soxr_lq' (low-quality FFT bandlimited), 'soxr_qq' (quick cubic, very fast, not bandlimited), 'kaiser_best' (resampy high-quality), 'kaiser_fast' (resampy faster), 
                                #'fft' / 'scipy' (Fourier resampling via scipy), 'polyphase' (fast polyphase filtering), 'linear' (very fast linear interpolation, not bandlimited), 'zero_order_hold' 
                                #(fast sample hold, not bandlimited), 'sinc_best' / 'sinc_medium' / 'sinc_fastest' (high/medium/low-quality bandlimited sinc interpolation).
        )
    return effected

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

def flanger(y, sr, freq=3.5, m_0_ms=5, depth_ms=5, g=1, mix = 1): # Depth <= m_0!
    # Flanger: y(n)=x(n)+g*x(n-M(n))

    m_0=m_0_ms*sr/1000
    depth = depth_ms*sr/1000
    n=len(y)
    t = np.arange(n) / sr
    # M(n) = M0 + D * LFO(n) = M0 + D * sin(2*pi*f_LFO*n/fs)
    lfo = np.sin(2 * np.pi * freq * t)
    m = m_0 + depth*lfo

    y_flanger = np.zeros(n)
    for i in range(n):
            delay = m[i]
            idx = i - delay # Compute position where we must add the delay: x(n-M(n)) = x(idx)

            if idx < 0:
                delayed_sample = 0 # Set negative indices to 0
            else:
                # Lineal interpolation
                i0 = int(np.floor(idx)) # Avoid decimal indeces
                i1 = min(i0 + 1, n - 1) # Avoid surpassing vector size 
                
                # For non-whole numbers, compute average of nearby samples:
                frac = idx - i0
                delayed_sample = (1 - frac) * y[i0] + frac * y[i1]

            y_flanger[i] = (1 - mix) * y[i] + mix * (y[i] + g * delayed_sample) # Mix to control how much the effect is applied

    return y_flanger




# Load audio
audio_name="stress_tired_woman.wav"
y, sr = librosa.load("Generated_audios/"+audio_name, sr=None)

# Shift formants
y_pitch_c = pitch_contour(y,sr)
y_pitch_s = pitch_switch(y,sr)
y_dist = distortion(y,sr)
y_bitcrush = bitcrush(y,sr)
y_clip = clipping(y,sr)
y_vel = speaking_vel(y, sr, v_rate=0.75, quality="low")

y_shift_1 = flanger(y,sr)
y_shift= pitch_switch(y_shift_1,sr,n_steps=1,bins_octave=12, type='soxr_hq')
# Save generated audios
#sf.write("Generated/Post-processing/pitch_c_"+audio_name, y_pitch_c, sr)
#sf.write("Generated/Post-processing/pitch_s_"+audio_name, y_pitch_s, sr)
#sf.write("Generated/Post-processing/distorted_"+audio_name, y_dist, sr)
#sf.write("Generated/Post-processing/bitcrush_"+audio_name, y_bitcrush, sr)
#sf.write("Generated/Post-processing/clipping_"+audio_name, y_clip, sr)

sf.write("Generated_audios/Flanger + Pitch/pitch_1_soxr_hq_NO_"+audio_name, y_shift, sr)
