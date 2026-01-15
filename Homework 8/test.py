################################################################################
# Created on Wed Nov 9, 2025                                                   #
# @author: Carson Wagner                                                        #
# Homework 8: Filtering Real Noisy Audio Signals                               #
################################################################################

# --- REPLACED IMPORTS ---
# Removed: import soundfile as sf
# Removed: from pydub import AudioSegment

import numpy as np
import sounddevice as sd
# --- NEW STANDARD IMPORT ---
from scipy.io.wavfile import read 


# Load the audio file using scipy.io.wavfile.read
# Ensure 'Audio_Carson_Wagner.wav' is in the same directory as your script.
try:
    sample_rate, sig_raw = read("Audio_Carson_Wagner.wav")
except FileNotFoundError:
    print("Error: Audio file not found. Ensure 'Audio_Carson_Wagner.wav' is in the correct path.")
    exit()

# Convert the signal data type to float and handle potential multi-channel audio
# If it's multi-channel, grab only the first channel (column 0)
if sig_raw.ndim > 1:
    sig_raw = sig_raw[:, 0]

sig_clean = sig_raw.astype(np.float64)


##############################################################
#######           Your Code goes below this line:          ######
##############################################################

### Normalize the clean signal (REQUIRED STEP)
sig_clean = sig_clean / np.max(np.abs(sig_clean))


### Generate random noise and add it to the clean signal:
noise_amplitude = 0.1             # Amplitude of the Noise (REQUIRED VALUE)

# Add the noise to the clean signal
noise = noise_amplitude * np.random.randn(len(sig_clean))
sig_noise = sig_clean + noise

# Uncomment to play the noisy signal for testing (COMMENT BEFORE SUBMISSION)
# sd.play(sig_noise, sample_rate)
# sd.wait()


### Take the signal to the frequency domain (FFT Operation) and
### null out the signal with frequencies beyond the cut-off frequency F_CO:
fft_signal = np.fft.fft(sig_noise)

N = len(sig_clean)
F_s = sample_rate
F_c = 3000     # Hz (Adjust this cut-off frequency F_c based on your voice)

# 1. Calculate the index K_cut corresponding to the cut-off frequency F_c
# This index marks the beginning of the frequencies we want to zero out.
K_cut = int(np.floor((F_c / F_s) * N))

# 2. Filter the signal by setting the high-frequency components to zero.
# We must maintain conjugate symmetry for a real-valued IFFT result.
# The slice [K_cut : N - K_cut] zeroes the high positive frequencies AND 
# their corresponding high negative frequency mirrors.
fft_signal[K_cut : N - K_cut] = 0 


### Take the signal back to the time domain:
sig_filtered = np.fft.ifft(fft_signal)
# Check for correctness: The imaginary part should be close to zero.
sig_filtered = np.real(sig_filtered)     # Keep only the real parts

# Uncomment to play the filtered signal for testing (COMMENT BEFORE SUBMISSION)
# sd.play(sig_filtered, sample_rate)
# sd.wait()

### Calculate the MSE (mean square error) and print it:
mean_square_error = np.mean((sig_clean - sig_filtered) ** 2)

# Print Result (ONLY print this number for submission)
print(mean_square_error)