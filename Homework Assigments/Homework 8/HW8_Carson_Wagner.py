################################################################################
# Created on Wed Nov 9, 2025                                                   #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Homework 8: Filtering Real Noisy Audio Signals                               #
################################################################################

import numpy as np           # Used for Freq and Fourier calculations
import sounddevice as sd     # Used to play audio (Commented out for Submission)
from scipy.io import wavfile # Used to read the WAV file


# Load the audio file (Change format as needed)
# Name your audio file with your first and last name as Audio_FirstName_LastName.m4a (or any other extension):
#audio = AudioSegment.from_file("Audio_Carson_Wagner.wav", format="wav")

sample_rate, sig_clean = wavfile.read("Audio_Carson_Wagner.wav")

if len(sig_clean.shape) > 1:
    sig_clean = sig_clean[:,0]

### Fetch the sample rate and calculate the time step
#sample_rate = audio.frame_rate
time_step = 1/sample_rate


##############################################################
#######        Your Code goes below this line:          ######
##############################################################

### Play the audio signal through your computer's speaker (normalize then play)
### Normalize for optimal audio clarity (sd.play is expecting a signal with amplitude between -1 and 1):

sig_clean = sig_clean.astype(float)
sig_clean = sig_clean / np.max(np.abs(sig_clean))

# ### Uncomment to send signal to the speaker (comment before submitting your code):
#sd.play(sig_clean, sample_rate)
#sd.wait()  # Wait until playback finishes

### Generate random noise and add it to the clean signal:
noise_amplitude = 0.1          # Amplitude of the Noise

# Add the noise to the clean signal
noise = noise_amplitude * np.random.randn(len(sig_clean))
sig_noise = sig_clean + noise


# Play the Noisy Signal for Testing (Comment out before submitting)
#sd.play(sig_noise, sample_rate)
#sd.wait()

### Take the signal to the frequency domain (FFT Operation) and
### null out the signal with frequencies beyond the cut-off frequency F_CO:
fft_signal = np.fft.fft(sig_noise)

# Create Frequency Vector Corresponding to the FFT
freq = np.fft.fftfreq(len(sig_clean), d=time_step)

# Null out any frequency beyond the cut-off frequency
freq_cut = 3000    # Hz, adjust based on my voice
fft_signal[np.abs(freq) > freq_cut] = 0


### Take the signal back to the time domain:
sig_filtered = np.fft.ifft(fft_signal)
sig_filtered = np.real(sig_filtered)       # Keep only the real parts


# Play Filtered Audio for Test Purposing (Comment out before submitting)
#sd.play(sig_filtered, sample_rate)
#sd.wait()

### Calculate the MSE (mean square error) with respect to the clean signal and print it (only print a single number):
mean_square_error = np.mean((sig_clean - sig_filtered) ** 2)

# Print Result
print(mean_square_error)
    