import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as signal
import numpy as np

# Read the CSV file
df = pd.read_csv('fingerproper.csv', sep=' ', header=None, names=['R', 'G', 'B'])

red = df["R"].to_numpy()
green = df["G"].to_numpy()
blue = df["B"].to_numpy()



def band_pass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a band-pass filter to the input signal using a Butterworth filter.
    
    Args:
    signal (numpy array): Input signal to filter.
    lowcut (float): Low cutoff frequency in Hz.
    highcut (float): High cutoff frequency in Hz.
    fs (float): Sampling rate of the input signal.
    order (int): Order of the Butterworth filter (default is 5).
    
    Returns:
    numpy array: Filtered signal.
    """
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, data)
    return filtered_signal

red = band_pass_filter(red, 30, 100, 250)
green = band_pass_filter(green, 30, 100, 250)
blue = band_pass_filter(blue, 30, 100, 250)

fft_red = np.fft.fft(red) 
fft_red = np.fft.fft(green) 
fft_red = np.fft.fft(blue) 
t = range(2706)

freq_red = np.fft.fftfreq(len(red), t[1]-t[0])
freq_green = np.fft.fftfreq(len(green), t[1]-t[0])
freq_blue = np.fft.fftfreq(len(blue), t[1]-t[0])



# Create a figure and three subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
fig_fft = plt.figure()


# Plot data in the first column
axs[0].plot(range(2706), red, label='Red Wavelengths', color='red')
axs[0].set_xlabel('R')
axs[0].set_ylabel('Intensity')

# Plot data in the second column
axs[1].plot(range(2706), green, label='Green Wavelengths', color='green')
axs[1].set_xlabel('G')
axs[1].set_ylabel('Intensity')

# Plot data in the third column
axs[2].plot(range(2706), blue, label='Blue Wavelengths', color='blue')
axs[2].set_xlabel('B')
axs[2].set_ylabel('Intensity')

# Add a title to the figure
fig.suptitle('RGB Absorption')

plt.figure(fig_fft.number)
plt.plot(freq_red, np.abs(fft_red),color='red')
plt.title('FFT of Red Wavelengths')



# Show the plot
plt.show()