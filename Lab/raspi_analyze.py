import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
sliced = 500

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))
    return sample_period, data
     


# Import data from bin file
fname = "original_test_6.bin"
try:
    sample_period, data = raspi_import(fname)
except FileNotFoundError as err:
    print(f"File {fname} not found. Check the path and try again.")
    exit(1)

# Uncomment to remove linear in/decrease and DC component
# data = signal.detrend(data, axis=0)
# sample period is given in microseconds, so this changes units to seconds
sample_period *= 1e-6

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(0, num_of_samples*sample_period, num_of_samples,
        endpoint=False)
# t = np.delete(t,4,axis=1)
# t = t[sliced:]


# Generate frequency axis and take FFT
# Use FFT shift to get monotonically increasing frequency
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
freq = np.fft.fftshift(freq)
# takes FFT of all channels
spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

#data from RPi in Volts [V]
data2 = (data/4096)*3.3
# data2 = np.delete(data2,[3,4],axis=1)
# data2 = data[sliced:]




#array of logarithms
spectrum_log = 20*np.log10(np.abs(spectrum[len(freq)//2:])) 
#obtains max values
max_values = np.amax(spectrum_log, axis=0) 

spectrum_log_rel = spectrum_log/max_values

#frequency spectrum

spectrum_1 = spectrum_log_rel[:,0]
spectrum_2 = spectrum_log_rel[:,1]
spectrum_3 = spectrum_log_rel[:,2]
spectrum_4 = spectrum_log_rel[:,3]
spectrum_5 = spectrum_log_rel[:,4]

#sinusoids



# Plot the results in two subplots
# If you want a single channel, use data[:,n] to get channel n
# For the report, write labels in the same language as the report

plt.subplot(2, 1, 1)
plt.title("Tids-domene representasjon til signalet x(t)")
plt.xlabel("Tid [s]")
# plt.xlim(0.4, 0.404)
# plt.ylim(0,3.2)
#plt.xlim(0.4, 0.41)
plt.ylabel("Spenning [V]")
plt.plot(t, data2)

plt.subplot(2, 1, 2)
plt.title("Effektspekteret til signalet x(t)")
plt.xlim(600,2000)
plt.xlabel("Frekvens [Hz]")
plt.ylabel("Relativ Effekt [dB]")
# Plot positive half of the spectrum (why?)
plt.plot(freq[len(freq)//2:], spectrum_log_rel)

# Required if you have not called plt.ion() first
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=1.2,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
num_rows, num_cols = spectrum_log.shape
# print(spectrum_log)
# print(data2)

for i in range(5):
    signal_power = np.sum((spectrum_log_rel[:,i])[np.argmax(spectrum_log_rel[:,i])]) 
    noise_power = np.sum(spectrum_log_rel[:,i])- signal_power 
    snr = 20*np.log10(signal_power/noise_power) 
    snr1 = abs(snr)
    print(snr1)
    
print((76.86900818225337+75.9966179089168+75.5279070023249)/3)
print("1")
print((77.03806915741153+76.42171320755597+76.21340787214704)/3)
print("2")
print((77.2357087845343+76.50945441873088+76.57174476391882)/3)
print("3")
print((77.48575409748554+76.43086803995962+76.29903631205272)/3)
print("4")
print((77.90176205745448+76.65994698467023+75.775598544306)/3)
print("5")