import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys
import csv
 

# =============================================================================
# ##### ACOUSTICS LAB PARAMETERS #####
# =============================================================================
fs = 31250 # sample frequency from command line
channels = 3 # number of channels (3 for acoustic lab)
mik1 = 0 # index of microphone 1
mik2 = 1 # index of microphone 2
mik3 = 2 # index of microphone 3
INTERPOLATION_FACTOR = 10 # interpolation factor for cross-correlation


##### FUNCTIONS #####
def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """
    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64') # the "dangling" `.astype('float64')` casts data to double precision
        data = data.reshape((-1, channels)) # stops noisy autocorrelation due to overflow
        data = signal.detrend(data, axis=0) # remove DC offset
    return sample_period, data
    #return data

def get_data():
    # Import data from bin file
    fname =  sys.argv[1] # take in file name from command line 'Data/'
    try:
        sample_period, data = raspi_import(fname) 
        #data = raspi_import(fname) 
        print(f"Fil: {fname}, vellykket import.")
    except FileNotFoundError as err:
        print(f"Fil: {fname}, ikke funnet.")
        exit(1)
    sample_period *= 1e-6 # sample period is given in microseconds, so this changes units to seconds
    return sample_period, data


def xcorr(data1, data2):
    """
    Calculates the cross-correlation of two signals.
    :param data1: first signal
    :param data2: second signal
    :return: xcorrelation
    """
    # Not interested in negative values of the cross-correlation
    xcorr_12 = np.abs(np.correlate(data1[1:], data2[1:], mode='full'))  # cross-correlation of data1 and data2
    return xcorr_12



def all_xcorrs(data, INTERPOLATION_FACTOR, fs):
    """
    Calculates the cross-correlations of all channels.
    :param data: data
    :param INTERPOLATION_FACTOR: interpolation factor
    :param fs: sampling frequency
    :return: list of cross-correlations
    """
    xcorrs_data = [] # list of cross-correlations
    
    mik_21 = xcorr(data[:,mik1], data[:,mik2])
    mik_31 = xcorr(data[:,mik1], data[:,mik3])
    mik_32 = xcorr(data[:,mik2], data[:,mik3])
    
    range = np.linspace(-(len(mik_21) - 1) / 2, (len(mik_21) - 1) / 2, len(mik_21)) # original range
    range_interp = np.linspace(-(len(mik_21) - 1) / 2, (len(mik_21) - 1) / 2, INTERPOLATION_FACTOR*len(mik_21)) # interpolated range
    
    mik_21_interp = xcorrs_data.append(np.interp(range_interp, range, mik_21)) # interpolated
    mik_31_interp = xcorrs_data.append(np.interp(range_interp, range, mik_31)) # interpolated
    mik_32_interp = xcorrs_data.append(np.interp(range_interp, range, mik_32)) # interpolated

    fs_interp = xcorrs_data.append(INTERPOLATION_FACTOR*fs) # interpolated sampling frequency

    return xcorrs_data, range # return list of interpolated cross-correlations (order: 21, 31, 32) and interpolated sampling frequency

def theta(xcorrs_data):
    """
    Calculates the angle theta and prints it.
    :param xcorrs_data: list of all cross-correlations
    """
    n_21 = int(xcorrs_data[0].argmax()) # max lag for cross-correlation mik_21    
    n_31 = int(xcorrs_data[1].argmax()) # max lag for cross-correlation mik_31
    n_32 = int(xcorrs_data[2].argmax()) # max lag for cross-correlation mik_32

    n_21 = int(n_21 - (len(xcorrs_data[0]) - 1) / 2) # max lag for cross-correlation mik_21 centered around 0
    n_31 = int(n_31 - (len(xcorrs_data[1]) - 1) / 2) # max lag for cross-correlation mik_31 centered around 0
    n_32 = int(n_32 - (len(xcorrs_data[2]) - 1) / 2) # max lag for cross-correlation mik_32 centered around 0
    
    print('Krysskorrelasjonens maks lag for mik_21: {}'.format(n_21))
    print('Krysskorrelasjonens maks lag for mik_31: {}'.format(n_31))
    print('Krysskorrelasjonens maks lag for mik_32: {}'.format(n_32))
    
    denominator_minus = -n_21 + n_31 + 2*n_32 # minus denominator of atan
    denominator = n_21 - n_31 - 2*n_32 # denominator of atan
    if denominator_minus < 0:
        theta = np.arctan(np.sqrt(3)*((n_21 + n_31) / denominator)) + np.pi # (n21 + n31) / (n21 - n31 - 2*n32)
    elif denominator_minus == 0:
        print('Deling på 0. Theta er udefinert.')
    else:
        theta = np.arctan(np.sqrt(3)*((n_21 + n_31) / denominator)) # theta
    #print('Theta: {:.3f} rad'.format(theta))
    
    theta_deg = theta * 180 / np.pi
    if theta_deg >= 180:
        theta_deg -= 360 # to show angle in range [-180, 180]
    
    print('Theta: {:.3f} [grader]'.format(theta_deg))
    return theta_deg

def std(std_vec):
    """
    Calculates and prints the standard deviation of the estimated angles.
    :param std_vec: list with theta values (for similar angles)
    :return: standard deviation
    """
    std = np.std(std_vec)
    print('Standardavvik: {:.3f} [grader]'.format(std))
    return std

def plot(data, INTERPOLATION_FACTOR, fs):
    """
    Plots the cross-correlation of two signals.
    :param data: data
    :param INTERPOLATION_FACTOR: interpolation factor
    :param fs: sampling frequency
    """    
    # Raw data
    plt.figure()
    plt.plot(data[1:,mik1], label='Mik 1')
    plt.plot(data[1:,mik2], label='Mik 2')
    plt.plot(data[1:,mik3], label='Mik 3')
    plt.title('Rådata fra mikrofonene')
    plt.xlabel('Målenummer [n]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Cross-correlation
    xcorr_data = all_xcorrs(data, INTERPOLATION_FACTOR, fs)
    xcorr = xcorr_data[0] # one of the cross-correlations
    range_interp = np.linspace(-(len(xcorr) - 1) / 2, (len(xcorr) - 1) / 2, len(xcorr)) # interpolated range
    plt.figure()
    plt.plot(range_interp, xcorr_data[0], label='Mik 21') # interpolated [NOT WORKING]
    plt.plot(range_interp, xcorr_data[1], label='Mik 31') # interpolated [NOT WORKING]
    plt.plot(range_interp, xcorr_data[2], label='Mik 32') # interpolated [NOT WORKING]
    plt.title('Krysskorrelasjon mellom mikrofonene')
    plt.xlabel('Målenummer [n]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Autocorrelation
    mik_11 = np.correlate(data[1:,mik1], data[1:,mik1], mode='same')
    range = np.linspace(-(len(mik_11) - 1) / 2, (len(mik_11) - 1) / 2, len(mik_11)) # original range
    plt.figure()
    plt.plot(range, mik_11, label='Mik 1')
    plt.title('Autokorrelasjon av mik 1')
    plt.xlabel('Målenummer [n]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # FFT
    # not implemented
    
def export_to_csv(filename, data):
    """
    Exports the data to a .csv file.
    :param filename: name of file
    :param data: list of data to append to file
    """
    with open(filename, 'a', newline='') as file: # a = append
        writer = csv.writer(file)
        writer.writerow(data)

def import_from_csv(filename):
    """
    Imports the data from a .csv file.
    :param filename: name of file
    :return: list of data
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def data_analysis_measurement():
    sample_period, data = get_data()
    xcorr_data, range = all_xcorrs(data, INTERPOLATION_FACTOR, fs)

    theta(xcorr_data)
    plot(data, INTERPOLATION_FACTOR, fs)
    

def export_data():
    filename_export = sys.argv[1]
    data_export = [theta, INTERPOLATION_FACTOR, INTERPOLATION_FACTOR*fs, 'Measurement 5']
    export_to_csv(filename_export, data_export) # append data to .csv file

def data_analysis_import():
    filename_import = sys.argv[1]
    data_import = import_from_csv(filename_import)
    
    theta_arr = []
    for i in range(1, len(data_import[:])):
        theta_arr.append(float(data_import[i][0]))
    print(theta_arr)
    
    # Standard deviation
    theta_std = np.std(theta_arr)
    print(f'Standardavvik for målinger på {(str(filename_import).split("-"))[0]} grader: {theta_std} [grader]')


##### MAIN #####
# [i] Argument in terminal (filename) depends on which function is used
data_analysis_measurement() # data analysis from measurements 
#export_data() # export analysed data
#data_analysis_import() # data analysis from imported data






