#! /bin/usr/python

import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir
from scipy import signal
import sys


def read_ligo_data():

    filename = 'LIGO Data/H-H1_LOSC_4_V1-1135210496-4096.hdf5'
    f = h5py.File(filename, 'r')
    data = np.array(f['strain']['Strain'])

    time = np.array(range(data.size))*list(f['strain']['Strain'].attrs.values())[3]


    f.close()

    return time, data


def read_waveform():
    path = 'CBC Waveforms'
    ignore_list = ['README_signal', 'signal_data.tar.gz']
    files_list = [f for f in listdir(path) if not(f in ignore_list)]

    index=np.random.randint(0,files_list.__len__())

    f = open(path+'/'+files_list[index])
    fin = f.read()
    fin = fin.split('\n')
    fsize = fin.__len__()

    time = np.zeros(fsize)
    data = np.zeros(fsize)
    for i in range(fsize):
        try:
            time[i] = float(fin[i].split(' ')[0])*10**-3
            data[i] = float(fin[i].split(' ')[1])
        except ValueError:
            time[i] = 0
            data[i] = 0

    f.close()

    return time, data


def data_match(x_data, x_time, y_time):

    # Matches the sampling rate of x to y

    y_sampf = y_time[2] - y_time[1]
    x_sampf = x_time[2] - x_time[1]
    out_data = signal.resample(x_data, int(x_sampf / y_sampf * x_data.__len__()))
    out_time = np.array(range(out_data.__len__())) * x_sampf / int(x_sampf / y_sampf * x_data.__len__())

    out_data = np.pad(out_data, (0, y_time.size-x_time.size), 'constant', constant_values=(0, 0))

    return out_time, out_data


def data_cut(x_data, x_time, seconds):

    # Cuts x down to the same size as y with some random offset

    length = int(seconds/(x_time[2]-x_time[1]))
    offset = np.random.randint(0, x_data.size-length)

    out_data = x_data[offset:length+offset]
    out_time = x_time[offset:length+offset]

    return out_time, out_data

# gainList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
# gainList = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
gainlist = 0.001
# start = int(sys.argv[1])
# end = int(sys.argv[2])

start = 1
end = 1001

for gain in gainList:
    for i in range(start, end):
        print(i)
        waveform_time, waveform_data = read_waveform()

        LIGO_time, LIGO_data = read_ligo_data()

        # wave_time, wave_data = data_match(gain*waveform_data, waveform_time, LIGO_time)
        # noise_time, noise_data = data_cut(LIGO_data, LIGO_time, 1)

        f=open('CBC Data/Gain'+str(gain)+'/signal'+str(i)+'.dat', 'w+')

        for j in range(4096):
            f.write(str(LIGO_time[j]) + ' ' + str(LIGO_data[j] + gain*waveform_data[j]) + '\n')

        f.close()

        f = open('CBC Data/Gain'+str(gain)+'/noise' + str(i) + '.dat', 'w+')

        for j in range(4096):
            f.write(str(LIGO_time[j]) + ' ' + str(LIGO_data[j]) + '\n')

        f.close()

