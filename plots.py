import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA

def read_data(index):

    f = open('NS Data/Gain1.0/signal' + str(index) + '.dat', 'r')

    lines = f.read().split('\n')
    l = lines.__len__() - 1
    time = np.zeros(l)
    wave_data = np.zeros(l)
    noise_data = np.zeros(l)

    for i in range(0, l):
        time[i] = float(lines[i].split(' ')[0])
        wave_data[i] = float(lines[i].split(' ')[1])

    f.close()

    f = open('NS Data/Gain1.0/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    for i in range(0, l):
        noise_data[i] = float(lines[i].split(' ')[1])

    f.close()

    return time, wave_data, noise_data



time, wave_data, noise_data = read_data(3)#np.random.randint(1,1000)

time = np.linspace(1, 4096, 4096).T/4096;

sampF = 4096.0

plt.figure(1)
plt.plot(time, wave_data, time, noise_data)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid(True,'both')
plt.draw()

plt.figure(6)
plt.plot(time, wave_data - noise_data)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid(True,'both')
plt.draw()

f, t, Sxx = signal.spectrogram(wave_data, sampF,'hann',512,500)

plt.figure(2)
plt.pcolormesh(t, f, (Sxx),vmax=4*10**-45)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.yscale('log')
plt.ylim([10**2, 10**3])
plt.draw()

f, t, Sxx = signal.spectrogram(noise_data, sampF,'hann',512,500)

plt.figure(3)
plt.pcolormesh(t, f, (Sxx),vmax=4*10**-45)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.yscale('log')
plt.ylim([10**2, 10**3])
plt.draw()

plt.figure(4)
f, P1 = signal.welch(wave_data, fs=sampF, nperseg=4096/4)
plt.loglog(f, P1)
f, P2 = signal.welch(noise_data, fs=sampF, nperseg=4096/4)
plt.loglog(f, P2)

# plt.xlim([10, 10**3])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [strain^2/Hz]')
plt.grid(True,'both')
plt.draw()

# x_train = np.concatenate((P1, P2)).reshape(-1,1)
#
# pca = PCA(n_components=2).fit(x_train.T)
#
# plt.figure(5)
#
# for i in range(len(x_train)/2):
#     plt.plot(pca.transform(x_train)[i, 0], pca.transform(x_train)[i, 1], '.')
#
# for i in range(len(x_train) / 2,len(x_train)):
#     plt.plot(pca.transform(x_train)[i, 0], pca.transform(x_train)[i, 1], 'o')
#
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.draw()

plt.show()
