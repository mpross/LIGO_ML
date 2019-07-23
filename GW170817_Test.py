import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convolutional neural net definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv1d(1, 4096, 4096)
        # self.conv2 = nn.Conv1d(4096, 2048, 2048)
        # self.conv3 = nn.Conv1d(2048, 1024, 1024)
        # self.conv4 = nn.Conv1d(1024, 512, 512)
        # self.pool = nn.MaxPool1d(4)
        # self.fc1 = nn.Linear(896, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 1)
        # self.out = nn.Sigmoid()
        #
        self.conv1 = nn.Conv1d(1, 4096, 512)
        # self.conv2 = nn.Conv1d(4096, 512, 512)
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(4096*896, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(1, 1, 4096)
        x = F.relu(self.pool(self.conv1(x)))
        # x = F.relu(self.pool(self.conv2(x)))
        # print(x.shape)
        # x = F.relu(self.pool(self.conv3(x)))
        # print(x.shape)
        # x = F.relu(self.pool(self.conv4(x)))
        # print(x.shape)
        x = x.view(-1, 4096*896)
        x = self.fc1(x)
        x = self.out(x)
        return x

filename = 'GW170817/H-H1_GWOSC_4KHZ_R1-1187008867-32.hdf5'
f = h5py.File(filename, 'r')
data = np.array(f['strain']['Strain'])

time = np.array(range(data.size))*list(f['strain']['Strain'].attrs.values())[3]

f.close()

net = torch.load("100.0Mpc_BNS_CNN.pb")

prediction=np.zeros((round(len(time)/4096.0),1))

for index in range(1, round(len(time)/4096.0)):

    cut=torch.from_numpy(data[index*4096:(index+1)*4096]).float()

    with torch.no_grad():
        output= net(cut)
        prediction[index]=round(float(output.data))


sampF=4096.0

plt.figure(1)
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid(True,'both')
plt.draw()

f, t, Sxx = signal.spectrogram(data, sampF,'hann',512,500)

plt.figure(2)
plt.pcolormesh(t, f, (Sxx),vmax=10**-45)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.yscale('log')
plt.ylim([10**2, 10**3])
plt.xlim([13, 16])
plt.draw()

plt.figure(4)
f, P1 = signal.welch(data, fs=sampF, nperseg=4096/4)
plt.loglog(f, P1)

# plt.xlim([10, 10**3])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [strain^2/Hz]')
plt.grid(True,'both')
plt.draw()

plt.figure(5)
plt.plot(prediction)

plt.show()