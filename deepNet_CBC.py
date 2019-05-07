import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
import os
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Data reading
def read_data(index, gain):

    f = open('CBC Data/Gain'+str(gain)+'/signal' + str(index) + '.dat', 'r')

    lines = f.read().split('\n')
    l = lines.__len__() - 1
    tim = np.zeros(l)
    wave_data = np.zeros(l)
    noise_data = np.zeros(l)

    for i in range(0, l):

        if not (np.isnan(float(lines[i].split(' ')[1]))):
            tim[i] = float(lines[i].split(' ')[0])
            wave_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    f = open('CBC Data/Gain'+str(gain)+'/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    l = lines.__len__() - 1
    for i in range(0, l):
        if not(np.isnan(float(lines[i].split(' ')[1]))):
            noise_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    return tim, wave_data, noise_data


# Convolutional neural net definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 32)
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(3840, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(1, 1, 513)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 3840)
        x = self.fc1(x)
        x = self.out(x)
        return x


# Gain to train the CNN on
gain = 0.5

x_train = np.zeros(513)
y_train = np.array(0)
x_test = np.zeros(513)
y_test = np.array(0)

# Splitting data into test and training sets
for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2+1):
    if i <= len(os.listdir('./CBC Data/Gain' + str(gain) + '/')) / 4:

        tim, wave_data, noise_data = read_data(i, gain)

        # Power spectra
        f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096/4)
        f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096/4)

        with np.errstate(divide='raise'):
            # Data stacking, 1 GW, 0 noise
            x_train = np.column_stack((x_train, np.log10(P2.T)))
            y_train = np.append(y_train, 1)
            x_train = np.column_stack((x_train, np.log10(P1.T)))
            y_train = np.append(y_train, 0)

    if i > len(os.listdir('./CBC Data/Gain' + str(gain) + '/')) / 4:

        tim, wave_data, noise_data = read_data(i, gain)

        # Power spectra
        f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096/4)
        f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096/4)

        with np.errstate(divide='raise'):
            # Data stacking, 1 GW, 0 noise
            x_test = np.column_stack((x_test, np.log10(P2.T)))
            y_test = np.append(y_test, 1)
            x_test = np.column_stack((x_test, np.log10(P1.T)))
            y_test = np.append(y_test, 0)


# Cut off first zero, normalize, and turn into tensor
train_data = torch.from_numpy((x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))/np.std(x_train[:, 1:])).float()
test_data = torch.from_numpy((x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))/np.std(x_test[:, 1:])).float()
train_labels = torch.from_numpy(y_train[1:]).float()
test_labels = torch.from_numpy(y_test[1:]).float()

# Net initialization, loss and optimizer definition
net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=10**-6, momentum=0)

# Net training
epochLim = 25

testAcc = np.zeros(epochLim)
trainAcc = np.zeros(epochLim)
for epoch in range(epochLim):

    running_loss = 0.0
    for i in range(0, len(train_data)):

        optimizer.zero_grad()

        output = net(train_data[i])
        loss = criterion(output.view(-1), train_labels[i].view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    correct = 0.0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(float(output.data))
            correct += (predicted == test_labels[j]).item()
    print('Test accuracy: %d %%' % (
            100 * correct / test_labels.size(0)))
    testAcc[epoch] = correct / test_labels.size(0)

    correct = 0.0
    with torch.no_grad():
        for j in range(len(train_data)):
            output = net(train_data[j])
            predicted = round(float(output.data))
            correct += (predicted == train_labels[j]).item()
    print('Train accuracy: %d %%' % (
            100 * correct / train_labels.size(0)))
    trainAcc[epoch] = correct / train_labels.size(0)

print('Finished Training')

# Apply trained net to other data sets with different gains
gainList = np.array((0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0))
gainAcc = np.zeros(gainList.size)
gainIndex = 0

for gain in gainList:

    x_test = np.zeros(513)
    y_test = np.array(0)

    for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2+1):

            tim, wave_data, noise_data = read_data(i, gain)
            # Power spectra
            f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096/4)
            f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096/4)

            with np.errstate(divide='raise'):
                # Data stacking, 1 GW, 0 noise
                x_test = np.column_stack((x_test, np.log10(P2.T)))
                y_test = np.append(y_test, 1)
                x_test = np.column_stack((x_test, np.log10(P1.T)))
                y_test = np.append(y_test, 0)

    # Normalize and convert to tensor
    test_data = torch.from_numpy((x_test[:, 1:].T - np.mean(x_test[:, 1:], axis=1)) / np.std(x_test[:, 1:])).float()
    test_labels = torch.from_numpy(y_test[1:]).float()

    correct = 0.0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(float(output.data))
            correct += (predicted == test_labels[j]).item()

    print('Accuracy on '+str(1/gain)+' Mpc dataset: %d %%' % (
            100 * correct / test_labels.size(0)))
    gainAcc[gainIndex] = correct / test_labels.size(0)

    gainIndex += 1


# Search data with known GW events
eventCounter = 0
for filename in os.listdir('GW Events'):

    # Read data
    f = h5py.File('GW Events/'+filename, 'r')
    data = np.array(f['strain']['Strain'])
    f.close()

    predicted = 0.0
    # Loop in 1 sec chunks
    for i in range(0, len(data), 4096):
        # Power spectra
        f, P = signal.welch(data[i:i+4096], fs=4096, nperseg=4096/4)
        # Normalize and turn into tensor
        x = torch.from_numpy((np.log10(P) - np.mean(np.log10(P))) / np.std(np.log10(P))).float()
        # Apply net
        with torch.no_grad():
            output = net(x)
            predicted += round(float(output.data))

        if (round(float(output.data)) == 1.0):
            f = open('Found Events/event' + str(eventCounter) + '.dat', 'w+')

            for j in range(4096):
                f.write(str(data[i+j]) + '\n')

            f.close()
            eventCounter += 1

    print('GW events found: ' + str(predicted))

# Apply a selection of simple methods from sklearn to the data to compare

NNAcc = np.zeros(gainList.size)
NearNAcc = np.zeros(gainList.size)
logRegAcc = np.zeros(gainList.size)
SVMAcc = np.zeros(gainList.size)

gainIndex = 0

gain = 0.5

x_train = np.zeros(513)
y_train = np.array(0)
for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2+1):
    if i <= len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/4:

        tim, wave_data, noise_data = read_data(i, gain)

        # Power spectra
        f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096/4)
        f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096/4)

        with np.errstate(divide='raise'):
            # Data stacking, 1 GW, -1 noise
            x_train = np.column_stack((x_train, np.log10(P2.T)))
            y_train = np.append(y_train, 1)
            x_train = np.column_stack((x_train, np.log10(P1.T)))
            y_train = np.append(y_train, -1)

# Cut off first zero, normalize, and turn into tensor
x_train = (x_train[:, 1:].T - np.mean(x_train[:, 1:], axis=1)) / np.std(x_train)
y_train = y_train[1:]

# Train and test algorithms
logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000)
logReg.fit(x_train, y_train.ravel())

svmAlg = svm.SVC(gamma='scale')
svmAlg.fit(x_train, y_train.ravel())

NearN = KNeighborsClassifier(20)
NearN.fit(x_train, y_train.ravel())

NN = MLPClassifier(max_iter=10000)
NN.fit(x_train, y_train.ravel())

for gain in gainList:

    print(str(gainList[gainIndex]))
    x_train = np.zeros(513)
    y_train = np.array(0)
    x_test = np.zeros(513)
    y_test = np.array(0)
    for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2+1):

        tim, wave_data, noise_data = read_data(i, gain)

        # Power spectra
        f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096/4)
        f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096/4)

        with np.errstate(divide='raise'):
            # Data stacking, 1 GW, -1 noise
            x_test = np.column_stack((x_test, np.log10(P2.T)))
            y_test = np.append(y_test, 1)
            x_test = np.column_stack((x_test, np.log10(P1.T)))
            y_test = np.append(y_test, -1)

    # Cut off first zero, normalize, and turn into tensor
    x_test = (x_test[:, 1:].T - np.mean(x_test[:, 1:], axis=1)) / np.std(x_test)
    y_test = y_test[1:]

    # Train and test algorithms
    logRegAcc[gainIndex] = logReg.score(x_test, y_test)

    SVMAcc[gainIndex] = svmAlg.score(x_test, y_test)

    NearNAcc[gainIndex] = NearN.score(x_test, y_test)

    NNAcc[gainIndex] = NN.score(x_test, y_test)

    print(NNAcc[gainIndex])

    gainIndex += 1


plt.figure()
plt.plot(gainList, logRegAcc)
plt.plot(gainList, SVMAcc)
plt.plot(gainList, NearNAcc)
plt.plot(gainList, NNAcc)
plt.plot(gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Gain')
plt.legend(('Logistic Regression', 'SVM', 'Nearest Neighbor', 'Neural Network', 'Convolutional Neural Network'))
plt.grid(True)
plt.draw()
plt.savefig('AccuracyCBC.pdf')

plt.figure()
plt.plot(1/gainList, logRegAcc)
plt.plot(1/gainList, SVMAcc)
plt.plot(1/gainList, NearNAcc)
plt.plot(1/gainList, NNAcc)
plt.plot(1/gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Distance (Mpc)')
plt.legend(('Logistic Regression', 'SVM', 'Nearest Neighbor', 'Neural Network', 'Convolutional Neural Network'))
plt.grid(True)
plt.draw()
plt.savefig('AccuracyDistanceCBC.pdf')

plt.figure()
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.legend(('Training', 'Testing'))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.draw()
plt.savefig('NNTrainingCBC.pdf')
plt.show()