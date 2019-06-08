import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
import os
import h5py


# Data reading
def read_data(index, gain):

    f = open('NS Data/Gain'+str(gain)+'/signal' + str(index) + '.dat', 'r')

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

    f = open('NS Data/Gain'+str(gain)+'/noise' + str(index) + '.dat', 'r')
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
        # self.conv1 = nn.Conv1d(1, 4096, 32)
        # self.conv2 = nn.Conv1d(4096, 1024, 16)
        # self.conv3 = nn.Conv1d(1024, 256, 8)
        # self.conv4 = nn.Conv1d(256, 64, 4)
        # self.pool = nn.MaxPool1d(4)
        # self.fc1 = nn.Linear(896, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 1)
        # self.out = nn.Sigmoid()

        self.conv1=nn.Conv1d(1, 1, 4096)
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(4096, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(1, 1, 4096)
        x = F.relu(self.pool(self.conv1(x)))
        # x = F.relu(self.pool(self.conv2(x)))
        # x = F.relu(self.pool(self.conv3(x)))
        # x = F.relu(self.pool(self.conv4(x)))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.out(x)
        return x


# Gain to train the CNN on
gainTrain = 0.01

x_train = np.zeros(4096)
y_train = np.array(0)
x_test = np.zeros(4096)
y_test = np.array(0)

# Splitting data into test and training sets
for i in range(1, int(len(os.listdir('./NS Data/Gain'+str(gainTrain)+'/'))/2+1)):
    if i <= len(os.listdir('./NS Data/Gain' + str(gainTrain) + '/')) / 4:

        tim, wave_data, noise_data = read_data(i, gainTrain)

        with np.errstate(divide='raise'):
            # Data stacking, 1 GW, 0 noise
            x_train = np.column_stack((x_train, wave_data))
            y_train = np.append(y_train, 1)
            x_train = np.column_stack((x_train, noise_data))
            y_train = np.append(y_train, 0)

    if i > len(os.listdir('./NS Data/Gain' + str(gainTrain) + '/')) / 4:

        tim, wave_data, noise_data = read_data(i, gainTrain)

        with np.errstate(divide='raise'):
            # Data stacking, 1 GW, 0 noise
            x_test = np.column_stack((x_test, wave_data))
            y_test = np.append(y_test, 1)
            x_test = np.column_stack((x_test, noise_data))
            y_test = np.append(y_test, 0)


# Cut off first zero, normalize, and turn into tensor
train_data = torch.from_numpy((x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))/np.std(x_train[:, 1:])).float()
test_data = torch.from_numpy((x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))/np.std(x_test[:, 1:])).float()
train_labels = torch.from_numpy(y_train[1:]).float()
test_labels = torch.from_numpy(y_test[1:]).float()

# Net initialization, loss and optimizer definition
# net = Net()
net = torch.load("1Mpc_BNS_CNN.pb")
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=5e-5, momentum=1e-5)

# Net training
epochLim = 10

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
            print('[%d, %5d] loss: %.5f' %
                (epoch + 1, i + 1, running_loss))
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

if max(testAcc)>0.5:
    torch.save(net, str(1/gainTrain)+'Mpc_BNS_CNN.pb')

# Apply trained net to other data sets with different gains
# gainList = np.array((0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1.0))
gainList = np.array((0.01, 0.02, 0.03, 1.0))
gainAcc = np.zeros(gainList.size)
gainIndex = 0

for gain in gainList:

    x_test = np.zeros(4096)
    y_test = np.array(0)

    for i in range(1, int(len(os.listdir('./NS Data/Gain'+str(gain)+'/'))/2+1)):

            tim, wave_data, noise_data = read_data(i, gain)

            with np.errstate(divide='raise'):
                # Data stacking, 1 GW, 0 noise
                x_test = np.column_stack((x_test, wave_data))
                y_test = np.append(y_test, 1)
                x_test = np.column_stack((x_test, noise_data))
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

plt.figure()
plt.plot(gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Gain')
plt.legend(('Trained at '+str(1/gainTrain)+' Mpc'))
plt.grid(True)
plt.draw()
plt.savefig('AccuracyNS.pdf')

plt.figure()
plt.plot(1/gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Distance (Mpc)')
plt.legend(('Trained at '+str(1/gainTrain)+' Mpc'))
plt.grid(True)
plt.draw()
plt.savefig('AccuracyDistanceNS.pdf')

plt.figure()
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.legend(('Training', 'Testing'))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.draw()
plt.savefig('NNTrainingNS.pdf')
plt.show()