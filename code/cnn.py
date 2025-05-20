#import seaborn as sns
#import tensorflow_io as tfio
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
import pickle as pk

with open('spectro_data', 'rb') as data:
    spectro_accent = pk.load(data)

with open('spectro_data_test', 'rb') as data:
    spectro_accent_test = pk.load(data)

class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transforms = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        spectro = data[0]
        tensor = torch.from_numpy(spectro).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0) # adding dimensions so it fits the expect shape(batch_size, channels, H, W) for conv2d
        label = data[1]
        return tensor, label

trainset = CustomImageDataset(spectro_accent)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = CustomImageDataset(spectro_accent_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('entering first conv')
        x = self.pool(F.relu(self.conv1(x)))
        print('entering second conv')
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        print('entering first relu')
        x = F.relu(self.fc1(x))
        print('entering second relu')
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        print('finished one pass')
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

print('Finished Training, onto testing')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))