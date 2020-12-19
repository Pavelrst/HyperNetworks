import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torch


def get_dataloaders(train_batch_size=128, test_batch_size=1):
    ########### Data Loader ###############

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


class PrimaryNetwork(nn.Module):
    '''
    MLP for classification
    with functionals.
    '''
    def __init__(self):
        super().__init__()
        self.fc1_shape = torch.Size([200, 3072])
        self.fc2_shape = torch.Size([100, 200])
        self.fc3_shape = torch.Size([10, 100])

    def forward(self, x, w):
        w1, w2, w3 = w

        assert w1.shape == self.fc1_shape
        assert w2.shape == self.fc2_shape
        assert w3.shape == self.fc3_shape

        x = F.linear(x, w1)
        x = F.relu(x)
        x = F.linear(x, w2)
        x = F.relu(x)
        x = F.linear(x, w3)
        x = F.relu(x)
        return x


class HyperNetwork(nn.Module):
    '''
    Hypernetwork, predicts weights
    based on input vector
    '''
    def __init__(self):
        super().__init__()
        self.out1_shape = torch.Size([-1, 200, 3072])
        self.out2_shape = torch.Size([-1, 100, 200])
        self.out3_shape = torch.Size([-1, 10, 100])

        self.fc1 = nn.Linear(in_features=3072, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)

        # Heads for prediction of weight of each layer
        self.head1 = nn.Linear(in_features=100, out_features=3072*200)
        self.head2 = nn.Linear(in_features=100, out_features=200*100)
        self.head3 = nn.Linear(in_features=100, out_features=100*10)

    def forward(self, x):
        # TODO: do we need RELUS?
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        w1 = self.head1(x)
        w2 = self.head2(x)
        w3 = self.head3(x)

        out1 = w1.view(self.out1_shape)
        out2 = w2.view(self.out2_shape)
        out3 = w3.view(self.out3_shape)

        # Mean reduce the weights if they batch,
        # or only remove batch dim if x single sample
        out1 = torch.mean(out1, dim=0)
        out2 = torch.mean(out2, dim=0)
        out3 = torch.mean(out3, dim=0)

        return out1, out2, out3


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.primary_network = PrimaryNetwork()  # List of functionals
        self.hyper_network = HyperNetwork()

    def forward(self, x_batch):
        x_batch_reduced = torch.mean(x_batch, dim=0)
        w = self.hyper_network(x_batch_reduced)
        logits_batch = self.primary_network(x_batch, w)
        return logits_batch


class RefModel(nn.Module):
    '''
    Just a regular MLP for sanity check.
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 200)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 100)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x



def main():
    batchsize = 512
    epochs = 2
    trainloader, testloader = get_dataloaders(train_batch_size=batchsize, test_batch_size=batchsize)

    model = Model().to('cuda')
    optimizer = torch.optim.Adam(model.hyper_network.parameters(), lr=0.001)
    #model = RefModel().to('cuda')
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    total_loss_list = []
    for epoch in range(epochs):
        epoch_loss_list = []
        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            inputs = inputs.view(-1, 3072)

            logits = model(inputs)
            batch_loss = criterion(logits, labels)
            batch_loss.backward()
            optimizer.step()
            epoch_loss_list.append(batch_loss.item())

        total_loss_list.append(np.mean(np.array(epoch_loss_list)))

        with torch.no_grad():
            correct = 0.
            total = 0.
            acc_list = []
            for data in tqdm(testloader):
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                inputs = inputs.view(-1, 3072)

                logits = model(inputs)
                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum()

            accuracy = ((100. * correct) / total).cpu().detach().item()
            acc_list.append(accuracy)
            print('Epoch {} Accuracy: {:.2f}'.format(epoch, accuracy))

    plt.plot(total_loss_list, label='training loss')
    plt.plot(acc_list, label='val acc')
    plt.xlabel('epoch')
    plt.show()
main()