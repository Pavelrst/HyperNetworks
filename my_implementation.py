import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dataloaders import get_dataloaders
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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
        self.out1_shape = torch.Size([200, 3072])
        self.out2_shape = torch.Size([100, 200])
        self.out3_shape = torch.Size([10, 100])

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

        assert out1.shape == self.out1_shape
        assert out2.shape == self.out2_shape
        assert out3.shape == self.out3_shape

        return out1, out2, out3


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.primary_network = PrimaryNetwork()  # List of functionals
        self.hyper_network = HyperNetwork()

    def forward(self, x):
        w = self.hyper_network(x)
        x = self.primary_network(x, w)
        return x

class RefModel(nn.Module):
    '''
    Just a regular MLP for sanity check.
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 200)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(200, 100)
        self.act2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(100, 10)
        self.act3 = nn.Softmax(0)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x

def main():
    B = 250
    epochs = 10
    batches_per_epoch = -5 # for debuging
    trainloader, testloader = get_dataloaders(train_batch_size=B, test_batch_size=1)

    model = Model().to('cuda')
    optimizer = torch.optim.Adam(model.hyper_network.parameters(), lr=0.0001)
    #model = RefModel().to('cuda')
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    criterion = nn.CrossEntropyLoss()

    total_loss_list = []
    for epoch in range(epochs):
        epoch_loss_list = []
        for batch_idx, batch in enumerate(tqdm(trainloader)):
            if batches_per_epoch > 0:
                if batch_idx > batches_per_epoch:
                    break
            optimizer.zero_grad()
            inputs, labels = batch
            batchsize = batch[0].shape[0]
            for idx in range(batchsize):
                input = inputs[idx]
                label = labels[idx]
                input, label = Variable(input.cuda()), Variable(label.cuda())
                logits = model(input.view(3072))
                if idx == 0:
                    batch_loss = criterion(logits.unsqueeze(0), label.unsqueeze(0)) / batchsize
                else:
                    batch_loss += criterion(logits.unsqueeze(0), label.unsqueeze(0)) / batchsize
            batch_loss.backward()
            optimizer.step()
            epoch_loss_list.append(batch_loss.item())

        total_loss_list.append(np.mean(np.array(epoch_loss_list)))

    plt.plot(total_loss_list, label='training loss')
    plt.xlabel('epoch')
    plt.show()

    correct = 0.
    total = 0.
    for data in tqdm(testloader):
        image, label = data
        logits = model(Variable(image.cuda()).view(3072))
        predicted = torch.argmax(logits.cpu().data)
        total += label.size(0)
        correct += (predicted == label).sum()


    accuracy = (100. * correct) / total
    print('Accuracy: %.4f %%' % (accuracy))


main()