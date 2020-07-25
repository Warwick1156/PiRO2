import numpy as np
import torch

from time import time
from torchvision import datasets, transforms
from torch import nn, optim

class MNISTClassifier:

    def __init__(self):
        self.model = None
        self.optimizer = None

    def from_file(self, filename):
        self.model = torch.load(filename)
        self.optimizer = optim.Adam(self.model.parameters())

    def to_file(self, filename):
        torch.save(self.model, filename)

    def initialize(self):
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(dim=0))

        self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        trainset = datasets.MNIST('../data/MNIST/trainset', download=True, train=True, transform=transform)
        testset = datasets.MNIST('../data/MNIST/testset', download=True, train=False, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        epochs = 20
        criterion = nn.NLLLoss()

        for ep in range(epochs):
            loss_total = 0
            for images, labels in trainloader:
                images = images.view(images.shape[0], -1)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()

            print("Epoch {}, Loss: {}".format(ep, loss_total / len(trainloader)))

            if ep % 5 == 0:
                TP = 0
                TOTAL = 0
                for images, labels in testloader:
                    images = images.view(images.shape[0], -1)
                    TOTAL += len(labels)
                    for i in range(len(labels)):
                        TP += self.predict(images[i]) == labels[i]

                print("Validation: {}/{} ({}%)".format(TP, TOTAL, TP * 100 / TOTAL))

    def predict(self, image):
        with torch.no_grad():
            logps = self.model(image)

        ps = torch.exp(logps)
        probab = list(ps.numpy())
        return np.argmax(probab)