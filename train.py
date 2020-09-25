import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import time
import argparse

from nmnist import nmnist
from LISNN import LISNN

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-gpu', type = int, default = 0)
parser.add_argument('-seed', type = int, default = 3154)
parser.add_argument('-epoch', type = int, default = 100)
parser.add_argument('-batch_size', type = int, default = 100)
parser.add_argument('-learning_rate', type = float, default = 1e-3)
parser.add_argument('-dts', type = str, default = 'MNIST')
parser.add_argument('-if_lateral', type = bool, default = True)

opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True

test_scores = []
train_scores = []
save_path = './' + opt.model + '_' + opt.dts + '_' + str(opt.seed)
if not os.path.exists(save_path):
    os.mkdir(save_path)

if opt.dts == 'MNIST':
    train_dataset = dsets.MNIST(root = './data/mnist/', train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = dsets.MNIST(root = './data/mnist/', train = False, transform = transforms.ToTensor())
elif opt.dts == 'Fashion-MNIST':
    train_dataset = dsets.FashionMNIST(root = './data/fashion/', train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = dsets.FashionMNIST(root = './data/fashion/', train = False, transform = transforms.ToTensor())
elif opt.dts == 'NMNIST':
    train_dataset = nmnist(datasetPath = 'nmnist/Train/', sampleFile = 'nmnist/Train.txt', samplingTime = 1.0, sampleLength = 20)
    test_dataset = nmnist(datasetPath = 'nmnist/Test/', sampleFile = 'nmnist/Test.txt', samplingTime = 1.0, sampleLength = 20)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = opt.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = opt.batch_size, shuffle = False)

model = LISNN(opt)
model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

def train(epoch):
    model.train()
    scheduler.step()
    start_time = time.time()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = Variable(images.cuda())
        one_hot = torch.zeros(opt.batch_size, model.fc[-1]).scatter(1, labels.unsqueeze(1), 1)
        labels = Variable(one_hot.cuda())

        outputs = model(images)
        loss = loss_function(outputs, labels)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_dataset) // (opt.batch_size * 6)) == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Time: %.2f' % (epoch + 1, opt.epoch, i + 1, len(train_dataset) // opt.batch_size, total_loss, time.time() - start_time))
            start_time = time.time()
            total_loss = 0

def eval(epoch, if_test):
    model.eval()
    correct = 0
    total = 0
    if if_test:
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            outputs= model(images)
            pred = outputs.max(1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum()

        acc = 100.0 * correct.item() / total
        print('Test correct: %d Accuracy: %.2f%%' % (correct, acc))
        test_scores.append(acc)
        if acc > max(test_scores):
            save_file = str(epoch) + '.pt'
            torch.save(model, os.path.join(save_path, save_file))
    else:
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            outputs = model(images)
            pred = outputs.max(1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum()

        acc = 100.0 * correct.item() / total
        print('Train correct: %d Accuracy: %.2f%%' % (correct, acc))
        train_scores.append(acc)

def main():
    for epoch in range(opt.epoch):
        train(epoch)
        if (epoch + 1) % 2 == 0:
            eval(epoch, if_test = True)
        if (epoch + 1) % 20 == 0:
            eval(epoch, if_test = False)
        if (epoch + 1) % 20 == 0:
            print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            print('Best Train Accuracy in %d: %.2f%%' % (epoch + 1, max(train_scores)))

if __name__ == '__main__':
    main()