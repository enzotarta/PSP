from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        z = []
        x = self.conv1(x)
        z.append(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        z.append(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        z.append(x)
        x = F.relu(x)
        x = self.fc2(x)
        z.append(x)
        return x, z

def train(args, model, device, train_loader, optimizer, epoch, mode=0, gamma=0.1, startlr = 0.1, l1_reg = 0):
    model.train()
    global_loss = 0
    global_regu = 0 #presynaptic potential

    for batch_idx, (data, target) in enumerate(train_loader):
        	
        data, target = data.to(device), target.to(device)
        	
        optimizer.zero_grad()
        output = model(data)

        if isinstance(output,tuple):
            output,z = output
        else:
            z=[]
 
        loss = F.cross_entropy(output, target, reduction='mean')
        global_loss += loss

        for act in z:
            q = torch.mean(act**2, dim=0)
            if len(q.size()) > 1:
                q = torch.mean(q, dim=[1, 2])

            PSP = torch.sum(q)
            global_regu += PSP.item()
            if mode == 0:
                loss += gamma/(len(z)*q.numel()) * PSP


        if mode == 2:
            for param in model.parameters():
                loss += l1_reg * torch.sum(torch.abs(param))
            

        loss.backward()
        optimizer.step()

    return global_loss / len(train_loader.dataset), global_regu / len(train_loader.dataset)

def test(args, model, device, test_loader, epoch, train_loss, train_regu):
    model.eval()
    test_loss = 0
    test_regu = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output,tuple):
                output,z = output
            else:
                z=[]

            for act in z:
                q = torch.mean(act**2, dim=0)
                if len(q.size()) > 1:
                    q = torch.mean(q, dim=[1, 2])
                test_regu += 1.0/(len(z) * q.numel())*torch.sum(q)
            test_loss += F.cross_entropy(output, target, reduction='mean').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_regu /= len(test_loader.dataset)

    perc = 100. * correct / len(test_loader.dataset)
    print('Epoch:{} Train loss: {:.8f} Regu: {:.8f} Test loss: {:.8f} Regu: {:.8f} Error {:.2f} % '.format(
        epoch, train_loss, train_regu, test_loss, test_regu, 100 - perc ))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
                        help='number of epochs to train (default: 450)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--mode', type=int, default=0, metavar='M',
                        help='expriment modes, mode:0 -> PSP, mode:1 -> L2, mode:2 -> L1, mode:3 -> Dropout, mode:4 -> NoRegu')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='number of seeds - stats')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    
    if use_cuda:
        device = torch.device(args.cuda)
        torch.cuda.set_device(device)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        kwargs = {'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    for i in range(args.n_seeds):
        seed = random.randrange(1, 9999)

        print('\n*****************\nLOG TRAINING NUMBER: ',str(i),'\n*****************\n')

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        model = models.resnet18(pretrained=False).to(device)
        
        weight_decay = 0.0
        gamma = 0
        l1 = 0
        # mode:0 -> PSP, mode:1 -> L2, mode:2 -> L1, mode:3 -> Dropout, mode:4 -> NoReg
        if args.mode == 0:
            gamma = 1e-2
        elif args.mode == 1: 
            weight_decay = 1e-4
        elif args.mode == 2:
            l1 = 1e-5
        elif args.mode == 3:
            model.fc = nn.Sequential( nn.Dropout(p=0.1), model.fc)
        

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = weight_decay)
        ms = [150,250,350]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.1)

        for epoch in range(1, args.epochs + 1):
            scheduler.step()
            tr_loss, tr_regu = train(args, model, device, train_loader, optimizer, epoch, mode = args.mode, gamma=gamma, startlr = args.lr, l1_reg=l1)
            test(args, model, device, test_loader, epoch, tr_loss, tr_regu)

            #if epoch%10 == 0:
            #    torch.save(model.state_dict(), './save/models/LeNet5_MNIST_mode_'+ str(args.mode) +'_seedn_'+ str(i) +'_ep_'+ str(epoch) +'.pt')

if __name__ == '__main__':
    main()
