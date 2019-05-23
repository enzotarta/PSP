from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np

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

def train(args, model, device, train_loader, optimizer, epoch, mode=0, gamma_mul=0.1, startlr = 0.1):
    model.train()
    global_loss = 0
    global_regu = 0
    gamma = gamma_mul
    #for param_group in optimizer.param_groups:
        #if param_group['lr'] != startlr:
    	    #gamma = 0.0 #
        #gamma = param_group['lr'] * gamma_mul


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
        ##add L2norm contribution here

        if gamma>0.0:
            for act in z:
                #q = torch.mean(act**2, dim=0)
                #loss += torch.sum(q) * gamma * 1/q.numel()
                q = torch.mean(act**2, dim=0)
                if len(q.size()) > 1:
                    q = torch.mean(q, dim=[1, 2])
                this_loss = gamma/(len(z)*q.numel())*torch.sum(q)
                loss += this_loss
                global_regu += this_loss
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
            output , z = model(data)
            for act in z:
                #q = torch.mean(act**2, dim=0)
                #loss += torch.sum(q) * gamma * 1/q.numel()
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
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--mode', type=int, default=0, metavar='M',
                        help='expriment modes')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/enzo/data/MNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/enzo/data/MNIST', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = Net().to(device)
    #checkpoint = torch.load('40.pt')
    #model.load_state_dict(checkpoint)
    #a = model.conv2.weight.data.cpu()
    #a = torch.reshape(a, [a.numel()]).numpy()
    #binwidth = 0.0025
    #plt.hist(a, bins=np.arange(min(a), max(a) + binwidth, binwidth), alpha=0.5, label='PSP')


    #checkpoint = torch.load('wd/40.pt')
    #model.load_state_dict(checkpoint)
    #a = model.conv2.weight.data.cpu()
    #a = torch.reshape(a, [a.numel()]).numpy()
    #plt.hist(a, bins=np.arange(min(a), max(a) + binwidth, binwidth), alpha=0.5, label='L2')


    #plt.legend()
    #plt.xlabel("w value")
    #plt.title("conv2 layer in LeNet5")
    #plt.show()
    #error()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = 0.0)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_regu = train(args, model, device, train_loader, optimizer, epoch, mode = 0, gamma_mul=0.0, startlr = args.lr)
        test(args, model, device, test_loader, epoch, tr_loss, tr_regu)

        if epoch%10 == 0:
            torch.save(model.state_dict(), str(epoch) + '.pt')

if __name__ == '__main__':
    main()
