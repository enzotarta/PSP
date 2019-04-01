from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
import numpy as np
import random

#import resnetMBN
#import resnetRBN
#import resnetKua


def train(args, model, device, train_loader, optimizer, opt_regu, epoch, mode=0, gamma_mul=0.1, startlr = 0.1):
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
        loss.backward()
        optimizer.step()

        ##REGU STEP
        opt_regu.zero_grad()
        output = model(data)
        if isinstance(output,tuple):
            output,z = output
        else:
            z=[]
        loss = 0
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
        opt_regu.step()
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
    parser.add_argument('--batch_size', type=int, default=128, metavar='B',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='TB',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 550)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD',
                        help='expriment modes')
    parser.add_argument('--gamma', type=float, default=0.001, metavar='GM',
                        help='gamma')
    parser.add_argument('--mode', type=int, default=0, metavar='MD',
                        help='expriment modes')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='L',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Cuda Device')
    parser.add_argument('--sched', action='store_true', default=False,
                        help='Scheduler')
    parser.add_argument('--name', type=str, default="CIFAR10_net",
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.cuda if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}#, 'pin_memory': True
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/tarta/data/CIFAR10', train=True, download=True,
                       transform=transforms.Compose([
                           #transforms.RandomRotation(15, expand=True),
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/tarta/data/CIFAR10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    if args.mode==0:
      import resnetCIFAR
      model = resnetCIFAR.resnet20().to(device)
    elif args.mode==1:
      import resnetCIFAR_RBN
      model = resnetCIFAR_RBN.resnetRBN20().to(device)
    elif args.mode==2:
      import resnetCIFAR_MBN
      model = resnetCIFAR_MBN.resnetMBN20().to(device)
    elif args.mode == 4:
      import resnetKua
      model = resnetKua.ResNet18().to(device)
    elif args.mode == 5:
      import resnetKuaRBN
      model = resnetKuaRBN.ResNet18().to(device)
    elif args.mode == 55:
      import resnetKuaMBN
      model = resnetKuaMBN.ResNet18().to(device)
    elif args.mode == 6:
      import densnet
      model = densnet.DenseNet121().to(device)
    elif args.mode == 7:
      import densnetRBN
      model = densnetRBN.DenseNet121().to(device)
    elif args.mode == 8:
      import ALL_CNN_C
      model = ALL_CNN_C.ALL_CNN_C().to(device)
    elif args.mode == 9:
      import ALL_CNN_C_RBN
      model = ALL_CNN_C_RBN.ALL_CNN_C().to(device)
    elif args.mode == 10:
      import rKREG
      model = rKREG.ResNet18().to(device)
    elif args.mode == 11:
      import rKREG_local
      model = rKREG_local.ResNet18().to(device)
    elif args.mode == 12:
      import rKREG_local2
      model = rKREG_local2.ResNet18().to(device)
    elif args.mode == 13:
      import ALL_CNN_C_RBN_local
      model = ALL_CNN_C_RBN_local.ALL_CNN_C().to(device)
    elif args.mode == 14:
      import densnetRBN_local
      model = densnetRBN_local.DenseNet121().to(device)
    elif args.mode == 15:
      import densnetRBN_nolast
      model = densnetRBN_nolast.DenseNet121().to(device)
    elif args.mode == 16:
      import rKREG_nolast_local2
      model = rKREG_nolast_local2.ResNet18().to(device)
    elif args.mode == 17:
      import ALL_CNN_C_RBN_nolast_local
      model = ALL_CNN_C_RBN_nolast_local.ALL_CNN_C().to(device)
    elif args.mode == 18:
      import ALL_CNN_C_MBN
      model = ALL_CNN_C_MBN.ALL_CNN_C().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum , weight_decay=args.weight_decay)
    optimizer_regu = optim.SGD(model.parameters(), lr=args.lr, momentum=0 , weight_decay=0)
    ms = [150, 250, 350]
    if args.sched:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.1) #[82,123,164]
        scheduler_REGU = optim.lr_scheduler.MultiStepLR(optimizer_regu, milestones=ms, gamma=0.1) #[82,123,164]

    gamma = args.gamma
    print('# ', args)
    for epoch in range(1, args.epochs + 1):
        if args.sched:
            scheduler.step()
            scheduler_REGU.step()
        #if epoch == ms[1]+1:
        #    for param_group in optimizer.param_groups:
        #        param_group['weight_decay'] = 0.0005
        #    args.gamma=0.0

        tr_loss, tr_regu = train(args, model, device, train_loader, optimizer, optimizer_regu, epoch, mode = args.mode, gamma_mul=gamma, startlr = args.lr)
        test(args, model, device, test_loader, epoch, tr_loss, tr_regu)
        #gamma /= 1.01
        if epoch%10 == 0:
            torch.save(model.state_dict(), args.name + '/'+ str(epoch) + '.pt')


if __name__ == '__main__':
    main()
