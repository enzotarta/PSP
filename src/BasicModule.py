import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        
        self.epoch = 0
        self.lr_pre_epoch = []
        self.loss_pre_epoch = []
        self.valacc_pre_epoch = []
        self.testacc_pre_epoch = []

        
    def load(self, path):
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.lr_pre_epoch = checkpoint['lr']
        self.loss_pre_epoch = checkpoint['loss']
        self.valacc_pre_epoch = checkpoint['valacc']
        self.testacc_pre_epoch = checkpoint['testacc']
        self.load_state_dict(checkpoint['state_dict'])

        
    def save(self, name = None):
        if name is None:
            name = 'checkpoints/' + self.model_name
        else:    
            name = 'checkpoints/' + name
            
        
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr_pre_epoch,
            'loss': self.loss_pre_epoch,
            'valacc': self.valacc_pre_epoch,
            'testacc': self.testacc_pre_epoch,
            'state_dict': self.state_dict(),
        }, name)
        
        return name
    
    def update_epoch(self, lr_, loss_, valacc_, testacc_):
        self.epoch += 1
        self.lr_pre_epoch.append(lr_)
        self.loss_pre_epoch.append(loss_)
        self.valacc_pre_epoch.append(valacc_)
        self.testacc_pre_epoch.append(testacc_)
        assert len(self.lr_pre_epoch) == len(self.loss_pre_epoch) == len(self.valacc_pre_epoch) == len(self.testacc_pre_epoch)
       
    def epoches(self):
        return len(self.lr_pre_epoch)
    