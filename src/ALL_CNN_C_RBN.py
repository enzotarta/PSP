import torch
from torch import nn
import torch.nn.functional as F
from BasicModule import BasicModule

# python3 main2.py train  --use_trained_model=False --model='ALL_CNN_C' --checkpoint_save_name='ALL_CNN_C-1234' --lr1=0.15 --lr2=0.1 --lr3=0.05 --lr4=0.01 --weight_decay=0.0005 --use_clip=True --clip=2.0

# python3 main2.py train --max-epoch=100 --lr=0.15 --use_trained_model=False --lr_decay=1 --model='ALL_CNN_C' --checkpoint_save_name='ALL_CNN_C-1'

# python3 main2.py train --max-epoch=50 --lr=0.05 --use_trained_model=True --lr_decay=1 --model='ALL_CNN_C' --checkpoint_load_name='ALL_CNN_C-1' --checkpoint_save_name='ALL_CNN_C-2'

# python3 main2.py train --max-epoch=50 --lr=0.01 --use_trained_model=True --lr_decay=1 --model='ALL_CNN_C' --checkpoint_load_name='ALL_CNN_C-2' --checkpoint_save_name='ALL_CNN_C-3'

# python3 main2.py train --max-epoch=50 --lr=0.001 --use_trained_model=True --lr_decay=1 --model='ALL_CNN_C' --checkpoint_load_name='ALL_CNN_C-3' --checkpoint_save_name='ALL_CNN_C-4'




class ALL_CNN_C(BasicModule):
    
    def __init__(self, num_classes = 10):
        
        super(ALL_CNN_C, self).__init__()
        
        self.model_name = 'ALL_CNN_C'
        
        self.dp0 = nn.Dropout2d(p = 0.2)
        
        self.conv1 = nn.Conv2d(3, 96, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(96)
        
        self.conv2 = nn.Conv2d(96, 96, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(96)

        self.conv3 = nn.Conv2d(96, 96, 3, stride = 2, padding = 1)
        self.dp1 = nn.Dropout2d(p = 0.5)
        self.bn3 = nn.BatchNorm2d(96)
        
        self.conv4 = nn.Conv2d(96, 192, 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(192)
        
        self.conv5 = nn.Conv2d(192, 192, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(192)
        
        self.conv6 = nn.Conv2d(192, 192, 3, stride = 2, padding = 1)
        self.dp2 = nn.Dropout2d(p = 0.5)
        self.bn6 = nn.BatchNorm2d(192)
        
        
        
        self.conv7 = nn.Conv2d(192, 192, 3, padding = 0)
        self.bn7 = nn.BatchNorm2d(192)
        
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.bn8 = nn.BatchNorm2d(192)
        
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.bn9 = nn.BatchNorm2d(10)
        
        self.avg = nn.AvgPool2d(6)
        
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.conv7.weight)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)
        
        '''
        for m in self.modules():
            name = m.__class__.__name__
            if name.find is 'Conv':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        '''
        
    def forward(self, x):
        z=[]
        
        x = self.dp0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.dp1(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv5(x)
        x = self.bn5(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv6(x)
        x = self.bn6(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape
        x = self.dp2(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv8(x)
        x = self.bn8(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv9(x)
        x = self.bn9(x)
        if self.training: z.append(x)
        x = F.relu(x)
        #x = self.bn3(x)
        #print(x.shape)
        x = self.avg(x)
        x = torch.squeeze(x)
        if self.training: 
            z.append(x)        
            return x, z
        else:
            return x
