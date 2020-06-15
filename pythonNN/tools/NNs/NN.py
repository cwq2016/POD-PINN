# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""


import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.utils.data as Data
from collections import OrderedDict
from Activations_plus import Swish

ACTIVATE     = Swish
torch.manual_seed(12)  # reproducible
np.random.seed(1234)
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_options_default ={'EPOCH':3000,\
                        'LR':0.01, \
                        'lamda': lambda epoch: 0.95**(epoch//200),\
                        'epoch_print': 1,\
                        'epoch_save':1000,
                        }

class POD_Net(nn.Module):
    def __init__(self, layers=None,OldNetfile=None):
        super(POD_Net, self).__init__()
        self.lossfun = nn.MSELoss(reduction='mean')
        NetDict = OrderedDict()
        #NetDict['start'] = nn.Linear(1,1)
        if not(layers or OldNetfile):
            raise Exception('At least one of the parameters "Layers" and "OldNerfile" should be given ')
        if OldNetfile:
            oldnet = torch.load(OldNetfile, map_location=lambda storage, loc: storage)
            layers = oldnet['layers']
        self.layers = layers
        for i in range(len(layers)-1):
            key    = "Layer%d_Linear"%i
            Value  = nn.Linear(layers[i],layers[i+1])
            NetDict[key] = Value
        
            if i != len(layers)-2:
                key    = "Layer%d_avtivate"%i
                Value  = ACTIVATE()
                NetDict[key] = Value
        self.unet = nn.Sequential(NetDict)
        
    def grad(self,a,b):
        if b.grad is not None:
            b.grad.zero_()
        da_db = ag.grad(a, b, None, 
                   create_graph=True, retain_graph=True)[0]
        return da_db
    @staticmethod
    def u_net(self,x):
        pass
    
    @staticmethod
    def forward(self,x):
        pass
        #return self.u_net(x).detach().cpu().numpy()
    @staticmethod
    def loss_NN(self, xlabel, ylabel):
            pass
    @staticmethod
    def loss_PINN(self, x, f):
        pass
    
    def loadnet(self, OldNetfile):
        #self.load_state_dict(torch.load(OldNetfile)['state_dict'],  map_location=lambda storage, loc: storage) 
        state_dict = torch.load(OldNetfile, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(state_dict) 
    def savenet(self, netfile):
        torch.save({'layers': self.layers, 'state_dict': self.state_dict()}, netfile )
        
    
def train(Net,data, netfile, options=train_options_default):
    if len(data) == 4:
        inputs, outputs, datatype, trainratio = data
    elif len(data) ==6:
        labeled_inputs, labeled_outputs, inputs, outputs,datatype,trainratio= data
    else:
        raise Exception('Expect inout <data> with 4 or 6 elements, but got %d'%len(data))

    weight = np.ones((inputs.shape[0], 1))
    if len(data) ==6:
        lb = Net.roeqs.design_space[0:1,:]; ub=Net.roeqs.design_space[1:2,:]
        dis = np.zeros((inputs.shape[0], 1))
        for i in range(inputs.shape[0]):
            diff = ( inputs[i:i+1,:]-labeled_inputs )/(ub-lb)
            dis[i,0] = np.linalg.norm(diff, axis=1).min()
        weight = dis/dis.max()
        
    dataset   = Data.TensorDataset(torch.tensor(inputs).float().to(DEVICE),\
                                   torch.tensor(outputs).float().to(DEVICE),\
                                   torch.tensor(weight).float().to(DEVICE),\
                                   )
    trainsize =  int(inputs.shape[0] * trainratio)
    testsize  =  inputs.shape[0] - trainsize
    trainset, testset = Data.random_split(dataset, [trainsize, testsize])
    trainloader = Data.DataLoader(trainset, batch_size= trainsize//options['NBATCH'], shuffle = True)
    testloader  = Data.DataLoader(testset , batch_size= testsize                    , shuffle = True) 
    
    
    def lossHybrid(xresi,yresi,weight):
        loss1 = Net.loss_NN(labeled_inputs,labeled_outputs)
        loss2 = Net.loss_PINN(xresi,yresi,weight)
        loss  = loss1 + loss2
        return loss
    
    if datatype == 'Label':
        Netloss = lambda x,y,weight: Net.loss_NN(x,y)
    elif datatype == 'Resi':
        Netloss = Net.loss_PINN
    elif datatype == 'Hybrid':
        labeled_inputs  = torch.tensor(labeled_inputs).float().to(DEVICE)
        labeled_outputs = torch.tensor(labeled_outputs).float().to(DEVICE)
        Netloss = lossHybrid
    else:
        raise Exception('wrong datatpe in data:' + datatype)
        
    def closure():
        optimizer.zero_grad()
        loss = Netloss(batch_in, batch_out, weight)
        loss.backward()
        return loss  
    
    loss_history_train = np.zeros((options['EPOCH'], 1))
    NBatch = len(trainloader)
    loss_history_test  = np.zeros((options['EPOCH'], 1))
    
    optimizer = torch.optim.Adam(Net.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])
    #optimizer = torch.optim.LBFGS(Net.parameters(), lr=options['LR'])
    #lamda1 = lambda epoch: 0.95**(epoch//50)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=options['lamda'])

    for epoch in range(options['EPOCH']):
        loss_train = 0; loss_test =0
        for nbatch, (batch_in, batch_out, weight) in enumerate(trainloader):      
            running_loss=optimizer.step(closure)
            loss_train += running_loss 
            with torch.no_grad():
                for nbatch2, (x_test, y_test, weight) in enumerate(testloader):  
                    loss_test   += Netloss(x_test, y_test, weight) 
            if epoch%options['epoch_print'] == 0:
                print("|epoch=%5d,nbatch=%5d | loss=(%11.7e,  %11.7e)"%(epoch,nbatch,running_loss,loss_test)) 
        loss_history_train[epoch,0] = loss_train/NBatch
        loss_history_test[ epoch,0] = loss_test/NBatch  
        
        if epoch>=1000 and datatype == 'Label':
            if np.all( loss_history_test[epoch-5:epoch+1,0]>loss_history_test[epoch-6:epoch,0]):
                Net.savenet(netfile)
                return loss_history_train, loss_history_test
        scheduler.step()
        if epoch % options['epoch_save'] == 0 or epoch == options['EPOCH']-1:
            Net.savenet(netfile)
    return loss_history_train, loss_history_test
    
     



