import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToTensor

from sklearn.cluster import KMeans

from neural_networks.TOOLS.TOOLS.functions import tyler_pdf
from matplotlib.ticker import MaxNLocator
import torch.nn.functional as F

from scipy.signal import savgol_filter
from time import time

# #################################################################
# ##### training and displaying weights, activations and losses ###
# #################################################################

# for n in range(1):
#     samplesize=2
#     num_samples=1000
#     mean1=0
#     mean2=2
#     var=1

#     X=[np.random.normal(mean1,var,samplesize).tolist() for i in range(num_samples)]
#     X+=[np.random.normal(mean2,var,samplesize).tolist() for i in range(num_samples)]
#     y=[[0] for i in range(num_samples)]
#     y+=[[1] for i in range(num_samples)]

#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

#     dataset = torch.utils.data.TensorDataset(X, y)
#     batch_size = 10
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     #######################################################################

#     #### MODEL DEFINITION #################
#     class NN(nn.Module):
#         def __init__(self):
#             super(NN, self).__init__()
#             self.fc1=nn.Linear(samplesize,samplesize)
#             self.relu1=nn.LeakyReLU(0.01)
#             self.fc2=nn.Linear(2,2)
#             # self.softmax=nn.Softmax(dim=1) #this line is not neccesary when you use crossentropyloss
        
#         def forward(self,input):
#             output=self.fc1(input)
#             output=self.relu1(output)
#             output=self.fc2(output)
#             # output=self.softmax(output) #this line is not neccesary when you use crossentropyloss
            
#             return (output)
#     #######################################
#     runs=300
#     colors=plt.cm.jet(np.linspace(0.1,0.9,runs))
#     for run in range(runs):
#         ### INSTANTIATE MODEL #################
#         model=NN()
#         #######################################

#         ### HOOKS ##############################################
#         hidden_activations1 = []
#         hidden_activations2 =[]
#         hidden_activations3=[]
#         def hook_fn1(module, input, output):
#             hidden_activations1.append(output.detach().clone())
#         # hook1 = model.fc1.register_forward_hook(hook_fn)
#         hook1=model.relu1.register_forward_hook(hook_fn1)

#         ########################################################
        
#         ##### TRAINING #################################################################
#         loss_fn = nn.CrossEntropyLoss()# binary cross entropy
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)
#         model.train()
#         loss_tracker=[]
#         weight_pdf_tracker=[]

#         output_tracker=[]

#         for epoch in range(500):
#             for X_batch, y_batch in loader:
#                 optimizer.zero_grad()
#                 y_pred = model(X_batch)
#                 loss = loss_fn(y_pred, y_batch.long().squeeze())
#                 loss.backward()
#                 optimizer.step()    
#                 # print(f"Epoch {epoch+1}")
#             loss_tracker.append(loss.detach())
#             weight_pdf_tracker.append(model.fc1.weight.clone().detach().flatten().numpy())
#             output_tracker.append(hidden_activations1[-1][-1].detach().flatten().tolist())
#             # print("Bias (fc1):", model.fc1.bias.data[0])
#         ################################################################################

#         model.eval()
#         testdat= torch.tensor([np.random.normal(0,1,samplesize)], dtype=torch.float32)

#         predict=model(testdat)
#         probs = torch.softmax(predict, dim=1)
#         predicted_class = torch.argmax(probs, dim=1)
#         print("Predicted class:", predicted_class.item())
#         # plt.subplot(4,1,1)
#         # plt.plot(loss_tracker,'-o')
#         # ax = plt.gca()
#         # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         # colors=plt.cm.jet(np.linspace(0.1,0.9,len(weight_pdf_tracker)))
#         # for i,s,col in zip(weight_pdf_tracker,output_tracker,colors):
#         #     # plt.subplot(4,1,2)
#         #     plt.plot(i[0],i[1],'o',color=col)

#         #     plt.subplot(4,1,3)
#         #     plt.plot(i[2],i[3],'o',color=col)

#         #     plt.subplot(4,1,4)
#         #     plt.plot(s,'-o',color=col)
#     # plt.subplot(4,1,4)
#     # ax = plt.gca()
#     # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         print(weight_pdf_tracker[-1][0])
#         plt.plot(weight_pdf_tracker[-1][0],weight_pdf_tracker[-1][1],'o',color=colors[run])
# # plt.xlabel('hidden layer neuron id')
# # plt.ylabel('hidden layer output')

#     # plt.subplot(4,1,1)
#     # plt.ylabel('Loss')
#     # plt.xlabel('epoch')

# plt.show()

# # #### PREDICT #########################
# # Xbatch = X[50:50+5]
# # ybatch=y[50:50+5].long().squeeze()
# # y_pred = model(Xbatch)
# # y_pred=torch.argmax(y_pred, dim=1)
# # print(y_pred)
# # print(ybatch)
# # ######################################

############################################
# ##### step by step hand calc validation ####
# ############################################

# #input layer dimension
# d=2

# #h hidden layer dimension
# h=2

# #output layer dimension
# D=2

# num_samples=1

# mean1=0
# mean2=2
# var=1

# X=[np.random.normal(mean1,var,d).tolist() for i in range(num_samples)]
# X+=[np.random.normal(mean2,var,d).tolist() for i in range(num_samples)]

# y=[[0] for i in range(num_samples)]
# y+=[[1] for i in range(num_samples)]

# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# dataset = torch.utils.data.TensorDataset(X, y)
# batch_size = 10

# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# def leaky_relu(x, alpha=0.01):
#     return x if x >= 0 else alpha * x
# #### MODEL DEFINITION #################
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.fc1=nn.Linear(d,h,bias=False)
#         self.relu1=nn.LeakyReLU(0.01)
#         # self.fc2=nn.Linear(h,D,bias=False)
#         # self.softmax=nn.Softmax(dim=1) #this line is not neccesary when you use crossentropyloss
    
#     def forward(self,input):
#         output=self.fc1(input)
#         output=self.relu1(output)
#         # output=self.fc2(output)
#         # output=self.softmax(output) #this line is not neccesary when you use crossentropyloss
#         return (output)
    
# ### INSTANTIATE MODEL #################
# model=NN()
# #######################################

# # ### HOOKS ##############################################
# hidden_activations1 = []
# def hook_fn1(module, input, output):
#     hidden_activations1.append(output.detach().clone())
# hook1 = model.fc1.register_forward_hook(hook_fn1)

# # hidden_activations0 = []
# # def hook_fn0(module, input, output):
# #     hidden_activations0.append(output.detach().clone())
# # hook0 = model.relu1.register_forward_hook(hook_fn0)

# # hidden_activations2 = []
# # def hook_fn2(module, input, output):
# #     hidden_activations2.append(output.detach().clone())
# # hook2 = model.fc2.register_forward_hook(hook_fn2)

# ### TRAINING
# loss_fn = nn.CrossEntropyLoss()# binary cross entropy
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# model.train()

# loss_tracker=[]
# weight_pdf_tracker=[]
# output_tracker=[]
# weight_pdf_tracker2=[]
# relu_tracker=[]
# for epoch in range(1):
#     for X_batch, y_batch in loader:
#         # print(X_batch.numpy(),X_batch.numpy()[0])
#         w=model.fc1.weight.clone().detach().numpy()
#         optimizer.zero_grad()
#         y_pred = model(X_batch)
#         loss = loss_fn(y_pred, y_batch.long().squeeze())
#         loss.backward()
#         optimizer.step()    
#         # print(f"Epoch {epoch+1}")
#     # loss_tracker.append(loss.detach())
#     weight_pdf_tracker.append(model.fc1.weight.clone().detach().flatten().numpy())
#     # output_tracker.append(hidden_activations0[-1][-1].detach().flatten().tolist())
#     # weight_pdf_tracker2.append(model.fc2.weight.clone().detach().flatten().numpy())




# ############################################
# ##### compute 2 point function ####
# ############################################

# #input layer dimension
# d=2

# #h hidden layer dimension
# h=2

# #output layer dimension
# D=2

# aa=[]
# bb=[]
# ab=[]
# ba=[]
# netsss=[2**i for i in range(17)]
# for num_networks in netsss:
#     G2xy=np.zeros(shape=(2,2))
#     for net in range(num_networks):
#         num_samples=2
#         mean1=0
#         mean2=2
#         var=1

#         X=[np.random.normal(mean1,var,d).tolist() for i in range(num_samples)]
#         X=[[1.0,0.4],[0.2,0.8]]
#         y=[[0] for i in range(num_samples)]

#         X = torch.tensor(X, dtype=torch.float32)
#         y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

#         dataset = torch.utils.data.TensorDataset(X, y)
#         batch_size = num_samples

#         loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         def leaky_relu(x, alpha=0.01):
#             return x if x >= 0 else alpha * x
#         #### MODEL DEFINITION #################
#         class NN(nn.Module):
#             def __init__(self):
#                 super(NN, self).__init__()
#                 self.fc1=nn.Linear(d,h,bias=False)
#                 # self.relu1=nn.LeakyReLU(0.01)
#                 # self.fc2=nn.Linear(h,D,bias=False)
#                 # self.softmax=nn.Softmax(dim=1) #this line is not neccesary when you use crossentropyloss
            
#             def forward(self,input):
#                 output=self.fc1(input)
#                 # output=self.relu1(output)
#                 # output=self.fc2(output)
#                 # output=self.softmax(output) #this line is not neccesary when you use crossentropyloss
#                 return (output)
            
#         ### INSTANTIATE MODEL #################
#         model=NN()
#         torch.nn.init.normal_(model.fc1.weight, mean=0.0, std=1.0 / np.sqrt(d))

#         #######################################

#         # ### HOOKS ##############################################
#         hidden_activations1 = []
#         def hook_fn1(module, input, output):
#             hidden_activations1.append(output.detach().clone())
#         hook1 = model.fc1.register_forward_hook(hook_fn1)

#         # hidden_activations0 = []
#         # def hook_fn0(module, input, output):
#         #     hidden_activations0.append(output.detach().clone())
#         # hook0 = model.relu1.register_forward_hook(hook_fn0)

#         # hidden_activations2 = []
#         # def hook_fn2(module, input, output):
#         #     hidden_activations2.append(output.detach().clone())
#         # hook2 = model.fc2.register_forward_hook(hook_fn2)

#         ### TRAINING
#         loss_fn = nn.CrossEntropyLoss()# binary cross entropy
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)

#         model.train()

#         loss_tracker=[]
#         weight_pdf_tracker=[]
#         output_tracker=[]
#         weight_pdf_tracker2=[]
#         relu_tracker=[]
#         for epoch in range(1):
#             correlations=np.zeros(shape=(2,2))
#             for X_batch, y_batch in loader:
#                 optimizer.zero_grad()
#                 y_pred = model(X_batch)
#                 correlations+=np.outer(y_pred.detach().numpy()[0],y_pred.detach().numpy()[1])
#                 loss = loss_fn(y_pred, y_batch.long().squeeze())
#                 loss.backward()
#                 optimizer.step()    
#                 # print(f"Epoch {epoch+1}")
#             # loss_tracker.append(loss.detach())
#             weight_pdf_tracker.append(model.fc1.weight.clone().detach().flatten().numpy())
#             # output_tracker.append(hidden_activations0[-1][-1].detach().flatten().tolist())
#             # weight_pdf_tracker2.append(model.fc2.weight.clone().detach().flatten().numpy())
#         G2xy+=correlations/num_samples
#     aa.append(G2xy[0][0]/num_networks)
#     bb.append(G2xy[1][1]/num_networks)
#     ab.append(G2xy[0][1]/num_networks)
#     ba.append(G2xy[1][0]/num_networks)

# plt.semilogx(netsss,aa)
# plt.semilogx(netsss,bb)
# plt.semilogx(netsss,ba)
# plt.semilogx(netsss,ab)

# plt.show()


# ############################################
# ##### compute 2 point function ####
# ############################################

# #input layer dimension
# d=2

# #h hidden layer dimension
# h=2

# #output layer dimension
# D=2

# aa=[]
# bb=[]
# ab=[]
# ba=[]
# netsss=[2**i for i in range(10)]
# for num_networks in netsss:
#     G2xy=np.zeros(shape=(2,2))
#     for net in range(num_networks):
#         num_samples=2
#         mean1=0
#         mean2=2
#         var=1

#         X=[np.random.normal(mean1,var,d).tolist() for i in range(num_samples)]
#         X=[[1.0,0.0],[0.0,1.0]]
#         y=[[0] for i in range(num_samples)]

#         X = torch.tensor(X, dtype=torch.float32)
#         y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

#         dataset = torch.utils.data.TensorDataset(X, y)
#         batch_size = num_samples

#         loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         def leaky_relu(x, alpha=0.01):
#             return x if x >= 0 else alpha * x
#         #### MODEL DEFINITION #################
#         class NN(nn.Module):
#             def __init__(self):
#                 super(NN, self).__init__()
#                 self.fc1=nn.Linear(d,h,bias=False)
#                 # self.relu1=nn.LeakyReLU(0.01)
#                 # self.fc2=nn.Linear(h,D,bias=False)
#                 # self.softmax=nn.Softmax(dim=1) #this line is not neccesary when you use crossentropyloss
            
#             def forward(self,input):
#                 output=self.fc1(input)
#                 # output=self.relu1(output)
#                 # output=self.fc2(output)
#                 # output=self.softmax(output) #this line is not neccesary when you use crossentropyloss
#                 return (output)
            
#         ### INSTANTIATE MODEL #################
#         model=NN()
#         torch.nn.init.normal_(model.fc1.weight, mean=0.0, std=1.0 / np.sqrt(d))

#         #######################################

#         # ### HOOKS ##############################################
#         hidden_activations1 = []
#         def hook_fn1(module, input, output):
#             hidden_activations1.append(output.detach().clone())
#         hook1 = model.fc1.register_forward_hook(hook_fn1)

#         # hidden_activations0 = []
#         # def hook_fn0(module, input, output):
#         #     hidden_activations0.append(output.detach().clone())
#         # hook0 = model.relu1.register_forward_hook(hook_fn0)

#         # hidden_activations2 = []
#         # def hook_fn2(module, input, output):
#         #     hidden_activations2.append(output.detach().clone())
#         # hook2 = model.fc2.register_forward_hook(hook_fn2)

#         ### TRAINING
#         loss_fn = nn.CrossEntropyLoss()# binary cross entropy
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)

#         model.train()

#         loss_tracker=[]
#         weight_pdf_tracker=[]
#         output_tracker=[]
#         weight_pdf_tracker2=[]
#         relu_tracker=[]
#         for epoch in range(1):
#             correlations=np.zeros(shape=(2,2))
#             for X_batch, y_batch in loader:
#                 optimizer.zero_grad()
#                 y_pred = model(X_batch)
#                 correlations+=np.outer(y_pred.detach().numpy()[0],y_pred.detach().numpy()[1])
#                 loss = loss_fn(y_pred, y_batch.long().squeeze())
#                 loss.backward()
#                 optimizer.step()    
#                 # print(f"Epoch {epoch+1}")
#             # loss_tracker.append(loss.detach())
#             weight_pdf_tracker.append(model.fc1.weight.clone().detach().flatten().numpy())
#             # output_tracker.append(hidden_activations0[-1][-1].detach().flatten().tolist())
#             # weight_pdf_tracker2.append(model.fc2.weight.clone().detach().flatten().numpy())
#         G2xy+=correlations/num_samples
#     aa.append(G2xy[0][0]/num_networks)
#     bb.append(G2xy[1][1]/num_networks)
#     ab.append(G2xy[0][1]/num_networks)
#     ba.append(G2xy[1][0]/num_networks)

# plt.semilogx(netsss,aa)
# plt.semilogx(netsss,bb)
# plt.semilogx(netsss,ba)
# plt.semilogx(netsss,ab)

# plt.show()

### avalanches? ############################################################################
start=time()
d=3000
D=2
h=max(2,d-1)
num_samples=5000
mean1=0
mean2=2
var=1

device = torch.device("mps")
# torch.set_num_threads(10)

X=[np.random.normal(mean1,var,d).tolist() for i in range(num_samples)]
y=[[0] for i in range(num_samples)]

X+=[np.random.normal(mean2,var,d).tolist() for i in range(num_samples)]
y+=[[1] for i in range(num_samples)]

X = torch.tensor(X, dtype=torch.float32).to(device, non_blocking=True)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device,non_blocking=True)

dataset = torch.utils.data.TensorDataset(X, y)
batch_size = int(num_samples/10)

loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     )
def leaky_relu(x, alpha=0.01):
    return x if x >= 0 else alpha * x
#### MODEL DEFINITION #################
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1=nn.Linear(d,h,bias=False)
        self.relu1=nn.LeakyReLU(0.01)
        self.fc2=nn.Linear(h,D,bias=False)
        # self.softmax=nn.Softmax(dim=1) #this line is not neccesary when you use crossentropyloss
    
    def forward(self,input):
        output=self.fc1(input)
        output=self.relu1(output)
        output=self.fc2(output)
        # output=self.softmax(output) #this line is not neccesary when you use crossentropyloss
        return (output)
############################################################################################
model=NN().to(device)
### TRAINING
loss_fn = nn.CrossEntropyLoss()# binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

loss_tracker=[]
weight_pdf_tracker=[]

epochs=np.arange(1,1000,1)
for epoch in epochs:
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.long().squeeze())
        loss.backward()
        optimizer.step()    
        # print(f"Epoch {epoch+1}")
    loss_tracker.append(loss.detach())
    weight_pdf_tracker.append(model.fc1.weight.clone().detach().flatten().cpu().numpy())
print(time()-start)

### test

# print("CPU threads available:", torch.get_num_threads())
# print("MPS available:", torch.backends.mps.is_available())

# def create_torch_tensors(device):
#     X=torch.rand((40000,40000),dtype=torch.float32)
#     Y=torch.rand((40000,40000),dtype=torch.float32)
#     X=X.to(device)
#     Y=Y.to(device)

#     return X, Y

# device=torch.device('cpu')
# x,y=create_torch_tensors(device)

# start=time()
# x*y
# print(time()-start)

# device=torch.device('mps')
# x,y=create_torch_tensors(device)

# start=time()
# x*y
# print(time()-start)