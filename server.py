#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install --upgrade wandb')


# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
import os
from typing import Any, Dict, List
import copy
import random
import wandb
import clients
import shamir
import numpy as np


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_API_KEY"] = "183c1a6a36cbdf0405f5baacb72690845ecc8573"


# In[3]:



class Server:
    import clients
    def __init__(self,
                 model: torch.nn.Module,
                 loss: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 optimizer_conf: Dict,
                 n_client: int = 10,
                 chosen_prob: float = 0.8,
                 local_batch_size: int = 8,
                 local_epochs: int = 10) -> None:

        # global model info
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_conf = optimizer_conf
        self.n_client = n_client
        self.local_batch_size = local_batch_size
        self.local_epochs = local_epochs
        self.total_data = 0

        # create clients
        self.client_pool: List[clients.Client] = []
        self.create_client()
        self.chosen_prob = chosen_prob
        self.avg_loss = 0
        self.avg_acc = 0
        self.coordinate = []
        self.secret = 1234
        self.shamir = shamir.Shamir(self.secret)
        #self.shares = self.shamir.generate_shares(self.n_client,3,1234)
        #print(f'Shares: {", ".join(str(share) for share in self.shares)}')
        #self.pool = random.sample(self.shares, 4)
        

    def create_client(self):
        import clients
        # this function is reusable, so reset client pool is needed
        self.client_pool: List[clients.Client] = []
        self.total_data = 0
        #self.shamir = shamir.Shamir(1234)
        for i in range(self.n_client):
            model = copy.deepcopy(self.model)
            #print(str("Model :")+str(model))
            # encrypt here
            
            #str_bytes = bytes(str(model), 'utf-8')
            #int_val = int.from_bytes(str_bytes, "big")
            #self.secret = int_val
            #self.shamir = shamir.Shamir(self.secret)
            #self.shares = self.shamir.generate_shares(6, 3, self.secret)
            #print(f'Shares: {", ".join(str(share) for share in self.shares)}')
            #self.pool = random.sample(self.shares, 3)
            
            #print(f'Reconstructed secret: {self.shamir.reconstruct_secret(self.pool)}')
            
            new_client = clients.Client(client_id=i,
                                model=model,
                                loss=self.loss,
                                optimizer=self.optimizer,
                                optimizer_conf=self.optimizer_conf,
                                batch_size=self.local_batch_size,
                                epochs=self.local_epochs,
                                server=self)
            self.client_pool.append(new_client)                          

    def broadcast(self):
        model_state_dict = copy.deepcopy(self.model.state_dict())        
        for client in self.client_pool:
            client.model.load_state_dict(model_state_dict)
            #print(str("client ")+str(client.model.load_state_dict(model_state_dict)))
            
            
    def aggregate(self):
        self.avg_loss = 0
        self.avg_acc = 0
        chosen_clients = random.sample(self.client_pool,
                                       int(len(self.client_pool) * self.chosen_prob))
        #print(str("\nchosen_clients ")+str(chosen_clients))
        global_model_weights = copy.deepcopy(self.model.state_dict())
        
        #print(type(global_model_weights))
        
        x = np.array(global_model_weights)
        #print(np.shape(x))
        #str_bytes = bytes(str(global_model_weights), 'utf-8')
        str_bytes = self.shamir.encode_bytes(str(global_model_weights))
        #int_val = int.from_bytes(str_bytes, "big")
        int_val = self.shamir.bytes_to_int(str_bytes)
        
        #self.secret = int_val        
        self.secret = self.shamir.generate_secret(str(int_val))
        self.shamir = shamir.Shamir(self.secret)
        print(f'Original Secret: {self.shamir.secret}')
        self.shares = self.shamir.generate_shares(self.n_client,2,int(self.secret), int_val)
        print(f'Shares: {", ".join(str(share) for share in self.shares)}')
        self.pool = random.sample(self.shares, (self.n_client - 1))
        print(f'Reconstructed secret: {self.shamir.reconstruct_secret(self.pool)}')
        
        #byte_val = int_val.to_bytes((int_val.bit_length() + 7) // 8, 'big')
        byte_val = self.shamir.int_to_bytes(int_val)
        #decoded_str = byte_val.decode()
        decoded_str = self.shamir.decode_bytes(byte_val)
        #print(decoded_str)
        
        for key in global_model_weights:
            #print(key)
            global_model_weights[key] = torch.zeros_like(
                global_model_weights[key])
        #print(str("\nglobal_model_weights ")+str(global_model_weights))
                
        for client in chosen_clients:
            client.update_weights()
            print(f"Client {client.client_id}: Acc {client.accuracy}, Loss: {client.total_loss}")
            self.avg_loss += 1 / len(chosen_clients) * client.total_loss
            self.avg_acc += 1 / len(chosen_clients) * client.accuracy
            local_model_weights = copy.deepcopy(client.model.state_dict())
            
            #print(str("client.model.state_dict ")+str(client.model.state_dict()))
            #print(str("local_model_weights ")+str(local_model_weights))
            for key in global_model_weights:
                global_model_weights[key] += 1 / len(chosen_clients) * local_model_weights[key]
                #print(str("\nglobal_model_weights for client ")+str(global_model_weights[key]))
                
        self.model.load_state_dict(global_model_weights)
        #print(str("global_model_weights - model ")+str(self.model.load_state_dict(global_model_weights)))
        #print(f'Reconstructed secret: {self.shamir.reconstruct_secret(self.pool)}')
        

# In[ ]:





# In[ ]:




