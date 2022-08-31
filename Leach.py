#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade wandb')


# In[2]:


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
from random import seed
from random import random
import math
import matplotlib.pyplot as plt


# In[3]:


class Leach:
    def __init__(self,coordinates):
       
        self.coordinates = coordinates
        self.P = 0.45
        self.r = 1
        self.Tn = (self.P)/((1-(self.P))*(self.r%(1/self.P)))
        self.seed = 1
        self.node_energy = []
        self.array_header = []
        self.clusters = []
        self.header_coord = []
        
    def assign_energy(self,coordinates):
        seed(self.seed)
        self.seed = self.seed + 1
        node_energy = []
        for _ in range(len(coordinates)):
            value = random()
            node_energy.append(value)
            self.node_energy = node_energy
            
        return self.node_energy
        
    def define_header(self,node_energy, coordinates):
        for i in range(len(coordinates)):
            if node_energy[i] > self.Tn:
                self.array_header.append(coordinates[i])
        
        return self.array_header

    def speedRate(self,coordinate):
        new_coord = []
        for i in range(len(coordinate)):
            if ((coordinate[i][0]-2) > 99) or ((coordinate[i][1]-2) > 99):
                new_coord.append((coordinate[i]/3))
            elif ((coordinate[i][0]-2) < 1) or ((coordinate[i][1]-2) < 1):
                new_coord.append((coordinate[i]*3))
            else:
                new_coord.append(coordinate[i]-2)        
        return new_coord
    
    def calculate_distance(self,x2,x1,y2,y1):
        x = (x2-x1)
        y = (y2-y1)
        x2 = x * x
        y2 = y * y
        dist = math.sqrt(x2 + y2)
        
        return dist
    
    def fillingNoneValue(self,cluster_coords,best_k,n_client):
        for i in range(int(best_k)):
            for j in range(int(n_client - len(cluster_coords[i]))):
                cluster_coords[i].append(None)
        return cluster_coords

    def initiaiseClusterCoordinates(self,n_leader,cluster):
        cluster_coords = [[] for _ in range(n_leader)]
        for i in range(int(n_leader)):    
            for j in range(len(cluster)):
                if (cluster[j] == i):
                    cluster_coords[i].append(j)
        return cluster_coords
    
    def assign_cluster(self,newdistance):
        self.clusters = [[] for _ in range(len(newdistance))]
        cluster = []
        for i in range(len(newdistance[0])):
            for j in range(len(newdistance)):
                if newdistance[0][i] < newdistance[j][i]:
                    self.clusters[j].append(0)
                else:
                    self.clusters[j].append(j)
        for i in range(len(newdistance[0])):
            cluster.append(0)  
        for i in range(len(newdistance)):
            for j in range(len(newdistance[0])):
                if ((i > 0) and (self.clusters[i][j] != 0)):
                    cluster[j] = self.clusters[i][j]
        #size = len(self.array_header) - 1
        return cluster
    
    def plot_cluster(self,clusters,coordinates):
        get_ipython().run_line_magic('matplotlib', '')
        for i in range(len(coordinates)):
            if clusters[i] == 0:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "red")
            elif clusters[i] == 1:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "blue")
            elif clusters[i] == 2:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "green")
            elif clusters[i] == 3:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "purple")
            elif clusters[i] == 4:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "cyan")
            elif clusters[i] == 5:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "orange")
            elif clusters[i] == 6:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "gray")
            elif clusters[i] == 7:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "yellow")
            elif clusters[i] == 8:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "magenta")
            elif clusters[i] == 9:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "pink")
            elif clusters[i] == 100:
                plt.scatter(coordinates[i:,0],coordinates[i:,1], c = "black")
            plt.annotate(clusters[i],(self.coordinates[i][0],self.coordinates[i][1]),xytext=(0,10),textcoords="offset points")
            plt.show()
            plt.pause(0.1)
            #plt.cla()


# In[ ]:





# In[ ]:




