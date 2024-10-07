#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from multilayer_percetron import MultilayerPerceptron


# In[2]:


neural_networks = MultilayerPerceptron()
neural_networks.init_network()


# # TEST CASES FOR MULTILAYER PERCEPTRON CLASS

# In[3]:


## INPUT 1
input_1 = [10,20]
y1 = neural_networks.forward(np.array(input_1))


# In[5]:


y1


# In[6]:


## INPUT 2 
input_2 = [30,40]
y2 = neural_networks.forward(np.array(input_2))
y2


# In[7]:


## INPUT 3 
input_3 = [50,60]
y3 = neural_networks.forward(np.array(input_3))
y3


# In[8]:


## INPUT 4
input_4 = [10,6]
y4 = neural_networks.forward(np.array(input_4))
y4

