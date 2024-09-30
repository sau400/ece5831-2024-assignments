#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np

class LogicGate():
    def __init__(self):
        pass

    def and_gate(self, x1, x2):
        b = -0.7
        w = np.array([0.5, 0.5, 1])
        x = np.array([x1, x2, b])
        
        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0
        
    def nand_gate(self, x1, x2):
        b = 0.7
        w = np.array([-0.5, -0.5, 1])
        x = np.array([x1, x2, b])
        
        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0
        
    def or_gate(self, x1, x2):
        b = -0.9
        w = np.array([1, 1, 1])
        x = np.array([x1, x2, b])
        
        y = np.sum(x*w)

        if y > 0:
            return 1
        else:
            return 0
        
    def nor_gate(self, x1, x2):
        b = -0.9
        w = np.array([1, 1, 1])
        x = np.array([x1, x2, b])
        
        y = np.sum(x*w)

        if y > 0:
            return 0
        else:
            return 1
        
    def xor_gate(self, x1, x2):
        y1 = self.or_gate(x1, x2)
        y2 = self.nand_gate(x1, x2)
        return self.and_gate(y1, y2)


# In[ ]:




