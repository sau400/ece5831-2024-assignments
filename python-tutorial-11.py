#!/usr/bin/env python
# coding: utf-8

# # Modules & Packages

# In[1]:


from drawing.draw import Draw

drawing = Draw(10,20)
print(drawing.drawing())


# In[2]:


from gaming.game import Game

gaming = Game("saurabh","aniket")
gaming.played_players()


# In[3]:


from Vehicle.saurabh import Vehicle

my_vehicle = Vehicle("audi","adjustable","white",200000)
print(my_vehicle.description())


# In[ ]:




