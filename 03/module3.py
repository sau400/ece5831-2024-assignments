#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logic_gate as lg

gate = lg.LogicGate()

# test cases
tests = [[0,0],[0,1],[1,0],[1,1]]

print('Test for AND gate')
for test in tests:
    y = gate.and_gate(test[0], test[1])
    print(f'{y}=AND({test[0]}, {test[1]})')

print('Test for NAND gate')
for test in tests:
    y = gate.nand_gate(test[0], test[1])
    print(f'{y}=NAND({test[0]}, {test[1]})')
    
print("Test for OR gate")
for test in tests:
    y = gate.or_gate(test[0], test[1])
    print(f'{y}=OR({test[0]}, {test[1]})')    

print("Test for XOr gate")
for test in tests:
    y = gate.xor_gate(test[0], test[1])
    print(f'{y}=Xor({test[0]}, {test[1]})')

print("Test for Nor gate")
for test in tests:
    y = gate.nor_gate(test[0], test[1])
    print(f'{y}=nor_gate({test[0]}, {test[1]})')    

