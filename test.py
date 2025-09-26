import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
import numpy as np
from micrograd.neural_network import MLP
from micrograd.engine import Value,draw_dot

a=Value(1.52,label='a')
b=Value(0.25,label='b')
c=Value(1,label='c')
d=Value(-.12,label='d')

e = a*b
e.label='e'
# print(e)
l=e+c
l.label='l'
r=(2*l).exp()
r.label='r'
o = (r -1 ) / (r+1)
o.label='o'
# print(o)

# print(o._prev)
# o.backward()

x=[2.,3,-1]
n =MLP(3,[4,4,1])
n(x)
# dot=draw_dot(n(x))
# dot.render('graph', view=True) 

xs=[
    [2.0,3.0,-1],
    [3.0,-1.0,-0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]
ys=[1.0,-1.0,-1.0,1.0]


for i in range(40):
    
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout-ygt)**2 for ygt,yout in zip(ys,ypred))
    
    for p in n.parameters():
        p.grad=0
    # backward pass
    loss.backward()
    
    for p in n.parameters():
        p.data += - 0.05*p.grad
        
    print(i,loss.data)
loss.label='loss'
print(len(n.parameters()))
print(ypred)
# dot=draw_dot(loss)
# dot.render('graph', view=True) 


