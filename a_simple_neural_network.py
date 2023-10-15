'''
Implementing a simple neural network with the
input layer consisting of three nodes, one hidden
layer consisting of five nodes, and the output layer
consisting of two nodes, where the outputs of the hidden
layer are activated with sigmoid activation.
'''

from google.colab import drive
drive.mount('/content/drive')

from IPython.display import clear_output
!pip3 install pyprind
clear_output()

# Imports

import torch

# Initializing the Parameters and the Variables
# y = a*x + b

x = torch.rand((5, 3), requires_grad=False)
y = torch.rand((5, 1), requires_grad=False)

a = torch.rand((1, 3), requires_grad=True)
b = torch.rand((1, 1), requires_grad=True)


# Forward Pass 1
y_pred1 = x@a.T + b

# Computing Loss
loss = torch.mean((y-y_pred1)**2)
print(loss.item())

# Back Propogation
loss.backward()

# Updating Gradients
with torch.no_grad():
    a = a - 0.01*a.grad
    b = b - 0.01*b.grad


# Forward Pass 2
y_pred2 = x@a.T + b

# Computing Loss
loss = torch.mean((y-y_pred2)**2)
print(loss.item())
