# Author: Ahmed Alkhayal - EE482 Fall 2022 (https://github.com/Alkhayal7)

import numpy as np
import matplotlib.pyplot as plt

# setting seed 
np.random.seed(0)

# Generate artificial training examples
x = np.arange(-10, 10, 0.1)     # one feature
y = -20 + 5.5 * x               # labels

# adding noise to labels
y = y + np.random.rand(y.shape[0]) * 10

# plotting the scatter plot of features and corresponding labels
plt.figure(figsize=(10,5))
plt.scatter(x, y)
plt.xlabel('X Values (Feature)')
plt.ylabel('Y Values (Label)')

# preparing algorithm data
train_x = np.stack((np.ones_like(x), x)) # features of training examples
train_y = y                              # labels of training examples
n = len(train_x[0])                      # no. of training examples
theta = np.random.rand(1,2)              # initial random weights
lr = 0.001                               # learning rate
loss = []                                # track loss over iterations


plt.figure(figsize=(10,5))     
# Solution using Gradient Descent Algorithm for Linear Regression
iter = 0
while(iter < 1e4):
    iter += 1
    h = np.matmul(theta, train_x)               # current hypothesis
    j = np.sum((h-train_y)**2) / (2*n)          # cost function
    dj = np.matmul(train_x, (h-train_y).T) / n  # partial gradients of Cost function using vectorized code
    theta = theta - lr * dj.T                   # update theta
    loss.append(j)                              # loss/cost history for plotting
    
    if iter % 10 == 0:  # plotting every 10 iterations
        plt.clf()       # clear figure
        
        # plotting Linear regression line
        plt.subplot(1,2,1)
        plt.scatter(x, y, color='red', marker='.')
        plt.plot(x, h[0], color='blue')
        plt.ylabel('x (feature)')
        plt.xlabel('y (label)')
        plt.title('Linear regression line')

        # plotting cost function vs iterations
        plt.subplot(1,2,2)
        plt.plot(loss)
        plt.ylabel('Loss / Cost')
        plt.xlabel('iteration no.')
        plt.title('Cost function vs. iterations')
        plt.pause(0.001)
        
        # check for convergence
        if len(loss) > 2:
            convg = abs(loss.pop() - loss.pop(-1)) / loss.pop()
            if convg < lr*1e-3: break

print(theta)
# keep figures alive after execution
plt.show()
