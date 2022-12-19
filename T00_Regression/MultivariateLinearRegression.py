import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# setting seed 
np.random.seed(0)

xx = pd.read_excel('LinearRegressionTrainingData_3f.xlsx')
xx = np.asarray(xx)
x = xx[:,[0,1,2]]
y = xx[:,[3]]
mx = np.amax(np.absolute(x), axis=0)
x = x/mx
x = np.hstack((np.ones_like(y), x))

xtest = pd.read_excel('LinearRegressionTestData_3f.xlsx')
xtest = np.asarray(xtest)
ytest = xtest[:,[3]]
xtest = xtest[:,[0,1,2]]
xtest = xtest/mx
xtest = np.hstack((np.ones_like(ytest), xtest))


# plotting the scatter plot of features and corresponding labels
#plt.figure(figsize=(10,5))
#plt.scatter(x, y)
#plt.xlabel('X Values (Feature)')
#plt.ylabel('Y Values (Label)')

# preparing algorithm data
#train_x = np.hstack((np.ones_like(y.T), x)) # features of training examples
train_x = x.T
train_y = y.T                              # labels of training examples
n = len(train_x[0])                      # no. of training examples
theta = np.random.rand(1,4)              # initial random weights
lr = 0.5                               # learning rate
loss = []                                # track loss over iterations

print(len(train_x))
print(len(train_x[0]))


plt.figure(figsize=(10,5))     
# Solution using Gradient Descent Algorithm for Linear Regression
iter = 0
while(iter < 2e4):
    iter += 1
    h = np.matmul(theta, train_x)               # current hypothesis
    j = np.sum((h-train_y)**2) / (n)          # cost function
    dj = np.matmul(train_x, (h-train_y).T) / n  # partial gradients of Cost function using vectorized code
    theta = theta - lr * dj.T                   # update theta
    loss.append(j)                              # loss/cost history for plotting
    
    if iter % 1000 == 0:  # plotting every 10 iterations
    	#print(j)
    	#testPredict = np.matmul(theta, train_x)
    	#sse = np.sum((testPredict-train_y)**2)/len(train_y)
    	#print(np.sqrt(sse))
    	plt.clf()       # clear figure
    	#plt.subplot(1,2,1)
    	#plt.scatter(x, y, color='red', marker='.')
    	#plt.plot(x, h[0], color='blue')
    	#plt.ylabel('x (feature)')
    	#plt.xlabel('y (label)')
    	#plt.title('Linear regression line')
    	#plt.subplot(1,2,2)
    	plt.plot(loss)
    	plt.ylabel('Loss / Cost')
    	plt.xlabel('iteration no.')
    	plt.title('Cost function vs. iterations')
    	plt.pause(0.001)
    	# check for convergence
    	if len(loss) > 2:
    		convg = abs(loss.pop() - loss.pop(-1)) / loss.pop()
    		if convg < lr*1e-3:
    			break


print(theta)
testPredict = np.matmul(theta, xtest.T)
err = testPredict-ytest.T

mse = np.sum(err**2)/len(xtest)
print("Mean Squared Error on Test Examples")
print(np.sqrt(mse))

#
# keep figures alive after execution
plt.show()
