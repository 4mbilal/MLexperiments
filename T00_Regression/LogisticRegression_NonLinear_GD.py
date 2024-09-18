import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate and display the image with two sets of points
np.random.seed(0)
n_points = 100
x1 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], n_points)
x2 = np.random.multivariate_normal([7, 7], [[1, 0], [0, 1]], 33)
x3 = np.random.multivariate_normal([3, 7], [[1, 0], [0, 1]], 33)
x4 = np.random.multivariate_normal([7, 3], [[1, 0], [0, 1]], 34)
x2 = np.vstack((x2,x3))
x2 = np.vstack((x2,x4))

# Create a black background image
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('black')

# Plot the points
ax.scatter(x1[:, 0], x1[:, 1], c='green', label='Class 1')
ax.scatter(x2[:, 0], x2[:, 1], c='red', label='Class 2')
ax.legend()
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show(block = False)

# Step 2: Train a logistic regression classifier using gradient descent
X = np.vstack((x1, x2))
y = np.hstack((np.zeros(n_points), np.ones(n_points)))

# Add intercept term to X
X = np.hstack((np.ones((X.shape[0], 1)), X))
X2 = np.power(X,2)
X = np.hstack((X,X2))
max_val = np.max(np.abs(X), axis=0)
X = X/max_val

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    return (-1/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1/m) * X.T @ (sigmoid(X @ theta) - y)
        theta -= learning_rate * gradient
        cost_history.append(cost_function(theta, X, y))
    return theta, cost_history

# Initialize parameters
theta = np.zeros(X.shape[1])
learning_rate = 10
iterations = 2000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Step 3: Visualize the classification results
xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
grid = np.hstack((np.ones((grid.shape[0], 1)), grid))
grid2 = np.power(grid,2)
grid = np.hstack((grid,grid2))
grid = grid/max_val
probs = sigmoid(grid @ theta).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('black')
ax.contourf(xx, yy, probs, levels=[0, 0.5, 1], colors=['green', 'red'], alpha=0.3)
ax.scatter(x1[:, 0], x1[:, 1], c='green', label='Class 1')
ax.scatter(x2[:, 0], x2[:, 1], c='red', label='Class 2')
ax.legend()
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show(block = False)

# Step 4: Plot the loss function
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history, 'b')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Loss Function')
plt.show()
