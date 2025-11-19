import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)  # For reproducibility
X = np.linspace(0, 2 * np.pi, 100)  # Feature data
y = np.sin(X) + np.random.normal(scale=0.1, size=X.shape)  # Target data with noise

# Visualize the training data
plt.scatter(X, y, label="Noisy Data")
plt.plot(X, np.sin(X), color="red", label="True Function")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Noisy Training Data")
plt.show()

# Define a non-linear regression hypothesis (e.g., a polynomial model)
def hypothesis(theta, X):
    return np.dot(theta, np.vstack([X**i for i in range(len(theta))]))

# Mean Squared Error
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Gradient Descent
def gradient_descent(X, y, degree, alpha, epochs):
    theta = np.random.randn(degree + 1)  # Initialize random coefficients
    m = len(y)  # Number of data points

    for epoch in range(epochs):
        y_pred = hypothesis(theta, X)
        error = y_pred - y

        # Compute gradients
        gradients = 2 * np.dot(error, np.vstack([X**i for i in range(len(theta))]).T) / m

        # Update coefficients
        theta -= alpha * gradients

        # Optional: Print loss at intervals
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {mse(y_pred, y):.4f}")

    return theta

# Training the model
degree = 5  # Degree of the polynomial
alpha = 0.01  # Learning rate
epochs = 10000  # Number of iterations

theta = gradient_descent(X, y, degree, alpha, epochs)
print("Learned coefficients:", theta)

# Plot the fitted model
y_pred = hypothesis(theta, X)
plt.scatter(X, y, label="Noisy Data")
plt.plot(X, y_pred, color="green", label="Fitted Model")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Fitted Non-linear Regression Model")
plt.show()
