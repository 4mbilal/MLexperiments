import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate test points
np.random.seed(42)
X = np.linspace(-1, 1, 20)
y = 0.1 * X**4 - 0.5 * X**3 + 2 * X**2 - X + 5 + np.random.normal(0, 0.3, size=X.shape)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_rand = X[indices]
y_rand = y[indices]


# Split the data into training and validation sets
train_size = int(0.25 * len(X))
X_train, X_val = X_rand[:train_size], X_rand[train_size:]
y_train, y_val = y_rand[:train_size], y_rand[train_size:]

# Plot the points
plt.figure(1)
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.scatter(X_val, y_val, color='green', label='Validation Data')
plt.legend()
plt.show(block = False)

# Prepare the data for TensorFlow
X_train = X_train.reshape(-1, 1)
X_val = X_val.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# Create polynomial features manually
N = 6

X_train_poly = np.hstack([X_train**i for i in range(N)])
X_val_poly = np.hstack([X_val**i for i in range(N)])

# Define the polynomial regression model
model = tf.keras.Sequential([
    # tf.keras.layers.Dense(units=1, input_shape=[X_train_poly.shape[1]],kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))
    tf.keras.layers.Dense(units=1, input_shape=[X_train_poly.shape[1]],kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00))
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2.5)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.4)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model and store the training history
history = model.fit(X_train_poly, y_train, epochs=500,  batch_size=5, validation_data=(X_val_poly, y_val), verbose=1)

# Plot the training and validation loss curves
plt.figure(2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show(block = False)

# Predict using the trained model
X = np.linspace(-1, 1, 20)
y = 0.1 * X**4 - 0.5 * X**3 + 2 * X**2 - X + 5 + np.random.normal(0, 0.2, size=X.shape)
y_pred = model.predict(np.hstack([X.reshape(-1, 1)**i for i in range(N)]))

# Plot the results
plt.figure(3)
# plt.scatter(X, y, label='Data with noise')
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.scatter(X_val, y_val, color='green', label='Validation Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression Fit')
plt.legend()
plt.show()
