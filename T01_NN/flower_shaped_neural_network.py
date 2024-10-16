import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import sys

# Generate data
def generate_flower_data(samples):
    # Center petal
    t = np.linspace(0, 2*np.pi, samples // 2)
    r1 = 0.75 + 0.25*np.sin(5*t)
    r2 = 0.5 + 0.25*np.sin(5*t)
    y_inner = np.cos(5*t)

    x_inner = np.multiply(r1,np.cos(t))
    y_inner = np.multiply(r1,np.sin(t))
    x_outer = np.multiply(r2,np.cos(t))
    y_outer = np.multiply(r2,np.sin(t))

    # Outer petal
    t_outer = np.linspace(0, 2*np.pi, samples // 2)
    # x_outer = 1.5 * np.sin(t_outer)
    # y_outer = 1.5 * np.cos(t_outer)

    x = np.concatenate([x_inner, x_outer])
    y = np.concatenate([y_inner, y_outer])
    labels = np.concatenate([np.zeros(samples // 2), np.ones(samples // 2)])

    return x, y, labels


samples = 2000
x, y, labels = generate_flower_data(samples)
X = np.column_stack((x, y))
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show(block=False)
# plt.show()


# sys.exit()


# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.3, random_state=42)

# Build model
model = Sequential([
    Dense(10, input_dim=2, activation='sigmoid'),
    # Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.3), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=10)
plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show(block=False)
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show(block=False)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(Z)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='k', marker='o')
plt.show()
