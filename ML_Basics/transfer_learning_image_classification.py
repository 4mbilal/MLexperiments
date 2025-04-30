import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler


# Load MobileNetV2 (excluding top classification layers)
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)  # Customize the number of units
predictions = Dense(5, activation='softmax')(x)  # Ten classes in your case

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()




# Compile the model
# model.compile(optimizer='adam' ,loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])


# Define data paths
# train_data_dir = 'D:\\RnD\\Frameworks\\Datasets\\NEOMchallenge\\'
train_data_dir = 'D:\\RnD\\Frameworks\\Datasets\\flower_photos\\'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Set the validation split here
)

# Load training data
train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Adjust based on your model input size
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Specify 'training' for the training set
)


validation_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Specify 'validation' for the validation set
)


# Now 'training_data' contains 80% of your data, and 'validation_data' contains 20%.

# Load your data and train the model
# (Assuming you have already prepared your data in separate folders)
def lr_schedule(epoch):
    if epoch < 5:
        return 0.01
    else:
        return 0.01 * tf.math.exp(0.1 * (5 - epoch))

# Set up the learning rate scheduler
lr_callback = LearningRateScheduler(lr_schedule)

from datetime import datetime
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# Run on python terminal in the same conda environment, "tensorboard --logdir logs/scalars"
# Then see the graphs on http://localhost:6006/


# Train your model
model.fit(train_data, validation_data=validation_data, epochs=10, callbacks=[lr_callback,tensorboard_callback])
model.save("flowers.h5")