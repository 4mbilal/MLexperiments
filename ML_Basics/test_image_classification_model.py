import tensorflow as tf
import pathlib
import PIL
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

# Load your pre-trained Keras model
model = tf.keras.models.load_model("flowers.h5")
# data_dir = 'D:\\RnD\\Frameworks\\Datasets\\NEOMchallenge'
data_dir = 'D:\\RnD\\Frameworks\\Datasets\\flower_photos\\'

# Parameters
batch_size = 1
img_height, img_width = 224, 224

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
test_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Adjust based on your model input size
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Specify 'training' for the training set
)


# Predict and display
for image_batch, label_batch in test_ds:
    predictions = model.predict(image_batch)
    # print(predictions[0])
    # print(np.argmax(predictions[0]))
    predicted_label = np.argmax(predictions[0])
    class_names = test_ds.class_indices
    print(list(class_names)[predicted_label])
    out_img = image_batch[0]*255
    plt.imshow(out_img.astype("uint8"))
    plt.title(list(class_names)[predicted_label])
    plt.pause(0.01)
    plt.close()
