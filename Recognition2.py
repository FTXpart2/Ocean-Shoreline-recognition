import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 28
img_width = 28
batch_size = 2

model = keras.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(16,3, padding = 'same'),
    layers.Conv2D(32,3, padding = 'same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])

ds_train = tf.keras.preprocessing.images_dataset_from_directory(
    'Ocean-Shoreline-Recognition/images/'
    labels='inferred',
    label_mode = "int", 
    class_names=['Ocean','Border']
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",

)

ds_validation = tf.keras.preprocessing.images_dataset_from_directory(
    'Ocean-Shoreline-Recognition/images/'
    labels='inferred',
    label_mode = "int", 
    class_names=['Ocean','Border']
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",

)

def augment(x,y):
    image = tf.image.random_brightness(x,max_delta=0.05)
    return image, y

ds_train = ds_train.map(augment)

for epochs in range(10):
    for x, y in ds_train:
        pass


model.compile(
    optimizer = keras.optimizers.Adam(),
    loss=[
        keras.losses.CategoricalCrossentrophy(from_logits=True),

    ],
    metrics=["accuracy"],

)
model.fit(ds_train, epochs = 10, verbose=2)

