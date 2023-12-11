import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.models import Model

BASE_DIR = "data"
BATCH_SIZE = 32
IMG_SIZE =(224,224)
EPOCHS = 10

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.:
            print()
            self.model.stop_training = True

def preprocessing(BASE_DIR,IMG_SIZE,BATCH_SIZE):

    Datagen = ImageDataGenerator(
        rescale=1/255.0,
        validation_split = 0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15)
    
    Train_Generator = Datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        target_size=IMG_SIZE,
        subset="validation",
        shuffle=False,
        class_mode="categorical"
    )
    Validation_Generator = Datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        target_size=IMG_SIZE,
        subset="training",
        shuffle=False,
        class_mode="categorical"
    )
    return Train_Generator,Validation_Generator

def get_class_names(Train_Generator):
    class_names = list(Train_Generator.class_indices.keys())
    return class_names

def create_model():
    model = keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, Train_Generator, Validation_Generator, EPOCHS):
        model.fit(Train_Generator, epochs=EPOCHS, validation_data=Validation_Generator)
        return model




if __name__ == '__main__':
    train_generator, validation_generator = preprocessing(BASE_DIR, IMG_SIZE, BATCH_SIZE)
    model = create_model()
    trained_model = train_model(model, train_generator, validation_generator, EPOCHS)
    trained_model.save("model_v1.h5")
