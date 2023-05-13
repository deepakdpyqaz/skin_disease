from messages.requests_pb2 import Input, Output
from tensorflow import keras
import tensorflow as tf
import numpy as np
from distributed import Client
from functools import lru_cache
import gdown
import os
from zipfile import ZipFile
import shutil
from pathlib import Path


EPOCHS = 2


def load_dir_generator(dir):
    for f in Path(dir.decode("ascii")).glob("*.npy"):
        yield np.load(f)


@lru_cache(maxsize=6)
def create_dataset(X_dir, y_dir, batch_size=32):
    X = tf.data.Dataset.from_generator(
        load_dir_generator,
        args=(X_dir,),
        output_signature=(tf.TensorSpec(shape=(192, 256, 3), dtype=tf.float32)),
    )
    y = tf.data.Dataset.from_generator(
        load_dir_generator,
        args=(y_dir,),
        output_signature=(tf.TensorSpec(shape=(7,), dtype=tf.float32)),
    )
    ds = tf.data.Dataset.zip((X, y))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE).repeat()
    return ds


def data_augmentation():
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomContrast(0.2),
        ]
    )
    return data_augmentation


def fitnessfunction(particle: Input) -> Output:
    """
    This is the fitness function that will be used to evaluate the particles
    """
    # Training Densenet

    train_ds = create_dataset("X_train", "y_train")
    test_ds = create_dataset("X_test", "y_test")
    val_ds = create_dataset("X_val", "y_val")
    train_steps_per_epoch = len(os.listdir("X_train")) // particle.batch_size
    val_steps_per_epoch = len(os.listdir("X_val")) // particle.batch_size
    test_steps_per_epoch = len(os.listdir("X_test")) // particle.batch_size

    input_layer = keras.layers.Input(shape=(192, 256, 3))
    augmented_image = data_augmentation()(input_layer)

    pre_trained_densenet_model = keras.applications.DenseNet201(
        input_shape=(192, 256, 3), include_top=False, weights="imagenet"
    )

    for layer in pre_trained_densenet_model.layers:
        layer.trainable = False

    x = pre_trained_densenet_model(augmented_image)
    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(7, activation="softmax")(x)

    densenet_model = keras.models.Model(input_layer, x)
    optimizer = keras.optimizers.Adam(
        learning_rate=particle.lr,
        beta_1=particle.b1,
        beta_2=particle.b2,
        epsilon=particle.epsilon,
        weight_decay=particle.weight_decay,
        amsgrad=particle.amsgrad,
    )
    densenet_model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    epochs = 1
    densenet_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
    )

    for layer in pre_trained_densenet_model.layers:
        layer.trainable = True

    optimizer = keras.optimizers.Adam(
        learning_rate=particle.lr,
        beta_1=particle.b1,
        beta_2=particle.b2,
        epsilon=particle.epsilon,
        weight_decay=particle.weight_decay,
        amsgrad=particle.amsgrad,
    )

    densenet_model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
    )

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc",
        patience=particle.patience,
        verbose=1,
        factor=particle.factor,
        min_lr=particle.lr / 10,
        cooldown=particle.cooldown,
    )

    epochs = EPOCHS
    densenet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[learning_rate_reduction],
    )

    # Training inception
    pre_trained_inception_model = keras.applications.InceptionV3(
        input_shape=(192, 256, 3), include_top=False, weights="imagenet"
    )

    for layer in pre_trained_inception_model.layers:
        layer.trainable = False

    x = pre_trained_inception_model(augmented_image)
    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(7, activation="softmax")(x)

    inception_model = keras.models.Model(pre_trained_inception_model.input, x)
    optimizer = keras.optimizers.Adam(
        learning_rate=particle.lr,
        beta_1=particle.b1,
        beta_2=particle.b2,
        epsilon=particle.epsilon,
        weight_decay=particle.weight_decay,
        amsgrad=particle.amsgrad,
    )
    inception_model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    epochs = 1
    history = inception_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        verbose=1,
    )

    for layer in pre_trained_inception_model.layers:
        layer.trainable = True
    optimizer = keras.optimizers.Adam(
        learning_rate=particle.lr,
        beta_1=particle.b1,
        beta_2=particle.b2,
        epsilon=particle.epsilon,
        weight_decay=particle.weight_decay,
        amsgrad=particle.amsgrad,
    )
    inception_model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
    )

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc",
        patience=particle.patience,
        verbose=1,
        factor=particle.factor,
        min_lr=particle.lr / 10,
        cooldown=particle.cooldown,
    )

    epochs = EPOCHS
    history = inception_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[learning_rate_reduction],
    )

    x_dense = densenet_model(augmented_image)
    x_incep = inception_model(augmented_image)
    x = keras.layers.Average()([x_dense, x_incep])
    ensemble_model = keras.models.Model(input_layer, x, "ensemble_model")
    ensemble_model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    scoresList = ensemble_model.evaluate(test_ds, verbose=1, steps=test_steps_per_epoch)
    score = scoresList[1]
    return score

client = Client(name="mrfo_btp",func=fitnessfunction)