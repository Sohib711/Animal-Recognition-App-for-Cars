import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import List, Any
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, MaxPooling2D, BatchNormalization, Dropout, 
                                     Conv2D, GlobalAveragePooling2D)
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback


from src.constants import BATCH_SIZE, SEED, INPUT_SHAPE
from src.utils import calculate_mean_std


def build_cnn_model(input_shape: tuple, learning_rate: float = 1e-3, lr_dense: float = 1e-3) -> Sequential:
    """
    Builds and compiles a CNN model for binary classification.

    Args:
        input_shape (tuple): The shape of the input data (height, width, channels).
        learning_rate (float): Learning rate for the optimizer.
        lr_dense (float): Learning rate for the regularizers in the Dense layers.

    Returns:
        Sequential: Compiled CNN model.
    """
    
    clear_session()  # Clear previous models from memory
    model = Sequential()

    # Input Layer
    model.add(Input(shape=input_shape, name="Input Layer"))

    # Block 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu',
                     kernel_initializer=initializers.GlorotUniform(), name="block1_conv1"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', name="block1_conv2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", name="block1_maxpool"))
    model.add(Dropout(0.3))

    # Block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu', name="block2_conv1"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu', name="block2_conv2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", name="block2_maxpool"))
    model.add(Dropout(0.3))

    # Block 3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu', name="block3_conv1"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu', name="block3_conv2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid", name="block3_maxpool"))
    model.add(Dropout(0.3))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.L2(lr_dense)))
    """model.add(Dropout(0.2))
    model.add(Dense(
        units = 128, activation = 'relu', kernel_regularizer = regularizers.L2(lr_dense)
    ))"""
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

    # Compile model
    adam_opt = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy', 'precision'])

    return model


def train_model(
    model: Any,
    train_generator: ImageDataGenerator,
    val_generator: ImageDataGenerator,
    epochs: int = 30,
    early_stop_patience: int = 5,
    class_imb: bool = False
) -> Any:
    """
    Trains a CNN model using data generators for training and validation.

    Args:
        model (Any): The CNN model to train.
        train_generator (ImageDataGenerator): The training data generator.
        val_generator (ImageDataGenerator): The validation data generator.
        epochs (int): Number of training epochs.
        early_stop_patience (int): Patience for early stopping.
        class_imb (bool): boolean to decide whether to apply class balancings

    Returns:
        Any: Training history of the model.
    """
    
    # Calculate steps per epoch and validation steps
    #epochs_step = train_generator.n // train_generator.batch_size
    #val_steps = val_generator.n // val_generator.batch_size
    
    # Early stopping callback
    callback_early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)

    if class_imb:
        # Compute class weights to handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
        weights = {0: class_weights[0], 1: class_weights[1]}

        history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=weights,
        validation_steps=None,
        steps_per_epoch=None,
        callbacks=[callback_early_stop],
        verbose=1
    )
    else:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=None,
            steps_per_epoch=None,
            callbacks=[callback_early_stop],
            verbose=1
        )

    return history




def build_resnet_model(adam_lr: float = 1e-3) -> keras.Model:
    """
    Builds and compiles a ResNet model for binary classification.

    Args:
        adam_lr (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: A compiled Keras model.
    """
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.UpSampling2D(size=(7, 7))(inputs)

    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.Dense(1028, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    adam_opt = Adam(learning_rate=adam_lr)

    model.compile(loss='binary_crossentropy', 
                  optimizer=adam_opt, 
                  metrics=['accuracy', 'precision'])
    
    return model



def build_inception_model(adam_lr: float = 1e-3) -> keras.Model:
    """
    Builds and compiles an InceptionV3 model for binary classification using transfer learning.

    Args:
        adam_lr (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: A compiled Keras model.
    """
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.UpSampling2D(size=(7, 7))(inputs)

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Fine-tuning the base model

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    adam_opt = Adam(learning_rate=adam_lr)

    model.compile(loss='binary_crossentropy', 
                  optimizer=adam_opt, 
                  metrics=['accuracy', 'precision'])
    
    return model


def build_efficientnet_model(adam_lr: float = 1e-3) -> keras.Model:
    """
    Builds and compiles an EfficientNetB0 model for binary classification using transfer learning.

    Args:
        adam_lr (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: A compiled Keras model.
    """
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.UpSampling2D(size=(7, 7))(inputs)

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Fine-tuning the base model

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    adam_opt = Adam(learning_rate=adam_lr)

    model.compile(loss='binary_crossentropy', 
                  optimizer=adam_opt, 
                  metrics=['accuracy', 'precision'])
    
    return model


def train_tf_model(model: keras.Model, 
                train_generator: keras.preprocessing.image.ImageDataGenerator, 
                val_generator: keras.preprocessing.image.ImageDataGenerator, 
                epochs: int = 30, 
                class_imb: bool = False) -> keras.callbacks.History:
    """
    Trains a given Keras model using the provided training and validation data generators.

    Args:
        model (keras.Model): The Keras model to train.
        train_generator (keras.preprocessing.image.ImageDataGenerator): Data generator for the training set.
        val_generator (keras.preprocessing.image.ImageDataGenerator): Data generator for the validation set.
        epochs (int): Number of epochs for training. Default is 30.
        class_imb (bool): boolean to decide whether to apply class balancings

    Returns:
        keras.callbacks.History: History object containing training metrics.
    """
    callbacks = get_callbacks()
    #epochs_step = train_generator.n // train_generator.batch_size
    #val_steps = val_generator.n // val_generator.batch_size

    if class_imb:
        # Compute class weights to handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
        weights = {0: class_weights[0], 1: class_weights[1]}

        history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=weights,
        validation_steps=None,
        steps_per_epoch=None,
        callbacks=callbacks,
        verbose=1)
    else:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=None,
            steps_per_epoch=None,
            callbacks=callbacks,
            verbose=1
        )

    return history


def get_callbacks(acc_stop: bool = False) -> List[Callback]:
    """
    Creates a list of callbacks for model training.

    Args:
        acc_stop (bool): Whether to include a custom callback to stop training 
                         when a certain accuracy is reached.

    Returns:
        List[Callback]: A list of Keras callbacks.
    """
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    if acc_stop:
        class CustomCallbackAccStop(Callback):
            def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
                if logs.get('accuracy', 0) > 0.99:
                    print(f"Reached 99% accuracy on epoch {epoch}, stopping training!")
                    self.model.stop_training = True

        custom_cb = CustomCallbackAccStop()
        return [early_stopping, reduce_lr, model_checkpoint, custom_cb]
    else:
        return [early_stopping, reduce_lr, model_checkpoint]


def add_to_report(model_name: str, 
                  metrics: dict, 
                  description: str, 
                  report: pd.DataFrame, 
                  file_path: str) -> None:
    """
    Add model details, including name, description, and performance metrics, to a report.

    Args:
        model_name (str): The name of the model.
        metrics (dict): A dictionary of metrics where keys are metric names and values are lists of values.
        description (str): A brief description of the model.
        report (pd.DataFrame): The existing report DataFrame to which new data will be added.
        file_path (str): The file path to save the updated report.

    Returns:
        None: Updates the report CSV file.
    """
    
    metrics_calculated = {}
    for key, values in metrics.items():
        mean, std = calculate_mean_std(values)
        metrics_calculated[f'{key} Mean'] = mean
        metrics_calculated[f'{key} Std'] = std

    new_row = {
        'Model': model_name,
        'Description': description,
        **metrics_calculated
    }
    new_row_df = pd.DataFrame([new_row])
    report = pd.concat([report, new_row_df], ignore_index=True)
    report.to_csv(file_path, index=False)


def evaluate_model_and_save_results(model: tf.keras.Model, 
                                     model_name: str, 
                                     X_test: np.ndarray, 
                                     y_test_binary: np.ndarray, 
                                     results_file: str = 'Results/test_metrics.csv') -> None:
    """
    Evaluate the model on the test set and save the results to a CSV file.

    Args:
        model (tf.keras.Model): The trained Keras model to evaluate.
        model_name (str): The name of the model for reporting.
        X_test (np.ndarray): The test set features.
        y_test_binary (np.ndarray): The binary labels for the test set.
        results_file (str): The file path to save the evaluation results.

    Returns:
        None: The function saves the results to a CSV file and prints evaluation metrics.
    """
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(
        X_test,
        y_test_binary,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False  # Keep shuffle False for evaluation
    )
    
    evaluation = model.evaluate(test_generator)
    test_loss, test_accuracy, test_precision = evaluation[0], evaluation[1] * 100, evaluation[2] * 100
    
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Test Accuracy: {test_accuracy:.3f}%')
    print(f'Test Precision: {test_precision:.3f}%')
    
    results_dir = os.path.dirname(results_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
    else:
        df = pd.DataFrame(columns=['Model', 'Test Loss', 'Test Accuracy', 'Test Precision'])
    
    new_row = {
        'Model': model_name,
        'Test Loss': round(test_loss, 3),
        'Test Accuracy': round(test_accuracy, 3),
        'Test Precision': round(test_precision, 3)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(results_file, index=False)
    
    y_pred_prob = model.predict(test_generator)
    y_pred = tf.where(y_pred_prob <= 0.5, 0, 1).numpy()  # Convert probabilities to binary labels
    
    cm = confusion_matrix(test_generator.y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(xticks_rotation='horizontal', ax=ax, cmap=plt.cm.Blues)
    plt.show()
    
    print(classification_report(test_generator.y, y_pred))