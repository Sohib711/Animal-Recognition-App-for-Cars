import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from src.constants import BINARY_LABELS, LABELS, BATCH_SIZE, SEED


def data_gen(rescale_train: bool = True) -> tuple:
    """
    Create two ImageDataGenerators: one for training and one for testing/validation.

    Args:
        rescale_train (bool): If True, the training data will be rescaled by 1./255. 
                              If False, the training data will not be rescaled. 
                              Default is True.

    Returns:
        tuple: A tuple containing:
               - train_datagen (ImageDataGenerator): Data generator for the training set.
               - test_datagen (ImageDataGenerator): Data generator for the testing/validation set.
    """
    
    if rescale_train:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            #shear_range = 0.2,
            #brightness_range=[0.1, 1.5],
            #zoom_range=[0.3, 1.5],
            #fill_mode = 'nearest',
            horizontal_flip=True
        )
    else:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            #shear_range = 0.2,
            #brightness_range=[0.1, 1.5],
            #zoom_range=[0.3, 1.5],
            #fill_mode = 'nearest',
            horizontal_flip=True
        )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    return train_datagen, test_datagen


def norm_X(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Normalize the training and testing features by scaling pixel values to [0, 1].

    Args:
        X_train (np.ndarray): The training feature set.
        X_test (np.ndarray): The testing feature set.

    Returns:
        tuple: A tuple containing:
               - train_norm (np.ndarray): The normalized training features.
               - test_norm (np.ndarray): The normalized testing features.
    """
    
    train_norm = (X_train / 255.0).astype('float32')
    test_norm = (X_test / 255.0).astype('float32')

    print("=" * 50)
    print(f"Min and Max values for X train: {np.min(train_norm)}, {np.max(train_norm)}")
    print(f"Min and Max values for X test: {np.min(test_norm)}, {np.max(test_norm)}")
    print("=" * 50)

    return train_norm, test_norm


def binary_y(y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Transform labels to binary labels: vehicles (0) and animals (1).

    Args:
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The testing labels.

    Returns:
        tuple: A tuple containing:
               - y_train_binary (np.ndarray): The binary labels for the training set.
               - y_test_binary (np.ndarray): The binary labels for the testing set.
    """
    
    y_train_binary = np.vectorize(BINARY_LABELS.get)(y_train.flatten())
    y_test_binary = np.vectorize(BINARY_LABELS.get)(y_test.flatten())

    print("=" * 70)
    print(f"First 10 binary labels for training set: \n{y_train_binary[:10]}\n{np.vectorize(LABELS.get)(y_train.flatten())[:10]}")
    print("=" * 70)
    print(f"First 10 binary labels for test set: \n{y_test_binary[:10]}\n{np.vectorize(LABELS.get)(y_test.flatten())[:10]}")
    print("=" * 70)

    return y_train_binary, y_test_binary


def train_splitting(X_train: np.ndarray, y_train: np.ndarray, test_size: float = 0.1) -> tuple:
    """
    Split the training set into training and validation sets.

    Args:
        X_train (np.ndarray): The training feature set.
        y_train (np.ndarray): The training labels.
        test_size (float): The proportion of the dataset to include in the validation set (default is 0.1).

    Returns:
        tuple: A tuple containing:
               - X_train_split (np.ndarray): The training features after splitting.
               - X_val (np.ndarray): The validation features.
               - y_train_binary_split (np.ndarray): The training labels after splitting.
               - y_val_binary (np.ndarray): The validation labels.
    """
    
    X_train_split, X_val, y_train_binary_split, y_val_binary = train_test_split(
        X_train, y_train, test_size=test_size, shuffle=True, stratify=y_train
    )

    print("=" * 60)
    print("Training information:")
    print(f"X train percentage for training after splitting: {(X_train_split.shape[0] * 100 / X_train.shape[0]):.0f}%")
    print(f"X train shape after splitting: {X_train_split.shape}")
    print(f"y train shape after splitting: {y_train_binary_split.shape}")
    print("=" * 60)
    print("Validation information:")
    print(f"X val percentage for validation after splitting: {(X_val.shape[0] * 100 / X_train.shape[0]):.0f}%")
    print(f"X val shape after splitting: {X_val.shape}")
    print(f"y val shape after splitting: {y_val_binary.shape}")
    print("=" * 60)

    return X_train_split, X_val, y_train_binary_split, y_val_binary


def apply_data_gen(train_datagen: ImageDataGenerator, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray, 
                   test_datagen: ImageDataGenerator, 
                   X_val: np.ndarray, 
                   y_val: np.ndarray) -> tuple:
    """
    Apply data generators to training and validation datasets.

    Args:
        train_datagen (ImageDataGenerator): The data generator for the training set.
        X_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        test_datagen (ImageDataGenerator): The data generator for the validation set.
        X_val (np.ndarray): The validation features.
        y_val (np.ndarray): The validation labels.

    Returns:
        tuple: A tuple containing:
               - train_gen (DataFrameIterator): The training data generator.
               - val_gen (DataFrameIterator): The validation data generator.
    """
    
    train_gen = train_datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_gen = test_datagen.flow(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    return train_gen, val_gen