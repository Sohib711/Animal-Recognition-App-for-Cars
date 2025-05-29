# Set names labels
LABELS = {
    0 : 'airplane',
    1 : 'automobile',
    2 : 'bird',
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'frog',
    7 : 'horse',
    8 : 'ship',
    9 : 'truck'
}

# Set binary labels: vehicle vs animal
BINARY_LABELS = {
    0: 0,  # airplane -> vehicle
    1: 0,  # automobile -> vehicle
    8: 0,  # ship -> vehicle
    9: 0,  # truck -> vehicle
    2: 1,  # bird -> animal
    3: 1,  # cat -> animal
    4: 1,  # deer -> animal
    5: 1,  # dog -> animal
    6: 1,  # frog -> animal
    7: 1   # horse -> animal
}

# Define seed
SEED = 59

# Define number of classes
NUM_CLASSES = 2

# Define batch size 
BATCH_SIZE = 32

# Define input shape
INPUT_SHAPE = (32, 32, 3)