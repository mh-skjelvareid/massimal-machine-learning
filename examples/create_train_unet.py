from pathlib import Path

import tensorflow as tf
from keras.optimizers import RMSprop

from massimal_machine_learning.hyspec_cnn import unet

# Define parameters
LEARNING_RATE = 0.0001
FIRST_LAYER_CHANNELS = 32
OUTPUT_CHANNELS = 8
BATCH_SIZE = 8
DEPTH = 2
EPOCHS = 5

# Define dataset paths
dataset_base_path = Path(
    r"C:\Users\mha114\Dropbox\Datasets\Hyperspectral\Massimal_Vega_IGARSS2023"
)
train_tiles_path = (
    dataset_base_path
    / "PCA_images_and_tiles/Training/PCA-Tiles/20220823_Vega_Sola_Train_Tiles"
)
val_tiles_path = (
    dataset_base_path
    / "PCA_images_and_tiles/Validation/PCA-Tiles/20220823_Vega_Sola_Val_Tiles"
)

# Load datasets
train_dataset = tf.data.Dataset.load(str(train_tiles_path))
val_dataset = tf.data.Dataset.load(str(val_tiles_path))

# Get number of tiles in each dataset, and dataset shape
n_tiles_train = train_dataset.cardinality()
n_tiles_val = val_dataset.cardinality()
tile_nrows, tile_ncols, tile_nchannels = train_dataset.element_spec[0].shape.as_list()
print(f"Number of training tiles: {n_tiles_train}")
print(f"Number of validation tiles: {n_tiles_val}")
print(f"Tile data shape (PCA tiles): {(tile_nrows, tile_ncols, tile_nchannels)}")


# Create the U-Net model
unet_model = unet(
    input_channels=tile_nchannels,
    output_channels=OUTPUT_CHANNELS,
    first_layer_channels=FIRST_LAYER_CHANNELS,
    depth=DEPTH,
)
unet_model.summary()


def add_sample_weights(image, label, name):
    """Add sample weights, compatible with dataset "map" method."""
    # class_weights = tf.constant([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # Hard-coded for 7 classes
    class_weights = tf.constant(
        [0.0, 0.89, 1.24, 0.82, 0.65, 1.08, 1.54, 0.77]
    )  # Hard-coded for 7 classes
    class_weights = class_weights / tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return image, label, sample_weights


# Shuffle training dataset (tiles are originally ordered by image) and add sample weights
train_dataset = train_dataset.shuffle(buffer_size=n_tiles_train)
train_dataset = train_dataset.map(add_sample_weights)
val_dataset = val_dataset.map(add_sample_weights)


# Compile model
unet_model.compile(
    optimizer=RMSprop(0.0001),
    loss="sparse_categorical_crossentropy",
    weighted_metrics=[
        "sparse_categorical_accuracy"  # Sparse because classes are numbered, not one-hot
    ],
    metrics=[],
)
print("Model compiled")


# Train model
history = unet_model.fit(
    train_dataset.batch(BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_dataset.batch(BATCH_SIZE),
)
