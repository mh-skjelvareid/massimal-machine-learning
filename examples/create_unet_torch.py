from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop

from massimal_machine_learning.hyspec_cnn import unet
from massimal_machine_learning.hyspec_cnn_torch import UNet, unet_classify_single_image

# Define parameters
LEARNING_RATE = 0.0001
FIRST_LAYER_CHANNELS = 32
INPUT_CHANNELS = 8
OUTPUT_CHANNELS = 8
BATCH_SIZE = 8
DEPTH = 2
EPOCHS = 5

# Define dataset paths
dataset_base_path = Path(r"C:\Users\mha114\Dropbox\Datasets\Hyperspectral\Massimal_Vega_IGARSS2023")
train_tiles_path = (
    dataset_base_path / "PCA_images_and_tiles/Training/PCA-Tiles/20220823_Vega_Sola_Train_Tiles"
)
val_tiles_path = (
    dataset_base_path / "PCA_images_and_tiles/Validation/PCA-Tiles/20220823_Vega_Sola_Val_Tiles"
)


def main():
    # Load datasets
    # train_dataset = tf.data.Dataset.load(str(train_tiles_path))
    val_dataset = tf.data.Dataset.load(str(val_tiles_path))

    # # Get number of tiles in each dataset, and dataset shape
    # n_tiles_train = train_dataset.cardinality()
    # n_tiles_val = val_dataset.cardinality()
    # tile_nrows, tile_ncols, tile_nchannels = train_dataset.element_spec[0].shape.as_list()
    # print(f"Number of training tiles: {n_tiles_train}")
    # print(f"Number of validation tiles: {n_tiles_val}")
    # print(f"Tile data shape (PCA tiles): {(tile_nrows, tile_ncols, tile_nchannels)}")

    images = []
    labels = []
    for data in val_dataset.take(3).as_numpy_iterator():
        image_tile, label_tile, file_name = data
        print(f"Tile from {file_name}")
        print(f"{image_tile.shape=}")
        print(f"{label_tile.shape=}\n")
        images.append(image_tile)
        labels.append(label_tile)

    # # Create the U-Net model
    unet_model = UNet(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        first_layer_channels=FIRST_LAYER_CHANNELS,
        depth=DEPTH,
    )

    if images:
        unet_classify_single_image(unet_model, images[0])
    # Fails at line 380 in hyspec_cnn_torch because upsampled image doesn't match size
    # of "skip" dataset (upsampled width/height is power of 2 + 1, e.g. 129, while skip
    # is only power of 2). Need to change padding for downsampling policy?


if __name__ == "__main__":
    main()
