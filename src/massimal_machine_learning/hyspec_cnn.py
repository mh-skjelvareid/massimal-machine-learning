import math
from collections.abc import Iterable

import numpy as np
import tensorflow as tf


def pad_image_to_multiple(image: np.ndarray, multiple: int) -> np.ndarray:
    """
    Zero-pad image spatially to a multiple of a given number.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D numpy array.
    multiple : int
        The integer multiple to pad to.

    Returns
    -------
    np.ndarray
        Padded image array.

    Examples
    --------
    >>> image = np.ones((2, 5))
    >>> pad_image_to_multiple(image, 4)
    array([[1., 1., 1., 1., 1., 0., 0., 0.],
           [1., 1., 1., 1., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    nrows_pad = math.ceil(image.shape[0] / multiple) * multiple
    ncols_pad = math.ceil(image.shape[1] / multiple) * multiple
    if image.ndim == 2:
        image_padded = np.zeros_like(image, shape=[nrows_pad, ncols_pad])
        image_padded[0 : image.shape[0], 0 : image.shape[1]] = image
    else:
        image_padded = np.zeros_like(
            image, shape=[nrows_pad, ncols_pad, image.shape[-1]]
        )
        image_padded[0 : image.shape[0], 0 : image.shape[1], :] = image
    return image_padded


# Function for extracting tiles
def labeled_image_to_tensor_tiles(
    image: np.ndarray,
    labels: np.ndarray,
    tile_shape: tuple[int, int],
    tile_strides: tuple[int, int] | None = None,
    padding: str = "SAME",
    min_labeled_fraction: float = 0.05,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Split image and label mask into smaller tiles.

    Parameters
    ----------
    image : np.ndarray
        3D numpy array with dimensions (rows, columns, channels).
    labels : np.ndarray
        2D numpy array with dimensions (rows, columns).
    tile_shape : tuple of int
        (tile_rows, tile_cols).
    tile_strides : tuple of int, optional
        (row_stride, col_stride). If None, set equal to tile_shape (no overlap).
    padding : str, optional
        'VALID' or 'SAME'. Default is 'SAME'.
    min_labeled_fraction : float, optional
        Filter out tiles with low number of labeled pixels. Default is 0.05.

    Returns
    -------
    image_tiles : tf.Tensor
        Tensor of image tiles.
    label_tiles : tf.Tensor
        Tensor of label tiles.
    """
    if tile_strides is None:
        tile_strides = tile_shape

    image_tensor = tf.reshape(tf.convert_to_tensor(image), (1,) + image.shape)
    label_tensor = tf.reshape(tf.convert_to_tensor(labels), (1,) + labels.shape + (1,))

    sizes = [1, *tile_shape, 1]
    strides = [1, *tile_strides, 1]
    rates = [1, 1, 1, 1]

    image_tiles = tf.image.extract_patches(
        image_tensor, sizes, strides, rates, padding=padding
    )
    image_tiles = tf.reshape(image_tiles, [-1, *tile_shape, image.shape[-1]])
    label_tiles = tf.image.extract_patches(
        label_tensor, sizes, strides, rates, padding=padding
    )
    label_tiles = tf.reshape(label_tiles, [-1, *tile_shape])

    # Filter out tiles with zero or few annotated pixels (optional)
    if min_labeled_fraction > 0:
        labeled_tiles_mask = np.array(
            [
                (np.count_nonzero(tile) / np.size(tile)) > min_labeled_fraction
                for tile in label_tiles
            ]
        )
        image_tiles = tf.boolean_mask(image_tiles, labeled_tiles_mask)
        label_tiles = tf.boolean_mask(label_tiles, labeled_tiles_mask)

    return image_tiles, label_tiles


def resampling_layer(
    resampling_type: str,
    filter_channels: int,
    kernel_size: int,
    resampling_factor: int = 2,
    name: str | None = None,
    initializer_mean: float = 0.0,
    initializer_std: float = 0.02,
    apply_batchnorm: bool = True,
    apply_dropout: bool = False,
    dropout_rate: float = 0.5,
) -> tf.keras.Sequential:
    """
    Spatial resampling 2D convolutional layer.

    Parameters
    ----------
    resampling_type : str
        'downsample' (convolution) or 'upsample' (transpose convolution).
    filter_channels : int
        Number of filters / output channels.
    kernel_size : int
        Spatial size of convolutional kernel.
    resampling_factor : int, optional
        Stride for resampling. Default is 2.
    name : str, optional
        Name of the layer.
    initializer_mean : float, optional
        Mean for kernel initializer. Default is 0.0.
    initializer_std : float, optional
        Std for kernel initializer. Default is 0.02.
    apply_batchnorm : bool, optional
        Whether to apply batch normalization. Default is True.
    apply_dropout : bool, optional
        Whether to apply dropout. Default is False.
    dropout_rate : float, optional
        Dropout rate. Default is 0.5.

    Returns
    -------
    tf.keras.Sequential
        The resampling layer.

    Notes
    -----
    Based on TF example pix2pix: https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    # Validate resampling layer type
    if resampling_type not in ["downsample", "upsample"]:
        raise ValueError(f"{resampling_type} is not a valid resampling type.")

    # Create kernel initializer for normally distributed random numbers
    initializer = tf.random_normal_initializer(
        mean=initializer_mean, stddev=initializer_std
    )

    # Initialize as sequential (stack of layers)
    resamp_layer = tf.keras.Sequential(name=name)

    # Add 2D convolutional layer
    if resampling_type == "downsample":
        resamp_layer.add(
            tf.keras.layers.Conv2D(
                filter_channels,
                kernel_size,
                strides=resampling_factor,
                padding="same",
                kernel_initializer=initializer,
                use_bias=not (apply_batchnorm),
            )
        )
    else:
        resamp_layer.add(
            tf.keras.layers.Conv2DTranspose(
                filter_channels,
                kernel_size,
                strides=resampling_factor,
                padding="same",
                kernel_initializer=initializer,
                use_bias=not (apply_batchnorm),
            )
        )

    # Add (optional) batch normalization layer
    if apply_batchnorm:
        resamp_layer.add(tf.keras.layers.BatchNormalization())

    # Add (optional) dropout layer
    if apply_dropout:
        resamp_layer.add(tf.keras.layers.Dropout(dropout_rate))

    # Add activation layer
    if resampling_type == "downsample":
        resamp_layer.add(tf.keras.layers.LeakyReLU())
    else:
        resamp_layer.add(tf.keras.layers.ReLU())

    return resamp_layer


def unet(
    input_channels: int,
    output_channels: int,
    first_layer_channels: int,
    depth: int,
    model_name: str | None = None,
    flip_aug: bool = True,
    trans_aug: bool = False,
    apply_batchnorm: bool | Iterable[bool] = True,
    apply_dropout: bool | Iterable[bool] = False,
) -> tf.keras.Model:
    """
    Simple encoder-decoder U-Net architecture.

    Parameters
    ----------
    input_channels : int
        Number of channels in input image.
    output_channels : int
        Number of classes to segment between.
    first_layer_channels : int
        Number of channels in first downsampling layer.
    depth : int
        Number of resampling steps to perform.
    model_name : str, optional
        Name of model.
    flip_aug : bool, optional
        If True, include RandomFlip augmentation layer.
    trans_aug : bool, optional
        If True, include RandomTranslation augmentation layer.
    apply_batchnorm : bool or Iterable[bool], optional
        Use batch normalization in layers.
    apply_dropout : bool or Iterable[bool], optional
        Use dropout in layers.

    Returns
    -------
    tf.keras.Model
        Keras U-Net model.

    Notes
    -----
    Based on TF tutorial: https://www.tensorflow.org/tutorials/images/segmentation
    """
    resamp_kernel_size = 4

    # Create vectors for batchnorm / dropout booleans if scalar
    if not isinstance(apply_batchnorm, Iterable):
        apply_batchnorm = [apply_batchnorm for _ in range(depth * 2)]

    if not isinstance(apply_dropout, Iterable):
        apply_dropout = [apply_dropout for _ in range(depth * 2)]

    # Define input
    inputs = tf.keras.layers.Input(
        shape=[None, None, input_channels], name="input_image"
    )  # Using None to signal variable image width and height (Ny,Nx,3)
    x = inputs  # x used as temparary variable for data flowing between layers

    # Add augmentation layer(s)
    if flip_aug or trans_aug:
        aug_layer = tf.keras.Sequential(name="augmentation")
        if flip_aug:
            aug_layer.add(tf.keras.layers.RandomFlip())
        if trans_aug:
            aug_layer.add(
                tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
            )
        x = aug_layer(x)

    # Add initial convolution layer with same resolution as input image
    x = tf.keras.layers.Conv2D(
        first_layer_channels,
        kernel_size=3,
        padding="same",
        name="initial_convolution",
        activation="relu",
    )(x)

    # Define downsampling layers
    down_stack = []
    nchannels_downsamp = [first_layer_channels * (2 ** (i + 1)) for i in range(depth)]
    names_downsamp = [f"downsamp_res_1_{(2 ** (i + 1))}" for i in range(depth)]
    for channels, name, batchnorm, dropout in zip(
        nchannels_downsamp,
        names_downsamp,
        apply_batchnorm[0:depth],
        apply_dropout[0:depth],
    ):
        down_stack.append(
            resampling_layer(
                "downsample",
                channels,
                resamp_kernel_size,
                name=name,
                apply_batchnorm=batchnorm,
                apply_dropout=dropout,
            )
        )

    # Define upsampling layers
    up_stack = []
    nchannels_upsamp = [first_layer_channels * (2 ** (depth - 1))] + [
        first_layer_channels * (2**i) for i in range(depth - 1, 0, -1)
    ]
    names_upsamp = [f"upsamp_res_1_{2**i}" for i in range(depth - 1, -1, -1)]
    for channels, name, batchnorm, dropout in zip(
        nchannels_upsamp, names_upsamp, apply_batchnorm[depth:], apply_dropout[depth:]
    ):
        up_stack.append(
            resampling_layer(
                "upsample",
                channels,
                resamp_kernel_size,
                name=name,
                apply_batchnorm=batchnorm,
                apply_dropout=dropout,
            )
        )

    # Downsampling through the model
    skips = [x]  # Add output from first layer (before downsampling) to skips list
    for down in down_stack:
        x = down(x)  # Run input x through layer, then set x equal to output
        skips.append(x)  # Add layer output to skips list

    skips = reversed(
        skips[:-1]
    )  # Reverse list, and don't include skip for last layer ("bottom of U")

    # Upsampling and establishing the skip connections
    names_skip = [f"skipconnection_res_1_{2**i}" for i in range(depth - 1, -1, -1)]
    for up, skip, skipname in zip(up_stack, skips, names_skip):
        x = up(x)  # Run input x through layer, then set x to output
        x = tf.keras.layers.Concatenate(name=skipname)(
            [x, skip]
        )  # Stack layer output together with skip connection

    # Final layer
    last = tf.keras.layers.Conv2D(
        output_channels,
        kernel_size=3,
        padding="same",
        activation="softmax",
        name="classification",
    )
    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name=model_name)

    return model


def add_background_zero_weight(
    image: tf.Tensor, labels: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Add weight image with zero weight for background.

    Parameters
    ----------
    image : tf.Tensor
        Input image tensor.
    labels : tf.Tensor
        Label tensor.

    Returns
    -------
    image : tf.Tensor
        Input image tensor (unchanged).
    labels : tf.Tensor
        Label tensor (unchanged).
    sample_weights : tf.Tensor
        Tensor with zero weight for background pixels.
    """
    label_mask = tf.greater(labels, 0)
    zeros = tf.zeros_like(labels, dtype=tf.float32)
    ones = tf.ones_like(labels, dtype=tf.float32)

    # "Multiplex" using label mask, ones for annotated pixels, zeros for background
    sample_weights = tf.where(label_mask, ones, zeros)  #

    return image, labels, sample_weights


def unet_classify_single_image(unet: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """
    Classify single image using UNet.

    Parameters
    ----------
    unet : tf.keras.Model
        Trained Unet model (Keras).
    image : np.ndarray
        Single image (3D numpy array).

    Returns
    -------
    np.ndarray
        2D image with integer class labels.
    """

    # Get activations by running predict(), insert extra dimension for 1-element batch
    activations = np.squeeze(unet.predict(np.expand_dims(image, axis=0)))
    labels = np.argmax(activations, axis=2)
    return labels


def unet_classify_image_batch(unet: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """
    Classify image batch using UNet.

    Parameters
    ----------
    unet : tf.keras.Model
        Trained Unet model (Keras).
    batch : np.ndarray
        Batch of images (4D numpy array).

    Returns
    -------
    np.ndarray
        3D array with integer class labels.
    """

    # Get activations by running predict(), use argmax to find class label
    activations = unet.predict(batch)
    labels = np.argmax(activations, axis=3)
    return labels
