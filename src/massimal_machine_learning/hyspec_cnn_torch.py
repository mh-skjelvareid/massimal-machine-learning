# Converted from TF to PyTorch by Claude Haiku 4.5 (GitHub CoPilot).
# NOTE: Not used for compare_hsi_msi_rgb project

import math
from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
        image_padded = np.zeros((nrows_pad, ncols_pad), dtype=image.dtype)
        image_padded[: image.shape[0], : image.shape[1]] = image
    else:
        image_padded = np.zeros((nrows_pad, ncols_pad, image.shape[-1]), dtype=image.dtype)
        image_padded[: image.shape[0], : image.shape[1], :] = image
    return image_padded


def labeled_image_to_tensor_tiles(
    image: np.ndarray,
    labels: np.ndarray,
    tile_shape: tuple[int, int],
    tile_strides: tuple[int, int] | None = None,
    padding: str = "SAME",
    min_labeled_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    image_tiles : torch.Tensor
        Tensor of image tiles with shape (num_tiles, tile_rows, tile_cols, channels).
    label_tiles : torch.Tensor
        Tensor of label tiles with shape (num_tiles, tile_rows, tile_cols).

    Notes
    -----
    PyTorch implementation using unfold operation. The 'SAME' padding is achieved
    by manually padding the input before unfold, and 'VALID' uses unfold directly.
    """
    if tile_strides is None:
        tile_strides = tile_shape

    # Convert numpy arrays to torch tensors
    # Convert from (H, W, C) to (1, C, H, W) for unfold operation
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    # Convert from (H, W) to (1, 1, H, W) for unfold operation
    label_tensor = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0).float()

    tile_h, tile_w = tile_shape
    stride_h, stride_w = tile_strides

    # Apply padding if 'SAME'
    if padding == "SAME":
        # Calculate padding needed
        pad_h = (
            (image.shape[0] - tile_h) % stride_h if (image.shape[0] - tile_h) % stride_h != 0 else 0
        )
        pad_w = (
            (image.shape[1] - tile_w) % stride_w if (image.shape[1] - tile_w) % stride_w != 0 else 0
        )
        pad_h = stride_h - pad_h if pad_h != 0 else 0
        pad_w = stride_w - pad_w if pad_w != 0 else 0

        # Pad: (left, right, top, bottom)
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
        label_tensor = F.pad(label_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    # Use unfold to extract patches
    # unfold(dimension, size, step)
    image_unfold = image_tensor.unfold(2, tile_h, stride_h).unfold(3, tile_w, stride_w)
    # shape: (1, C, num_h, num_w, tile_h, tile_w)
    image_tiles = image_unfold.permute(0, 2, 3, 4, 5, 1).contiguous()
    # shape: (1, num_h, num_w, tile_h, tile_w, C)
    image_tiles = image_tiles.view(-1, tile_h, tile_w, image.shape[-1])
    # shape: (num_tiles, tile_h, tile_w, C)

    label_unfold = label_tensor.unfold(2, tile_h, stride_h).unfold(3, tile_w, stride_w)
    # shape: (1, 1, num_h, num_w, tile_h, tile_w)
    label_tiles = label_unfold.squeeze(1).permute(0, 2, 3, 4, 5).contiguous()
    # shape: (1, num_h, num_w, tile_h, tile_w)
    label_tiles = label_tiles.view(-1, tile_h, tile_w)
    # shape: (num_tiles, tile_h, tile_w)

    # Filter out tiles with zero or few annotated pixels (optional)
    if min_labeled_fraction > 0:
        labeled_tiles_mask = np.array(
            [
                (torch.count_nonzero(tile).item() / tile.numel()) > min_labeled_fraction
                for tile in label_tiles
            ]
        )
        image_tiles = image_tiles[labeled_tiles_mask]
        label_tiles = label_tiles[labeled_tiles_mask]

    return image_tiles, label_tiles


class ResamplingLayer(nn.Module):
    """
    Spatial resampling 2D convolutional layer.

    Parameters
    ----------
    resampling_type : str
        'downsample' (convolution) or 'upsample' (transpose convolution).
    in_channels : int
        Number of input channels.
    filter_channels : int
        Number of filters / output channels.
    kernel_size : int
        Spatial size of convolutional kernel.
    resampling_factor : int, optional
        Stride for resampling. Default is 2.
    apply_batchnorm : bool, optional
        Whether to apply batch normalization. Default is True.
    apply_dropout : bool, optional
        Whether to apply dropout. Default is False.
    dropout_rate : float, optional
        Dropout rate. Default is 0.5.

    Notes
    -----
    Based on TF example pix2pix: https://www.tensorflow.org/tutorials/generative/pix2pix
    """

    def __init__(
        self,
        resampling_type: str,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        resampling_factor: int = 2,
        apply_batchnorm: bool = True,
        apply_dropout: bool = False,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        if resampling_type not in ["downsample", "upsample"]:
            raise ValueError(f"{resampling_type} is not a valid resampling type.")

        self.resampling_type = resampling_type
        layers = []

        if resampling_type == "downsample":
            # Downsampling with Conv2D
            layers.append(
                nn.Conv2d(
                    in_channels,
                    filter_channels,
                    kernel_size,
                    stride=resampling_factor,
                    padding=kernel_size // 2,
                    bias=not apply_batchnorm,
                )
            )
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(filter_channels))
            if apply_dropout:
                layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        else:
            # Upsampling with Conv2DTranspose
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    filter_channels,
                    kernel_size,
                    stride=resampling_factor,
                    padding=kernel_size // 2,
                    output_padding=resampling_factor - 1,
                    bias=not apply_batchnorm,
                )
            )
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(filter_channels))
            if apply_dropout:
                layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """
    Simple encoder-decoder U-Net architecture for image segmentation.

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
    flip_aug : bool, optional
        If True, include horizontal flip augmentation. Default is True.
    trans_aug : bool, optional
        If True, include translation augmentation. Default is False.
    apply_batchnorm : bool or list[bool], optional
        Use batch normalization in layers. Default is True.
    apply_dropout : bool or list[bool], optional
        Use dropout in layers. Default is False.

    Notes
    -----
    Based on TF tutorial: https://www.tensorflow.org/tutorials/images/segmentation
    Input tensors should be in (batch, channels, height, width) format.
    Output is (batch, output_channels, height, width).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        first_layer_channels: int,
        depth: int,
        flip_aug: bool = True,
        trans_aug: bool = False,
        apply_batchnorm: bool | list[bool] = True,
        apply_dropout: bool | list[bool] = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_layer_channels = first_layer_channels
        self.depth = depth

        resamp_kernel_size = 4

        # Create vectors for batchnorm / dropout booleans if scalar
        if not isinstance(apply_batchnorm, Iterable) or isinstance(apply_batchnorm, bool):
            apply_batchnorm = [apply_batchnorm for _ in range(depth * 2)]

        if not isinstance(apply_dropout, Iterable) or isinstance(apply_dropout, bool):
            apply_dropout = [apply_dropout for _ in range(depth * 2)]

        # Augmentation layers
        aug_layers = []
        if flip_aug:
            aug_layers.append(transforms.RandomHorizontalFlip())
        if trans_aug:
            aug_layers.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
        self.augmentation = transforms.Compose(aug_layers) if aug_layers else None

        # Initial convolution layer
        self.initial_conv = nn.Conv2d(
            input_channels,
            first_layer_channels,
            kernel_size=3,
            padding=1,
        )

        # Downsampling layers
        self.down_stack = nn.ModuleList()
        nchannels_downsamp = [first_layer_channels * (2 ** (i + 1)) for i in range(depth)]
        in_channels = first_layer_channels
        for i, (channels, batchnorm, dropout) in enumerate(
            zip(
                nchannels_downsamp,
                apply_batchnorm[0:depth],
                apply_dropout[0:depth],
            )
        ):
            self.down_stack.append(
                ResamplingLayer(
                    "downsample",
                    in_channels,
                    channels,
                    resamp_kernel_size,
                    apply_batchnorm=batchnorm,
                    apply_dropout=dropout,
                )
            )
            in_channels = channels

        # Upsampling layers
        self.up_stack = nn.ModuleList()
        nchannels_upsamp = [first_layer_channels * (2 ** (depth - 1))] + [
            first_layer_channels * (2**i) for i in range(depth - 1, 0, -1)
        ]
        for i, (channels, batchnorm, dropout) in enumerate(
            zip(nchannels_upsamp, apply_batchnorm[depth:], apply_dropout[depth:])
        ):
            # After concatenation, input channels are doubled
            up_in_channels = in_channels * 2 if i > 0 else in_channels
            self.up_stack.append(
                ResamplingLayer(
                    "upsample",
                    up_in_channels,
                    channels,
                    resamp_kernel_size,
                    apply_batchnorm=batchnorm,
                    apply_dropout=dropout,
                )
            )
            in_channels = channels

        # Final classification layer
        self.final_conv = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Apply augmentation if in training mode and augmentation is enabled
        if self.training and self.augmentation is not None:
            x = self.augmentation(x)

        # Initial convolution
        x = self.initial_conv(x)
        x = F.relu(x)

        # Downsampling path with skip connections
        skips = [x]
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        # Reverse skips for upsampling (excluding the last one)
        skips = skips[:-1]
        skips.reverse()

        # Upsampling path with skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)

        # Final classification layer
        x = self.final_conv(x)
        x = F.softmax(x, dim=1)

        return x


def add_background_zero_weight(
    image: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add weight tensor with zero weight for background.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (batch, channels, height, width).
    labels : torch.Tensor
        Label tensor with shape (batch, height, width).

    Returns
    -------
    image : torch.Tensor
        Input image tensor (unchanged).
    labels : torch.Tensor
        Label tensor (unchanged).
    sample_weights : torch.Tensor
        Tensor with zero weight for background pixels, shape (batch, height, width).
    """
    label_mask = labels > 0
    sample_weights = torch.where(
        label_mask,
        torch.ones_like(labels, dtype=torch.float32),
        torch.zeros_like(labels, dtype=torch.float32),
    )

    return image, labels, sample_weights


def unet_classify_single_image(
    unet_model: UNet, image: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """
    Classify single image using UNet.

    Parameters
    ----------
    unet_model : UNet
        Trained UNet model (PyTorch).
    image : np.ndarray
        Single image with shape (height, width, channels).
    device : str, optional
        Device to run inference on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    np.ndarray
        2D array with integer class labels, shape (height, width).
    """
    unet_model.to(device)
    unet_model.eval()

    # Convert from (H, W, C) to (1, C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = unet_model(image_tensor)
        # logits shape: (1, output_channels, height, width)
        labels = torch.argmax(logits, dim=1).squeeze(0)
        # labels shape: (height, width)

    return labels.cpu().numpy()


def unet_classify_image_batch(
    unet_model: UNet, batch: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """
    Classify image batch using UNet.

    Parameters
    ----------
    unet_model : UNet
        Trained UNet model (PyTorch).
    batch : np.ndarray
        Batch of images with shape (batch_size, height, width, channels).
    device : str, optional
        Device to run inference on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    np.ndarray
        3D array with integer class labels, shape (batch_size, height, width).
    """
    unet_model.to(device)
    unet_model.eval()

    # Convert from (B, H, W, C) to (B, C, H, W)
    batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)

    with torch.no_grad():
        logits = unet_model(batch_tensor)
        # logits shape: (batch_size, output_channels, height, width)
        labels = torch.argmax(logits, dim=1)
        # labels shape: (batch_size, height, width)

    return labels.cpu().numpy()
