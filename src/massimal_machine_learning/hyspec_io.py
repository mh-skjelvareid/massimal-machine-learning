# Imports
from pathlib import Path

import numpy as np
import spectral

DEFAULT_WL_RGB = (640, 550, 460)  # Default wavelengths for RGB display


def load_envi_image(
    header_filename: Path | str,
    image_filename: Path | str | None = None,
    rgb_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict]:
    """
    Load image in ENVI format, with wavelengths and RGB indices.

    Parameters
    ----------
    header_filename : Path or str
        Path to ENVI file header.
    image_filename : Path or str or None, optional
        Path to ENVI data file, useful if data file is not found automatically. If None,
        the function will try to find the data file automatically.
    rgb_only : bool, optional
        If True, only RGB bands are loaded. Default is False.

    Returns
    -------
    image : np.ndarray
        Image (full cube as default, 3 RGB bands if rgb_only = True).
    wl : np.ndarray
        Wavelength vector.
    rgb_ind : tuple of int
        3-element tuple with indices to default RGB bands, [640, 550, 460] nm.
    metadata : dict
        Image metadata. Can be used as input to spectral.io.envi.save_image().
    """

    # Open image handle
    im_handle = spectral.io.envi.open(header_filename, image_filename)

    # Read wavelengths
    wl = np.array([float(i) for i in im_handle.metadata["wavelength"]])

    # Get indices for standard RGB render
    rgb_ind = tuple(int((np.abs(wl - value)).argmin()) for value in DEFAULT_WL_RGB)

    # Read data from disk
    if rgb_only:
        image = im_handle[
            :, :, rgb_ind
        ]  # Subscripting the image handle imports the requested data (RGB bands)
    else:
        image = np.array(
            im_handle.load()
        )  # Read full 3D cube, cast as numpy array, converting from spectral.image.ImageArray

    # Returns
    return (image, wl, rgb_ind, im_handle.metadata)


def save_envi_image(
    header_filename: Path | str,
    image: np.ndarray,
    metadata: dict,
    dtype: str | None = None,
    **kwargs,
) -> None:
    """
    Save ENVI file.

    Parameters
    ----------
    header_filename : Path or str
        Path to header file. Data file will be saved in the same location and with the
        same name, but without the '.hdr' extension.
    image : np.ndarray
        Numpy array with hyperspectral image.
    metadata : dict
        Dict containing (updated) image metadata. See load_envi_image().
    dtype : str or None, optional
        Data type for ENVI file. Follows numpy naming convention. Typically 'uint16' or
        'single' (32-bit float). If None, dtype = image.dtype.
    **kwargs
        Additional keyword arguments passed on to spectral.envi.save_image().
    """

    # Save file
    spectral.envi.save_image(
        header_filename,
        image,
        dtype=dtype,
        metadata=metadata,
        force=True,
        ext=None,
        **kwargs,
    )
