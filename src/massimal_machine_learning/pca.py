import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def save_pca_model_numpy(
    pca_model: PCA,
    X_notscaled: NDArray,
    npz_filename: Path | str,
    n_components: int | Literal["all"] = "all",
    feature_labels=None,
):
    """Save PCA weights and X mean and std as NumPy npz file

    # Arguments:
    pca_model       sklearn.decomposition.PCA model which has beed fitted to (scaled) data X_scaled
    X_notscaled     X array before scaling - used to calculate "original" mean / std
    npz_filename    Path to *.npz file where data will be saved

    # Keyword arguments
    n_components    Number of PCA components to include
    feature_labels  Array with labels for each feature (each column of X)

    # Notes:
    The function will save the following arrays to the npz. file:
        - W_pca:  PCA "weights", shape (N_components, N_features)
        - X_mean: X mean values, shape (N_features,)
        - X_std:  X standard deviations, shape (N_features,)
        - explained_variance_ratio, shape (N_components,)
        - feature_labels (optional), shape (N_features,)
    """
    # If n_components specified, only use n first components
    if n_components != "all":
        W_pca = pca_model.components_[0:n_components, :]
        explained_variance_ratio = pca_model.explained_variance_ratio_[0:n_components]
    else:
        W_pca = pca_model.components_
        explained_variance_ratio = pca_model.explained_variance_ratio_

    # Save as npz file
    np.savez(
        npz_filename,
        W_pca=W_pca,
        X_mean=np.mean(X_notscaled, axis=0),
        X_std=np.std(X_notscaled, axis=0),
        explained_variance_ratio=explained_variance_ratio,
        feature_labels=feature_labels,
    )


def read_pca_model_numpy(
    npz_filename: Path | str,
    include_explained_variance: bool = False,
    include_feature_labels: bool = False,
):
    """Load PCA weights and X mean and std from NumPy npz file

    # Arguments:
    npz_filename    Path to *.npz file where data is saved

    # Keyword arguments:
    include_explained_variance - whether to include explained variance
    include_feature_labels - whether to include feature labels

    # Returns:
    W_pca:    PCA "weights", shape (N_components, N_features)
    X_mean:   X mean values, shape (1,N_features,)
    X_std:    X standard deviations, shape (1,N_features)
    explained_variance_ratio (optional), shape (N_components,)
    feature_labels (optional), shape (N_features,)
    """
    return_list = []
    with np.load(npz_filename) as npz_files:
        return_list.append(npz_files["W_pca"])
        return_list.append(npz_files["X_mean"])
        return_list.append(npz_files["X_std"])
        if include_explained_variance:
            return_list.append(npz_files["explained_variance_ratio"])
        if include_feature_labels:
            return_list.append(npz_files["feature_labels"])

    return tuple(return_list)


def save_pca_model_json(pca: PCA, mean: NDArray, std: NDArray, output_path: Path | str):
    """Save trained PCA model with z-score parameters as JSON file

    Parameters
    ----------
    pca : PCA
        Trained slkearn.decomposition.PCA model
    mean : NDArray
        Mean values used for z-score normalization
    std : NDArray
        Standard deviation values used for z-score normalization
    output_path : Path | str
        Path to output JSON file
    """
    pca_model_data = {
        "pca_components": pca.components_.tolist(),
        "pca_explained_variance": pca.explained_variance_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(pca_model_data, f)


def read_pca_model_json(
    pca_json_path: Path | str, dtype: np.dtype = np.float32
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Load trained PCA model with z-score parameters from JSON file

    Parameters
    ----------
    pca_json_path : Path | str
        Path to JSON file with saved PCA and z-score parameters
    dtype: Numpy data type
        Type to cast returned arrays as. Default np.float32.

    Returns
    -------
    pca_components : NDArray
        PCA components, shape (n_components, n_channels)
    pca_explained_variance : NDArray
        Explained variance per component
    pca_explained_variance_ratio : NDArray
        Explained variance per component relative to total variance
    mean : NDArray
        Mean values used for z-score normalization
    std : NDArray
        Standard deviation values used for z-score normalization
    """
    with open(pca_json_path, "r") as f:
        pca_model_data = json.load(f)
    pca_components = np.array(pca_model_data["pca_components"], dtype=dtype)
    pca_explained_variance = np.array(pca_model_data["pca_explained_variance"], dtype=dtype)
    pca_explained_variance_ratio = np.array(
        pca_model_data["pca_explained_variance_ratio"], dtype=dtype
    )
    mean = np.array(pca_model_data["mean"], dtype=dtype)
    std = np.array(pca_model_data["std"], dtype=dtype)
    return (
        pca_components,
        pca_explained_variance,
        pca_explained_variance_ratio,
        mean,
        std,
    )


def pca_transform_image(
    image: NDArray, W_pca: NDArray, X_mean: NDArray, X_std: NDArray | None = None
) -> NDArray:
    """Apply PCA transform to 3D image cube

    Parameters
    ----------
    image : NDArray
        NumPy array with shape (n_rows,n_cols,n_channels)
    W_pca : NDArray
        PCA weight matrix with shape (n_components, n_channels)
    X_mean : NDArray
        Mean value vector for mean centering, shape (n_channels,)
    X_std : NDArray | None, optional
        Standard deviation vector for normalization (scaling), shape (n_channels,)
        If None, no scaling is performed

    Returns
    -------
    NDArray
        PCA transformed image, shape (n_rows,n_cols,n_components)
    """

    # Create mask for nonzero values
    nonzero_mask = ~np.all(image == 0, axis=2, keepdims=True)

    # Vectorize image
    im_vec = np.reshape(image, (-1, image.shape[-1]))

    # Subtract mean (always) and scale (optional)
    im_vec_norm = im_vec - X_mean
    if X_std is not None:
        im_vec_norm = im_vec_norm / X_std

    # PCA transform through matrix multiplication (projection to rotated coordinate system)
    im_vec_pca = im_vec_norm @ W_pca.T

    # Reshape into image, and ensure that zero-value input pixels are also zero in output
    im_pca = np.reshape(im_vec_pca, image.shape[0:2] + (im_vec_pca.shape[-1],)) * nonzero_mask

    return im_pca
