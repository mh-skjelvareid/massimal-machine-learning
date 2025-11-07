import numpy as np


def save_pca_model(
    pca_model, X_notscaled, npz_filename, n_components="all", feature_labels=None
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


def read_pca_model(
    npz_filename, include_explained_variance=False, include_feature_labels=False
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


def pca_transform_image(image, W_pca, X_mean, X_std=None):
    """Apply PCA transform to 3D image cube

    # Arguments:
    image       NumPy array with 3 dimensions (n_rows,n_cols,n_channels)
    W_pca       PCA weight matrix with 2 dimensions (n_components, n_channels)
    X_mean      Mean value vector, to be subtracted from data ("centering")
                Length (n_channels,)

    # Keyword arguments:
    X_std       Standard deviation vector, to be used for scaling (z score)
                If None, no scaling is performed
                Length (n_channels)

    # Returns:
    image_pca   Numpy array with dimension (n_rows, n_cols, n_channels)

    # Notes:
    - Input pixels which are zero across all channels are set to zero in the
    output PCA image as well.

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
    im_pca = (
        np.reshape(im_vec_pca, image.shape[0:2] + (im_vec_pca.shape[-1],))
        * nonzero_mask
    )

    return im_pca
