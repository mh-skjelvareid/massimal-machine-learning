import numpy as np
import numpy.random


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


def kfold_generator(dataset, k):
    """Generator for K-fold splitting into training and validation datasets

    # Arguments:
    dataset    Tensorflow dataset
    k          Number of folds (see https://scikit-learn.org/stable/modules/cross_validation.html)

    # Returns
    training_dataset      Tensorflow dataset
    validation_dataset    Tensorflow dataset

    # Notes:
    The generator returns k sets of training and validation datasets when iterated over.

    # Example use:
    dataset = tf.data.Dataset.from_tensor_slices((np.arange(9),np.arange(9)%3))
    for data,label in dataset.as_numpy_iterator():
        print(f'Data: {data}, label: {label}')
    for training_dataset, validation_dataset in kfold_generator(dataset,3):
        print('----')
        for data,label in training_dataset.as_numpy_iterator():
            print(f'Training data: {data}, label: {label}')
        for data,label in validation_dataset.as_numpy_iterator():
            print(f'Validation data: {data}, label: {label}')
    """
    n_datapoints = dataset.cardinality()
    dataset = dataset.shuffle(n_datapoints, reshuffle_each_iteration=False)
    samples_per_fold = n_datapoints // k
    for i in range(k):
        validation_dataset = dataset.skip(i * samples_per_fold).take(samples_per_fold)
        # Merge parts before/after validation dataset to create training dataset
        training_dataset = dataset.take(i * samples_per_fold)
        training_dataset = training_dataset.concatenate(
            dataset.skip((i + 1) * samples_per_fold)
        )
        yield (training_dataset, validation_dataset)


def sample_weights_balanced(y):
    """Create sample weigths which are inversely proportional to class frequencies

    # Arguments:
    y        Numpy vector with (numerical) labels

    # Returns:
    sample_weights  Numpy vector with same shape as y
                    Classes with a low number of samples get higher weights
                    See 'balanced' option in
                    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html

    # Notes
    - Useful in combination with score() function for various classifiers,
    to calculate a balanced score in case on unbalanced datasets
    """
    sample_weights = np.zeros(len(y), dtype=float)
    for label in np.unique(y):
        label_mask = y == label
        sample_weights[label_mask] = len(y) / np.count_nonzero(label_mask)
    return sample_weights


def random_sample_image(image, frac=0.05, ignore_zeros=True, replace=False):
    """Draw random samples from image

    # Usage:
    samp = random_sample_image(image,...)

    # Required arguments:
    image:  3D numpy array with hyperspectral image, wavelengths along
            third axis (axis=2)

    # Optional arguments:
    frac:           Number of samples expressed as a fraction of the total
                    number of samples in the image. Range: [0 - 1]
    ignore_zeros:   Do not include samples that are equal to zeros across all
                    bands.
    replace:        Whether to select samples with or without replacement.

    # returns
    samp:   2D numpy array of size NxB, with N denoting number of samples and B
            denoting number of bands.
    """

    # Create mask
    if ignore_zeros:
        mask = ~np.all(image == 0, axis=2)
    else:
        mask = np.ones(image.shape[:-1], axis=2)

    # Calculate number of samples
    n_samp = np.int64(frac * np.count_nonzero(mask))

    # Create random number generator
    rng = numpy.random.default_rng()
    samp = rng.choice(image[mask], size=n_samp, axis=0, replace=replace)

    return samp
