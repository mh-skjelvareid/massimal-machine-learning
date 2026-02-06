import numpy as np
import numpy.random
from numpy.typing import NDArray


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


def create_sample_count_matrix(X, class_col_ind, group_col_ind):
    """Create a sample count matrix from data matrix X.

    Parameters
    ----------
    X : ndarray
        2-D array with samples on rows. Must contain integer labels for class and
        group in the columns specified by class_col_ind and group_col_ind.
    class_col_ind : int
        Column index in X containing zero-indexed class labels (0 .. n_classes-1).
    group_col_ind : int
        Column index in X containing zero-indexed group/location labels
        (0 .. n_groups-1).

    Returns
    -------
    sample_count_matrix : ndarray of int, shape (n_groups, n_classes)
        Matrix where entry (i, j) is the number of samples that belong to group i
        and class j.

    Notes
    -----
    n_classes and n_groups are inferred from the maximum label values in the
    respective columns of X. The function asserts that the minimum label in each
    column is 0 and that there is at least two classes and two groups.
    """
    n_classes = np.max(X[:, class_col_ind]).astype(int) + 1
    n_groups = np.max(X[:, group_col_ind]).astype(int) + 1

    assert min(X[:, class_col_ind]) == 0 and n_classes >= 2
    assert min(X[:, group_col_ind]) == 0 and n_groups >= 2

    sample_count_matrix = np.zeros((n_groups, n_classes), dtype=int)

    for group_int in range(n_groups):
        for class_int in range(n_classes):
            sample_count_matrix[group_int, class_int] = np.sum(
                (X[:, group_col_ind] == group_int) & (X[:, class_col_ind] == class_int)
            )

    return sample_count_matrix


def gamma_weight_vector(vec: NDArray, gamma: float = 0.8):
    """Apply gamma weighting (softly reduce range of values) to input vector.

    Parameters
    ----------
    vec : ndarray
        1-D array of non-negative values to be gamma-weighted.
    gamma : float, optional
        Exponent for gamma weighting. Must be in range [0, 1].

    Returns
    -------
    vec_weighted : ndarray
        Gamma-weighted version of input vector.

    """

    assert np.all(vec >= 0), "Input vector elements must be non-negative."
    assert (gamma >= 0) & (gamma <= 1), "Gamma must be in range [0, 1]."

    nonzero_ind = vec > 0
    nonzero_min = np.min(vec[nonzero_ind])

    vec_weighted = np.zeros_like(vec)
    vec_weighted[nonzero_ind] = nonzero_min + (vec[nonzero_ind] - nonzero_min) ** gamma

    return vec_weighted


def balance_classes_per_group(sample_count_matrix, class_gamma=0.8):
    """Balance classes per group using gamma weighting.

    Parameters
    ----------
    sample_count_matrix : ndarray of int, shape (n_groups, n_classes)
        Matrix where entry (i, j) is the number of samples that belong to group i
        and class j.

    Returns
    -------
    balanced_sample_count_matrix : ndarray of int, shape (n_groups, n_classes)
        Matrix with same shape as input where classes have been balanced per group
        using gamma weighting.
    """
    balanced_sample_count_matrix = np.zeros_like(sample_count_matrix)

    # Apply gamma weighting to each group's class counts
    for group_int, class_counts in enumerate(sample_count_matrix):
        balanced_sample_count_matrix[group_int] = gamma_weight_vector(
            class_counts, gamma=class_gamma
        ).astype(int)

    return balanced_sample_count_matrix


def balance_samples_between_groups(sample_count_matrix, group_gamma=0.9):
    """Balance samples between groups using gamma weighting.

    Parameters
    ----------
    sample_count_matrix : array_like, shape (n_groups, n_categories)
        Non-negative counts for each group (rows) and category/feature (columns).

    group_gamma : float, optional (default=0.9)
        Gamma parameter controlling the strength of the weighting applied when
        computing target sample counts.

    Returns
    -------
    balanced_sample_matrix : ndarray, shape (n_groups, n_categories)
        Integer array with the same shape as ``sample_count_matrix``.
    """
    samples_per_group = sample_count_matrix.sum(axis=1)
    assert np.all(samples_per_group >= 0)

    balanced_samples_per_group = gamma_weight_vector(
        samples_per_group, gamma=group_gamma
    ).astype(int)
    balanced_sample_matrix = sample_count_matrix * (
        balanced_samples_per_group[:, None] / samples_per_group[:, None]
    )
    return balanced_sample_matrix.astype(int)


def balance_samples(sample_count_matrix, class_gamma=0.8, group_gamma=0.9):
    """Balance samples per class and between groups using gamma weighting.

    Parameters
    ----------
    sample_count_matrix : ndarray of int, shape (n_groups, n_classes)
        Matrix where entry (i, j) is the number of samples that belong to group i
        and class j.
    class_gamma : float, optional
        Gamma parameter for balancing classes per group.
    group_gamma : float, optional
        Gamma parameter for balancing samples between groups.

    Returns
    -------
    balanced_sample_count_matrix : ndarray of int, shape (n_groups, n_classes)
        Matrix with same shape as input where classes have been balanced per group
        and samples have been balanced between groups using gamma weighting.

    """
    sample_matrix_balanced_per_group = balance_classes_per_group(
        sample_count_matrix, class_gamma=class_gamma
    )
    balanced_sample_count_matrix = balance_samples_between_groups(
        sample_matrix_balanced_per_group, group_gamma=group_gamma
    )

    return balanced_sample_count_matrix
