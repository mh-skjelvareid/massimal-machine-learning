from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpectraDataset:
    """Data class to hold spectra data and associated class and group labels.

    Attributes
    ----------
    X : ndarray
        2-D array of shape (n_samples, n_wavelengths) containing spectral data for each sample.
    cls : ndarray
        1-D array of shape (n_samples,) containing integer class labels for each sample.
    grp : ndarray
        1-D array of shape (n_samples,) containing integer group/location labels for each sample
    """

    X: np.ndarray  # (n_samples, n_wavelengths)
    cls: np.ndarray  # (n_samples,)
    grp: np.ndarray  # (n_samples,)

    def __post_init__(self):
        assert self.X.shape[0] == self.cls.shape[0] == self.grp.shape[0], (
            "X, cls, and grp must have the same number of samples (rows)."
        )

    def subset(self, mask):
        return SpectraDataset(self.X[mask], self.cls[mask], self.grp[mask])

    def sample_count_matrix(self):
        n_classes = np.max(self.cls) + 1
        n_groups = np.max(self.grp) + 1

        sample_count_matrix = np.zeros((n_groups, n_classes), dtype=np.uint64)

        for grp_ind in range(n_groups):
            for class_ind in range(n_classes):
                mask = (self.cls == class_ind) & (self.grp == grp_ind)
                sample_count_matrix[grp_ind, class_ind] = np.sum(mask)

        return sample_count_matrix


class DatasetBalancer:
    """Class to balance a SpectraDataset by undersampling classes and groups."""

    def __init__(self, class_gamma=0.8, group_gamma=0.9):
        """Initialize DatasetBalancer with gamma parameters for class and group balancing."""
        self.class_gamma = class_gamma
        self.group_gamma = group_gamma

    def _balance_samples_per_group(self, sample_count_matrix) -> NDArray:
        """Apply gamma weighting to each group's class counts"""
        balanced_sample_count_matrix = np.zeros_like(sample_count_matrix)

        for group_int, class_counts in enumerate(sample_count_matrix):
            balanced_sample_count_matrix[group_int] = gamma_weight_vector(
                class_counts, gamma=self.class_gamma
            ).astype(int)

        return balanced_sample_count_matrix

    def _balance_samples_between_groups(self, sample_count_matrix) -> NDArray:
        """Apply gamma weighting to total samples per group to balance between groups."""
        samples_per_group = sample_count_matrix.sum(axis=1)
        assert np.all(samples_per_group >= 0)

        balanced_samples_per_group = gamma_weight_vector(
            samples_per_group, gamma=self.group_gamma
        ).astype(int)
        balanced_sample_matrix = sample_count_matrix * (
            balanced_samples_per_group[:, None] / samples_per_group[:, None]
        )
        return balanced_sample_matrix.astype(int)

    def balance_sample_count_matrix(self, sample_count_matrix: NDArray) -> NDArray:
        """Balance classes per group and then balance samples between groups."""
        balanced_per_group = self._balance_samples_per_group(sample_count_matrix)
        balanced_between_groups = self._balance_samples_between_groups(balanced_per_group)
        return balanced_between_groups

    def _undersample_dataset_to_match_sample_count_matrix(
        self, dataset: SpectraDataset, sample_count_matrix: NDArray
    ) -> SpectraDataset:
        """Undersample dataset to match target sample counts per group and class."""
        balance_sample_count_matrix = self.balance_sample_count_matrix(sample_count_matrix)
        n_groups, n_classes = sample_count_matrix.shape
        indices_to_keep = []

        for group_int in range(n_groups):
            for class_int in range(n_classes):
                mask = (dataset.grp == group_int) & (dataset.cls == class_int)
                sample_indices = np.where(mask)[0]
                n_samples_to_keep = balance_sample_count_matrix[group_int, class_int]
                sampled_indices = np.random.choice(
                    sample_indices, size=n_samples_to_keep, replace=False
                )
                indices_to_keep.extend(sampled_indices)
        indices_to_keep = np.array(indices_to_keep)

        return dataset.subset(indices_to_keep)

    def balance_dataset(self, dataset: SpectraDataset) -> SpectraDataset:
        """Balance dataset by undersampling classes and groups using gamma weighting."""
        sample_count_matrix = dataset.sample_count_matrix()
        balanced_sample_count_matrix = self.balance_sample_count_matrix(sample_count_matrix)
        balanced_dataset = self._undersample_dataset_to_match_sample_count_matrix(
            dataset, balanced_sample_count_matrix
        )
        return balanced_dataset


def gamma_weight_vector(vec: NDArray, gamma: float = 0.8) -> NDArray:
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
