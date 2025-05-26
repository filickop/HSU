from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2
import time

BINS = np.array([0.0, 1.0, 5.0, 10.0, 100.0])

class KeypointModel(ABC):
    @abstractmethod
    def find_matching_points(self, img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
        """Takes 2 images (as returned by cv2.imread) and finds matching points between them.
        Returns 2 numpy arrays containing the coordinates of the matching points."""


@dataclass
class TestResults:
    nearest_metric_histogram: np.ndarray
    weighted_metric_histogram: np.ndarray
    transform_metric_histogram: np.ndarray


def nearest_neighbor_metric(knn: NearestNeighbors,
                            corr1: np.ndarray, # (N, 2)
                            corr2: np.ndarray, # (N, 2)
                            points_gt1: np.ndarray, # (K, 2)
                            points_gt2: np.ndarray # (K, 2)
                            ) -> np.ndarray: # (K)
    neighbors_idx = knn.kneighbors(points_gt1, n_neighbors=1, return_distance=False)[:, 0]
    points2 = corr2[neighbors_idx]
    return np.linalg.norm(points2 - points_gt2, axis=1)

def weighted_neighbor_metric(knn: NearestNeighbors,
                             corr1: np.ndarray, # (N, 2)
                             corr2: np.ndarray, # (N, 2)
                             points_gt1: np.ndarray, # (K, 2)
                             points_gt2: np.ndarray # (K, 2)
                             ) -> np.ndarray: # (K)
    distances, neighbors_idx = knn.kneighbors(points_gt1, n_neighbors=3)

    weights = 1 / (distances + 1e-8)
    weights /= weights.sum(axis=1)[:, None] # (K, 3)

    # points2 = (corr2[neighbors_idx] * weights[:, :, None]).sum(axis=1) # (K, 3, 2)
    points2 = np.matvec(corr2[neighbors_idx].transpose(0, 2, 1), weights)

    return np.linalg.norm(points2 - points_gt2, axis=1)

def transform_prediction_metric(corr1: np.ndarray, # (N, 2)
                                corr2: np.ndarray, # (N, 2)
                                points_gt1: np.ndarray, # (K, 2)
                                points_gt2: np.ndarray # (K, 2)
                                ) -> np.ndarray: # (K)
    transform, _ = cv2.estimateAffine2D(corr1, corr2, cv2.RANSAC)
    points_gt1_ones = np.hstack((points_gt1, np.ones((len(points_gt1), 1))))
    points2 = points_gt1_ones @ transform.T

    return np.linalg.norm(points2 - points_gt2, axis=1)


def run_tests(model: KeypointModel, dataset: Dataset, *, bins: np.ndarray = BINS) -> TestResults:
    results = TestResults(
        np.zeros_like(bins, dtype=int),
        np.zeros_like(bins, dtype=int),
        np.zeros_like(bins, dtype=int),
    )

    bins = np.append(bins, np.inf) # Last bin should contain all values out of range
    dataset_size = len(dataset)

    start = time.time()
    total_match_duration = 0.0
    for i, data in enumerate(dataset):
        if i == dataset_size: break
        if i % 10 == 0:
            print(f"{i+1:5}/{dataset_size}" , end="")
        print(".", end="", flush=True)
        match_start = time.time()
        corr1, corr2 = model.find_matching_points(data["image1"], data["image2"])
        total_match_duration += time.time() - match_start
        points_gt1, points_gt2 = data["points1"], data["points2"]
        knn = NearestNeighbors().fit(corr1, corr2)

        distances_nearest = nearest_neighbor_metric(knn, corr1, corr2, points_gt1, points_gt2)
        results.nearest_metric_histogram += np.histogram(distances_nearest, bins)[0]

        distances_weighted = weighted_neighbor_metric(knn, corr1, corr2, points_gt1, points_gt2)
        results.weighted_metric_histogram += np.histogram(distances_weighted, bins)[0]

        distances_transform = transform_prediction_metric(corr1, corr2, points_gt1, points_gt2)
        results.transform_metric_histogram += np.histogram(distances_transform, bins)[0]

        if i % 10 == 9:
            duration = time.time() - start
            print(f" (total {duration / 10:.2f} s/test, matching {total_match_duration / 10:.2f} s/test)")
            total_match_duration = 0
            start = time.time()
    return results
