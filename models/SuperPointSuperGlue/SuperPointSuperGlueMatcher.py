import torch
import numpy as np
from evaluator import KeypointModel
from models.SuperPointSuperGlue.matching import Matching


class SuperPointSuperGlueMatcher(KeypointModel):
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matcher = Matching(config).eval().to(self.device)

    def find_matching_points(self, img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
        if img1 is None:
            raise RuntimeError(f"Obrázok {img1} sa nenačítal!")
        if img2 is None:
            raise RuntimeError(f"Obrázok {img2} sa nenačítal!")

        with torch.no_grad():
            pred = self.matcher({'image0': img1, 'image1': img2})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0_all = pred['keypoints0']
        kpts1_all = pred['keypoints1']
        matches = pred['matches0']

        valid = matches != -1
        matched_kpts0 = kpts0_all[valid]
        matched_kpts1 = kpts1_all[matches[valid]]

        return matched_kpts0, matched_kpts1