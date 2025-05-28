import sys

import torch
import kornia
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
from evaluator import KeypointModel

class LoFTRRunner(KeypointModel):
    def __init__(self, device='cuda', pretrained='indoor'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = kornia.feature.LoFTR(pretrained=pretrained).eval().to(self.device)

    def find_matching_points(self, img1, img2):
        return self.match_keypoints_loftr(img1, img2)

    def match_keypoints_loftr(self, img0, img1):
        input_dict = {"image0": img0, "image1": img1}
        with torch.no_grad():
            out = self.model(input_dict)
        return out['keypoints0'].cpu().numpy(), out['keypoints1'].cpu().numpy()

    def visualize_matches(self, img0, img1, matched_keypoints, matched_targets, gt_targets, names):
        img0_np = img0.squeeze().cpu().numpy()
        img1_np = img1.squeeze().cpu().numpy()
        concat_img = np.hstack([img0_np, img1_np])
        concat_img = cv2.cvtColor((concat_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        offset = img0_np.shape[1]
        for pt0, pt1, gt, name in zip(matched_keypoints, matched_targets, gt_targets, names):
            pt1_shifted = (int(pt1[0] + offset), int(pt1[1]))
            gt_shifted = (int(gt[0] + offset), int(gt[1]))

            cv2.line(concat_img, (int(pt0[0]), int(pt0[1])), pt1_shifted, (0, 0, 255), 1)
            cv2.circle(concat_img, (int(pt0[0]), int(pt0[1])), 4, (255, 0, 0), -1)
            cv2.circle(concat_img, pt1_shifted, 4, (0, 255, 0), -1)
            cv2.circle(concat_img, gt_shifted, 4, (0, 255, 255), -1)
            cv2.putText(concat_img, name, (int(pt0[0]), int(pt0[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        plt.figure(figsize=(14, 6))
        plt.imshow(concat_img[..., ::-1])
        plt.axis('off')
        plt.title("Modrý = referenčný, Zelený = nájdený, Žltý = skutočný")
        plt.show()

    def run_batch(self, batch, output_path=None):
        img0 = batch["image0"].to(self.device)
        img1 = batch["image1"].to(self.device)
        kp0 = batch["keypoints0"][0].cpu().numpy()
        kp1 = batch["keypoints1"][0].cpu().numpy()

        mkpts0, mkpts1 = self.match_keypoints_loftr(img0, img1)

        matched_ref_points = []
        matched_target_points = []
        gt_coords_list = []
        matched_names = []

        for idx, (x0, y0) in enumerate(kp0):
            ref_coords = np.array([x0, y0])
            if len(mkpts0) == 0:
                print(f"kp_{idx} - Nenájdené | Skutočné: {kp1[idx]} | Chyba: {999999999999} px")
                continue
            dists = np.linalg.norm(mkpts0 - ref_coords, axis=1)
            min_idx = np.argmin(dists)

            matched_coords = mkpts1[min_idx]
            gt_coords = kp1[idx]
            error = np.linalg.norm(matched_coords - gt_coords)

            matched_ref_points.append(ref_coords)
            matched_target_points.append(matched_coords)
            gt_coords_list.append(gt_coords)
            matched_names.append(f"kp_{idx}")
            print(f"kp_{idx} - Nájdené: {matched_coords} | Skutočné: {gt_coords} | Chyba: {error:.2f} px")

        self.visualize_matches(img0, img1, matched_ref_points, matched_target_points, gt_coords_list, matched_names)

        if output_path:
            self.save_keypoints(output_path, matched_names, matched_target_points)

    def save_keypoints(self, filepath, names, coords):
        keypoints = [{"name": name, "x": float(x), "y": float(y)} for name, (x, y) in zip(names, coords)]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({"keypoints": keypoints}, f, indent=4)
