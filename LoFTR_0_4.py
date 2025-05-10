import torch
import kornia
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# === Načítanie obrázka ===
def load_image(path, size=(200, 200)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = torch.from_numpy(img).float()[None, None] / 255.0
    return img.cuda()

# === Načítanie keypointov z JSON ===
def load_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(kp['name'], kp['x'], kp['y']) for kp in data['keypoints']]

# === Vizualizácia ===
def visualize_matches(img0, img1, matched_keypoints, matched_targets, gt_targets, names):
    img0_np = img0.squeeze().cpu().numpy()
    img1_np = img1.squeeze().cpu().numpy()
    concat_img = np.hstack([img0_np, img1_np])
    concat_img = cv2.cvtColor((concat_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    offset = img0_np.shape[1]
    for pt0, pt1, gt, name in zip(matched_keypoints, matched_targets, gt_targets, names):
        pt1_shifted = (int(pt1[0] + offset), int(pt1[1]))
        gt_shifted = (int(gt[0] + offset), int(gt[1]))

        cv2.line(concat_img, (int(pt0[0]), int(pt0[1])), pt1_shifted, (0, 0, 255), 1)
        cv2.circle(concat_img, (int(pt0[0]), int(pt0[1])), 4, (255, 0, 0), -1)  # modrý: referenčný
        cv2.circle(concat_img, pt1_shifted, 4, (0, 255, 0), -1)                 # zelený: nájdený
        cv2.circle(concat_img, gt_shifted, 4, (0, 255, 255), -1)               # žltý: skutočný
        cv2.putText(concat_img, name, (int(pt0[0]), int(pt0[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    plt.figure(figsize=(14, 6))
    plt.imshow(concat_img[..., ::-1])
    plt.axis('off')
    plt.title("Modrý = referenčný, Zelený = nájdený, Žltý = skutočný")
    plt.show()

# === Matching s LoFTR ===
def match_keypoints_loftr(img0, img1):
    matcher = kornia.feature.LoFTR(pretrained='outdoor').eval().cuda()
    input_dict = {"image0": img0, "image1": img1}
    with torch.no_grad():
        correspondences = matcher(input_dict)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    return mkpts0, mkpts1

# === ukladanie keypointov ===
def save_found_keypoints(filepath, names, coords):
    keypoints = []
    for name, (x, y) in zip(names, coords):
        keypoints.append({
            "name": name,
            "x": float(x),
            "y": float(y)
        })
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump({"keypoints": keypoints}, f, indent=4)


# === Hlavná časť ===
img0_path = "dataset/images/note_000.png"
kp0_path = "dataset/keypoints/note_000.json"
img1_path = "dataset/images/note_006.png"
kp1_path = "dataset/keypoints/note_006.json"
kp1_json_path = "foundKeypoints/note_006.json"

img0 = load_image(img0_path)
img1 = load_image(img1_path)

keypoints0 = load_keypoints(kp0_path)
keypoints1_dict = {kp[0]: (kp[1], kp[2]) for kp in load_keypoints(kp1_path)}

mkpts0, mkpts1 = match_keypoints_loftr(img0, img1)

matched_ref_points = []
matched_target_points = []
gt_coords_list = []
matched_names = []

for name, x0, y0 in keypoints0:
    ref_coords = np.array([x0, y0])
    if len(mkpts0) == 0:
        continue
    dists = np.linalg.norm(mkpts0 - ref_coords, axis=1)
    idx = np.argmin(dists)

    matched_coords = mkpts1[idx]
    matched_ref_coords = mkpts0[idx]

    if name in keypoints1_dict:
        gt_coords = np.array(keypoints1_dict[name])
        error = np.linalg.norm(matched_coords - gt_coords)
        print(f"{name} - Nájdené: ({matched_coords[0]}, {matched_coords[1]}) | "
              f"Skutočné: ({gt_coords[0]}, {gt_coords[1]}) | Vzdialenosť: {error:.2f} px")

        matched_ref_points.append(ref_coords)
        matched_target_points.append(matched_coords)
        gt_coords_list.append(gt_coords)
        matched_names.append(name)

visualize_matches(img0, img1, matched_ref_points, matched_target_points, gt_coords_list, matched_names)
output_json_path = kp1_json_path
save_found_keypoints(output_json_path, matched_names, matched_target_points)