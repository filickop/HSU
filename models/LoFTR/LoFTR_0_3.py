import torch
import kornia.feature as KF
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def load_image(path, size=(200, 200)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    tens = torch.from_numpy(img / 255.).float()[None, None]
    return tens.cuda() if torch.cuda.is_available() else tens


def load_keypoints(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [(kp["name"], kp["x"], kp["y"]) for kp in data["keypoints"]]


def match_keypoints_loftr(img0, img1):
    matcher = KF.LoFTR(pretrained='outdoor').eval()
    if torch.cuda.is_available():
        matcher = matcher.cuda()
    with torch.no_grad():
        correspondences = matcher({"image0": img0, "image1": img1})
    return correspondences['keypoints0'].cpu().numpy(), correspondences['keypoints1'].cpu().numpy()


def find_closest_match(src_kp, mkpts0, mkpts1):
    name, x, y = src_kp
    dists = np.linalg.norm(mkpts0 - np.array([x, y]), axis=1)
    if len(dists) == 0 or np.min(dists) > 5:  # voliteľný threshold
        return name, (x, y), None, None, None
    min_idx = np.argmin(dists)
    matched = mkpts1[min_idx]
    dist = np.linalg.norm(np.array([x, y]) - mkpts0[min_idx])
    return name, (x, y), matched, mkpts0[min_idx], dist


def visualize_matches(img0, img1, matched_keypoints, matched_targets, names):
    img0_np = img0.squeeze().cpu().numpy()
    img1_np = img1.squeeze().cpu().numpy()
    concat_img = np.hstack([img0_np, img1_np])
    concat_img = cv2.cvtColor((concat_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    offset = img0_np.shape[1]
    for pt0, pt1, name in zip(matched_keypoints, matched_targets, names):
        pt1_shifted = (int(pt1[0] + offset), int(pt1[1]))
        cv2.line(concat_img, (int(pt0[0]), int(pt0[1])), pt1_shifted, (0, 0, 255), 1)
        cv2.circle(concat_img, (int(pt0[0]), int(pt0[1])), 4, (255, 0, 0), -1)
        cv2.circle(concat_img, pt1_shifted, 4, (0, 255, 0), -1)
        cv2.putText(concat_img, name, (int(pt0[0]), int(pt0[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    plt.figure(figsize=(14, 6))
    plt.imshow(concat_img[..., ::-1])
    plt.axis('off')
    plt.title("Vizualizácia zodpovedajúcich bodov (referenčný → cieľový)")
    plt.show()


# === Cesty k súborom ===
img0_path = "../../dataset/Note/images/note_000.png"
kp0_path = "dataset/Note/keypoints/note_000.json"

img1_path = "../../dataset/Note/images/note_099.png"
kp1_path = "dataset/Note/keypoints/note_099.json"

# === Načítanie dát ===
img0 = load_image(img0_path)
img1 = load_image(img1_path)
keypoints0 = load_keypoints(kp0_path)
keypoints1_dict = {kp[0]: (kp[1], kp[2]) for kp in load_keypoints(kp1_path)}

# === LoFTR matching ===
mkpts0, mkpts1 = match_keypoints_loftr(img0, img1)

# === Porovnanie pre každý bod z obrázka 0 ===
matched_ref_points = []
matched_target_points = []
matched_names = []

print("=== Výsledky matching-u ===")
for kp in keypoints0:
    name, ref_coords, matched_coords, _, dist = find_closest_match(kp, mkpts0, mkpts1)
    if matched_coords is None:
        print(f"{name}: Nenájdené")
        continue

    gt_coords = keypoints1_dict.get(name)
    if gt_coords is None:
        print(f"{name}: Chýbajúca anotácia v cieľovom obrázku.")
        continue

    val_dist = np.linalg.norm(np.array(gt_coords) - matched_coords)

    print(f"{name} - Nájdené: ({matched_coords[0]:.1f}, {matched_coords[1]:.1f}) | "
          f"Skutočné: ({gt_coords[0]}, {gt_coords[1]}) | "
          f"Vzdialenosť: {val_dist:.2f} px")

    matched_ref_points.append(ref_coords)
    matched_target_points.append(matched_coords)
    matched_names.append(name)

# === Vizualizácia ===
if matched_ref_points:
    visualize_matches(img0, img1, matched_ref_points, matched_target_points, matched_names)
else:
    print("Žiadne body na vizualizáciu.")
