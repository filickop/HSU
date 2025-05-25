import torch
import kornia
import kornia.feature as KF
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json


# Načítanie obrázkov (v grayscale)
def load_and_preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))  # Zabezpečíme rovnakú veľkosť pre všetky obrázky
    tens = torch.from_numpy(img / 255.).float()[None, None]  # Normalizácia do [0, 1]
    return tens.cuda() if torch.cuda.is_available() else tens


# Načítanie keypointov zo súboru
def load_keypoints(path):
    with open(path, "r") as f:
        keypoints_data = json.load(f)
    # Pritom môžeme načítať aj názov bodu (napr. stick_top, stick_bottom, ...)
    keypoints = []
    for kp in keypoints_data["keypoints"]:
        keypoints.append((kp["name"], kp["x"], kp["y"]))  # Uložíme názov a súradnice
    return keypoints


# Inicializácia LoFTR
matcher = KF.LoFTR(pretrained='outdoor').eval()
if torch.cuda.is_available():
    matcher = matcher.cuda()

# Načítaj referenčný obrázok a jeho keypointy
img0_path = "../../dataset/Note/images/note_000.png"
keypoints0_path = "dataset/Note/keypoints/note_000.json"
img0 = load_and_preprocess(img0_path)
keypoints0 = load_keypoints(keypoints0_path)

# Načítaj cieľový obrázok (napr. z datasetu) a jeho skutočné keypointy
img1_path = "../../dataset/Note/images/note_099.png"
keypoints1_path = "dataset/Note/keypoints/note_099.json"
img1 = load_and_preprocess(img1_path)
keypoints1_true = load_keypoints(keypoints1_path)

# Matching s LoFTR
with torch.no_grad():
    input_dict = {
        "image0": img0,
        "image1": img1,
    }
    correspondences = matcher(input_dict)

# Získaj keypointy z LoFTR
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()

# Prehľadávame LoFTR výsledky a hľadáme presné zhodné body
matches = []
threshold_distance = 5  # Nastav vzdialenosť pre filtráciu zhodných bodov (pixely)
distances_list = []  # List pre vzdialenosti medzi keypointmi

for kp1_true in keypoints1_true:
    kp1_name, x_true, y_true = kp1_true  # Skutočný bod z JSON
    # Pre každý skutočný keypoint z druhého obrázka hľadáme najbližší keypoint z LoFTR
    distances = np.linalg.norm(mkpts1 - np.array([x_true, y_true]), axis=1)  # Vzdialenosť medzi keypointmi
    min_idx = np.argmin(distances)

    if distances[min_idx] < threshold_distance:
        # Uložíme bod, ktorý je zhodný
        matches.append((kp1_name, (x_true, y_true), mkpts1[min_idx], distances[min_idx]))
        distances_list.append(distances[min_idx])  # Uložíme vzdialenosť pre analýzu

# Výpis zhodných bodov a vzdialeností
print("Zhodné body a vzdialenosti:")
for match in matches:
    name, true_coords, matched_coords, dist = match
    true_x, true_y = true_coords
    matched_x, matched_y = matched_coords
    print(
        f"{name} - Skutočné: ({true_x}, {true_y}), Nájdené: ({matched_x:.2f}, {matched_y:.2f}), Vzdialenosť: {dist:.2f} pixely")


# Vizualizácia nájdených zodpovedajúcich bodov
def draw_matched_keypoints(img0, img1, keypoints0, keypoints1):
    img0_np = img0.squeeze().cpu().numpy()
    img1_np = img1.squeeze().cpu().numpy()
    img_comb = np.hstack([img0_np, img1_np])
    img_comb = cv2.cvtColor((img_comb * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for (x0, y0), (x1, y1) in zip(keypoints0, keypoints1):
        x1_shifted = x1 + img0_np.shape[1]  # Posun druhého obrázku v pravo
        color = (0, 0, 255)  # Červená
        cv2.line(img_comb, (int(x0), int(y0)), (int(x1_shifted), int(y1)), color, 1)
        cv2.circle(img_comb, (int(x0), int(y0)), 2, color, -1)
        cv2.circle(img_comb, (int(x1_shifted), int(y1)), 2, color, -1)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_comb)
    plt.axis('off')
    plt.title(f'LoFTR Matched Keypoints: {len(keypoints0)}')
    plt.show()


# Vizualizácia zhodných bodov
matches_keypoints_true = [match[1] for match in matches]
matches_keypoints_loftr = [match[2] for match in matches]
draw_matched_keypoints(img0, img1, matches_keypoints_true, matches_keypoints_loftr)

# Statistika vzdialenosti medzi skutočnými a nájdenými bodmi
if distances_list:
    avg_distance = np.mean(distances_list)
    max_distance = np.max(distances_list)
    min_distance = np.min(distances_list)
    print(f'Priemerná vzdialenosť: {avg_distance:.2f} pixely')
    print(f'Maximálna vzdialenosť: {max_distance:.2f} pixely')
    print(f'Minimálna vzdialenosť: {min_distance:.2f} pixely')
else:
    print("Žiadne zhodné body neboli nájdené.")

