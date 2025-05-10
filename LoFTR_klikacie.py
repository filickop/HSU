import torch
import kornia
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Toto je dôležité pre klikanie v okne
import matplotlib.pyplot as plt
import numpy as np

# === Inicializácia LoFTR ===
matcher = kornia.feature.LoFTR(pretrained='outdoor').eval().cuda()

# === Načítanie obrázkov ===
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("Uisti sa, že image1.jpg a image2.jpg sú v rovnakom priečinku ako skript.")

# === Zmenšenie veľkosti (pre rýchlosť) ===
max_size = 640
def resize_image(img):
    h, w = img.shape[:2]
    scale = min(max_size/h, max_size/w)
    return cv2.resize(img, (int(w*scale), int(h*scale)))

img1 = resize_image(img1)
img2 = resize_image(img2)

# === Prevod na torch.Tensor ===
img1_t = torch.from_numpy(img1).float()[None, None].cuda() / 255.
img2_t = torch.from_numpy(img2).float()[None, None].cuda() / 255.

# === Výber bodov v obrázku 1 ===
plt.figure(figsize=(8, 6))
plt.imshow(img1, cmap='gray')
plt.title("Klikni ľubovoľné body (ENTER = potvrdiť výber)")
selected_pts = np.array(plt.ginput(n=-1, timeout=0))
plt.close()

if len(selected_pts) == 0:
    print("⚠️ Neboli vybraté žiadne body.")
    exit()

print(f"✅ Vybraných {len(selected_pts)} bodov.")

# === LoFTR matching ===
with torch.no_grad():
    input_dict = {"image0": img1_t, "image1": img2_t}
    correspondences = matcher(input_dict)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

# === Filtrovanie: len najbližšie matchy k vybraným bodom ===
def find_closest_matches(selected_pts, mkpts0, mkpts1, max_dist=10):
    matched_src = []
    matched_dst = []
    for pt in selected_pts:
        dists = np.linalg.norm(mkpts0 - pt, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < max_dist:
            matched_src.append(mkpts0[min_idx])
            matched_dst.append(mkpts1[min_idx])
    return np.array(matched_src), np.array(matched_dst)

matched_src, matched_dst = find_closest_matches(selected_pts, mkpts0, mkpts1)

# === Zobrazenie výsledkov ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(img1, cmap='gray')
ax2.imshow(img2, cmap='gray')
ax1.set_title("Obrázok 1 (vybrané body)")
ax2.set_title("Obrázok 2 (matched body)")

for i in range(len(matched_src)):
    ax1.scatter(matched_src[i, 0], matched_src[i, 1], c='lime', s=40)
    ax2.scatter(matched_dst[i, 0], matched_dst[i, 1], c='cyan', s=40)
    ax1.plot([matched_src[i, 0], matched_dst[i, 0]],
             [matched_src[i, 1], matched_dst[i, 1]], 'y--', linewidth=0.7)

plt.suptitle("LoFTR: matched body pre vybraný výber")
plt.tight_layout()
plt.show()
