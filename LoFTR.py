import torch
import kornia
import cv2
import matplotlib.pyplot as plt

# Načítanie modelu (inicializácia s predvolenou konfiguráciou)
matcher = kornia.feature.LoFTR(pretrained='outdoor')
matcher = matcher.eval().cuda()  # Presun na GPU ak je dostupné

# Načítanie obrázkov
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

max_size = 640  # Môžete znížiť na 512 alebo 320 pre menšie GPU

def resize_image(img):
    h, w = img.shape[:2]
    scale = min(max_size/h, max_size/w)
    return cv2.resize(img, (int(w*scale), int(h*scale)))

img1 = resize_image(img1)
img2 = resize_image(img2)

# Konverzia na torch tensor a normalizácia
img1_t = torch.from_numpy(img1).float()[None, None].cuda() / 255.
img2_t = torch.from_numpy(img2).float()[None, None].cuda() / 255.

# Párovanie
with torch.no_grad():
    input_dict = {"image0": img1_t, "image1": img2_t}
    correspondences = matcher(input_dict)

    # Extrakcia párovaných bodov
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

# Vizualizácia výsledkov
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
ax1.imshow(img1, cmap='gray')
ax2.imshow(img2, cmap='gray')

# Kreslenie párovaných bodov a čiar
for i in range(len(mkpts0)):
    ax1.scatter(mkpts0[i, 0], mkpts0[i, 1], c='r', s=10)
    ax2.scatter(mkpts1[i, 0], mkpts1[i, 1], c='r', s=10)
    fig.axes[0].plot([mkpts0[i, 0], mkpts1[i, 0] + img1.shape[1]],
                     [mkpts0[i, 1], mkpts1[i, 1]], 'y-', linewidth=0.5)

plt.tight_layout()
plt.show()