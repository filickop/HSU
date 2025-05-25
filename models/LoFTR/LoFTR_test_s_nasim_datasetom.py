import torch
import kornia
import kornia.feature as KF
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Načítanie obrázkov (v grayscale)
def load_and_preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    tens = torch.from_numpy(img / 255.).float()[None, None]
    return tens.cuda() if torch.cuda.is_available() else tens

# Načítaj referenčný a cieľový obrázok
img0 = load_and_preprocess("../../dataset/Note/images/note_000.png")
img1 = load_and_preprocess("../../dataset/Note/images/note_015.png")

# Inicializácia LoFTR
matcher = KF.LoFTR(pretrained='outdoor').eval()
if torch.cuda.is_available():
    matcher = matcher.cuda()

# Matching
with torch.no_grad():
    input_dict = {
        "image0": img0,
        "image1": img1,
    }
    correspondences = matcher(input_dict)

# Výstupy
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()

# Vizualizácia
def draw_matches(img0, img1, kpts0, kpts1):
    img0_np = img0.squeeze().cpu().numpy()
    img1_np = img1.squeeze().cpu().numpy()
    img_comb = np.hstack([img0_np, img1_np])
    img_comb = cv2.cvtColor((img_comb * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for (x0, y0), (x1, y1) in zip(kpts0, kpts1):
        x1_shifted = x1 + img0_np.shape[1]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_comb, (int(x0), int(y0)), (int(x1_shifted), int(y1)), color, 1)
        cv2.circle(img_comb, (int(x0), int(y0)), 2, color, -1)
        cv2.circle(img_comb, (int(x1_shifted), int(y1)), 2, color, -1)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_comb)
    plt.axis('off')
    plt.title(f'LoFTR Matches ({len(kpts0)} points)')
    plt.show()

draw_matches(img0, img1, mkpts0, mkpts1)
