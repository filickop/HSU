import torch
import kornia
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Nastavenia
MAX_SIZE = 512
MODEL_TYPE = 'indoor'  # 'indoor' pre vnútorné scény, 'outdoor' pre vonkajšie

# Inicializácia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Používam zariadenie: {device}")


def load_and_preprocess(img_path):
    """Načíta a predspracuje obrázok"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Súbor {img_path} nebol nájdený v {os.getcwd()}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Nepodarilo sa načítať obrázok {img_path}")

    h, w = img.shape
    scale = min(MAX_SIZE / h, MAX_SIZE / w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    img_t = torch.from_numpy(img.astype(np.float32))[None, None].to(device) / 255.
    return img, img_t


# Hlavný kód
def main():
    # Načítanie obrázkov
    img1, img1_t = load_and_preprocess('image1.jpg')
    img2, img2_t = load_and_preprocess('image2.jpg')

    # Načítanie modelu
    matcher = kornia.feature.LoFTR(pretrained=MODEL_TYPE).eval().to(device)

    # Vytvorenie figúry pre označovanie
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img1, cmap='gray')
    ax.set_title(
        'Kliknite ľavým tlačidlom na body ktoré chcete porovnať\nPotvrdite pravým tlačidlom alebo stlačte Enter')

    selected_points = []

    def on_click(event):
        if event.button == 1:  # Ľavé tlačidlo myši
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                selected_points.append((x, y))
                ax.plot(x, y, 'ro', markersize=8)
                plt.draw()

    def on_key(event):
        if event.key == 'enter':
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    if not selected_points:
        print("Neboli označené žiadne body. Ukončujem.")
        return

    # Vytvorenie masky
    mask = np.zeros(img1.shape[:2], dtype=np.float32)
    for x, y in selected_points:
        x, y = int(round(x)), int(round(y))
        cv2.circle(mask, (x, y), 15, 1, -1)

    # Spracovanie
    with torch.no_grad():
        correspondences = matcher({
            'image0': img1_t,
            'image1': img2_t,
            'mask0': torch.from_numpy(mask)[None, None].float().to(device)
        })

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

    # Vizualizácia výsledkov
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(img2, cmap='gray')

    for i, (pt1, pt2) in enumerate(zip(mkpts0, mkpts1)):
        ax1.plot(pt1[0], pt1[1], 'ro', markersize=5)
        ax2.plot(pt2[0], pt2[1], 'go', markersize=5)
        if i < len(selected_points):  # Spojiť len vybrané body
            fig.axes[0].plot([pt1[0], pt2[0] + img1.shape[1]],
                             [pt1[1], pt2[1]], 'y-', linewidth=1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()