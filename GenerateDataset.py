import cv2
import numpy as np
import os
import json

# Cesty
INPUT_IMAGE = "generated_images/note_000.png"
INPUT_KPTS = "keypoints/note_000.json"
OUT_IMG_DIR = "dataset/images"
OUT_KPT_DIR = "dataset/keypoints"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_KPT_DIR, exist_ok=True)

# Načítaj základný obrázok
base_img = cv2.imread(INPUT_IMAGE)
h, w = base_img.shape[:2]
center = (w // 2, h // 2)

# Načítaj základné keypointy
with open(INPUT_KPTS, "r") as f:
    base_kpts_data = json.load(f)

base_kpts = np.array([[kp["x"], kp["y"]] for kp in base_kpts_data["keypoints"]])

# Generuj dataset
NUM_VARIANTS = 100

for i in range(NUM_VARIANTS):
    # Vytvor kópiu obrázka
    img = base_img.copy()

    # Náhodné parametre
    angle = np.random.uniform(-180, 180)
    scale = np.random.uniform(0.5, 1.5)
    tx = np.random.randint(-20, 20)
    ty = np.random.randint(-20, 20)

    # Transformácia (rotácia + škálovanie + posun)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += [tx, ty]  # pridaj transláciu

    # Transformuj obrázok
    transformed_img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # Transformuj keypointy
    kpts_homo = np.hstack([base_kpts, np.ones((len(base_kpts), 1))])
    transformed_kpts = (M @ kpts_homo.T).T

    # Ulož obrázok
    img_name = f"note_{i:03d}.png"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), transformed_img)

    # Ulož keypointy
    keypoints = [{"name": base_kpts_data["keypoints"][j]["name"],
                  "x": float(transformed_kpts[j][0]),
                  "y": float(transformed_kpts[j][1])}
                 for j in range(len(transformed_kpts))]

    with open(os.path.join(OUT_KPT_DIR, img_name.replace(".png", ".json")), "w") as f:
        json.dump({"keypoints": keypoints}, f, indent=2)

print("✅ Dataset bol úspešne vygenerovaný.")
