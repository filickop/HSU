import cv2
import numpy as np
import json
import os

# Výstupné adresáre
os.makedirs("generated_images", exist_ok=True)
os.makedirs("keypoints", exist_ok=True)

# Rozmery plátna
canvas_w, canvas_h = 200, 200
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

# Pozícia a rozmery hlavičky noty (elipsa)
center = (80, 130)
axes = (12, 8)
angle = -30

# Nakresli hlavičku
cv2.ellipse(canvas, center, axes, angle, 0, 360, (0, 0, 0), thickness=-1)

# Palička (od stredu hlavičky nahor)
stick_top = (center[0] + 10, center[1] - 60)
stick_bottom = (center[0] + 10, center[1])
cv2.line(canvas, stick_top, stick_bottom, (0, 0, 0), thickness=3)

# Závoj osminovej noty (voliteľne)
cv2.ellipse(canvas, (stick_top[0] + 10, stick_top[1] + 10), (10, 5), 0, 0, 180, (0, 0, 0), thickness=2)

# Definuj keypointy
keypoints = [
    {"name": "head_center", "x": center[0], "y": center[1]},
    {"name": "stick_top", "x": stick_top[0], "y": stick_top[1]},
    {"name": "stick_bottom", "x": stick_bottom[0], "y": stick_bottom[1]},
]

# Uloženie obrázku
cv2.imwrite("../../dataset/Note/note_000.png", canvas)

# Uloženie keypointov do JSON
with open("../../dataset/Note/note_000.json", "w") as f:
    json.dump({"keypoints": keypoints}, f, indent=2)

# Vizualizácia keypointov do kópie obrázka
for kp in keypoints:
    cv2.circle(canvas, (int(kp["x"]), int(kp["y"])), 4, (0, 0, 255), -1)

cv2.imwrite("generated_images/note_000_with_kpts.png", canvas)

print("✅ Obrázok noty a keypointy boli vygenerované.")
