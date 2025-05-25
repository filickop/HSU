import cv2
import numpy as np
import os
import json

class DatasetGenerator:
    def __init__(self, input_image, input_keypoints, output_image_dir, output_keypoints_path):
        self.input_image = input_image
        self.input_keypoints = input_keypoints
        self.output_image_dir = output_image_dir
        self.output_keypoints_path = output_keypoints_path

        os.makedirs(self.output_image_dir, exist_ok=True)

    def generate(self, num_variants):
        if os.path.exists(self.output_keypoints_path) and len(os.listdir(self.output_image_dir)) >= num_variants:
            print("Dataset už existuje, generovanie preskočené.")
            return
        # Načítaj základný obrázok
        base_img = cv2.imread(self.input_image)
        if base_img is None:
            raise FileNotFoundError(f"Obrázok '{self.input_image}' sa nenašiel.")
        h, w = base_img.shape[:2]
        center = (w // 2, h // 2)

        # Načítaj základné keypointy
        with open(self.input_keypoints, "r") as f:
            base_kpts_data = json.load(f)

        base_kpts = np.array([[kp["x"], kp["y"]] for kp in base_kpts_data["keypoints"]])

        # Priprav pole pre všetky záznamy
        all_entries = []

        for i in range(num_variants):
            # Vytvor kópiu obrázka
            img = base_img.copy()

            # Náhodné parametre
            angle = np.random.uniform(-180, 180)
            scale = np.random.uniform(0.5, 1.5)
            tx = np.random.randint(-20, 20)
            ty = np.random.randint(-20, 20)

            # Transformácia
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[:, 2] += [tx, ty]

            # Transformuj obrázok
            transformed_img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

            # Transformuj keypointy
            kpts_homo = np.hstack([base_kpts, np.ones((len(base_kpts), 1))])
            transformed_kpts = (M @ kpts_homo.T).T

            # Ulož obrázok
            img_name = f"note_{i:03d}.png"
            img_path = os.path.join(self.output_image_dir, img_name)
            rel_path = os.path.relpath(img_path, start=os.path.dirname(self.output_keypoints_path)).replace("\\", "/")
            cv2.imwrite(img_path, transformed_img)

            # Pridaj záznam
            entry = {
                "path": rel_path,
                "keypoints": [{"x": float(kp[0]), "y": float(kp[1])} for kp in transformed_kpts]
            }
            all_entries.append(entry)

        # Ulož všetky záznamy do JSON
        with open(self.output_keypoints_path, "w") as f:
            json.dump(all_entries, f, indent=2)

        print("✅ Dataset bol úspešne vygenerovaný vo forme zoznamu objektov.")