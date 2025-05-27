import cv2
import numpy as np
import os
import json

class DatasetGenerator:
    def __init__(self, input_images, input_keypoints_list, output_image_dirs, output_keypoints_paths, add_noise):
        self.input_images = input_images
        self.input_keypoints_list = input_keypoints_list
        self.output_image_dirs = output_image_dirs
        self.output_keypoints_paths = output_keypoints_paths
        self.add_noise = add_noise

        assert len(input_images) == len(input_keypoints_list) == len(output_image_dirs) == len(output_keypoints_paths), \
            "Všetky vstupné a výstupné zoznamy musia mať rovnakú dĺžku."

        for out_dir in self.output_image_dirs:
            os.makedirs(out_dir, exist_ok=True)

    def generate(self, num_variants):
        for idx in range(len(self.input_images)):
            input_image = self.input_images[idx]
            input_keypoints = self.input_keypoints_list[idx]
            output_image_dir = self.output_image_dirs[idx]
            output_keypoints_path = self.output_keypoints_paths[idx]

            if os.path.exists(output_keypoints_path) and len(os.listdir(output_image_dir)) >= num_variants:
                print(f"Dataset pre '{input_image}' už existuje, generovanie preskočené.")
                continue

            base_img = cv2.imread(input_image)
            if base_img is None:
                raise FileNotFoundError(f"Obrázok '{input_image}' sa nenašiel.")
            h, w = base_img.shape[:2]
            center = (w // 2, h // 2)

            with open(input_keypoints, "r") as f:
                base_kpts_data = json.load(f)

            base_kpts = np.array([[kp['x'], kp['y']] for kp in base_kpts_data[0]['keypoints']])
            all_entries = []

            for i in range(num_variants):
                angle = np.random.uniform(-180, 180)
                scale = np.random.uniform(0.5, 1.5)
                tx = np.random.randint(-20, 20)
                ty = np.random.randint(-20, 20)

                M_affine = cv2.getRotationMatrix2D(center, angle, scale)
                M_affine[:, 2] += [tx, ty]

                H = np.vstack([M_affine, [0, 0, 1]])

                corners = np.array([
                    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
                ], dtype=np.float32).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, H)

                x_min, y_min = np.floor(transformed_corners.min(axis=0).squeeze()).astype(int)
                x_max, y_max = np.ceil(transformed_corners.max(axis=0).squeeze()).astype(int)

                new_w, new_h = x_max - x_min, y_max - y_min

                offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
                H_shifted = offset @ H

                transformed_img = cv2.warpPerspective(base_img, H_shifted, (new_w, new_h), borderValue=(255, 255, 255))

                kpts_homo = np.hstack([base_kpts, np.ones((len(base_kpts), 1))]).T
                transformed_kpts_homo = H_shifted @ kpts_homo
                transformed_kpts = (transformed_kpts_homo[:2] / transformed_kpts_homo[2]).T

                # Zmenšenie alebo zväčšenie na požadovanú veľkosť
                target_w, target_h = base_img.shape[:2]
                resized_img = cv2.resize(transformed_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                scale_x = target_w / transformed_img.shape[1]
                scale_y = target_h / transformed_img.shape[0]
                transformed_kpts[:, 0] *= scale_x
                transformed_kpts[:, 1] *= scale_y

                img_name = f"{os.path.splitext(os.path.basename(input_image))[0]}_{i:03d}.png"
                img_path = os.path.join(output_image_dir, img_name)
                rel_path = os.path.relpath(img_path, start=os.path.dirname(output_keypoints_path)).replace("\\", "/")

                if (self.add_noise):
                    final_img, final_kps = self._add_stylized_noise(resized_img, transformed_kpts)
                else:
                    final_img = resized_img
                    final_kps = transformed_kpts

                cv2.imwrite(img_path, final_img)




                entry = {
                    "path": rel_path,
                    "keypoints": [{"x": float(kp[0]), "y": float(kp[1])} for kp in final_kps]
                }
                all_entries.append(entry)

            os.makedirs(os.path.dirname(output_keypoints_path), exist_ok=True)
            with open(output_keypoints_path, "w") as f:
                json.dump(all_entries, f, indent=2)

            print(f"✅ Dataset pre '{input_image}' bol úspešne vygenerovaný.")

    def _add_stylized_noise(self, image, keypoints):
        noisy = image.copy().astype(np.float32)
        kps = keypoints.copy()

        h, w = noisy.shape[:2]

        # 1. Random brightness change (does NOT affect keypoints)
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.6, 1.4)
            noisy *= brightness

        # 2. Random dark/bright dots (does NOT affect keypoints)
        if np.random.rand() < 0.5:
            for _ in range(np.random.randint(100, 500)):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                color = np.random.choice([0, 255])
                noisy[y, x] = color

        # 3. Asymmetric stretch (scaleX ≠ scaleY)
        if np.random.rand() < 0.5:
            scale_x = np.random.uniform(0.7, 1.3)
            scale_y = np.random.uniform(0.7, 1.3)

            # Resize image
            new_w, new_h = int(w * scale_x), int(h * scale_y)
            noisy = cv2.resize(noisy, (new_w, new_h))
            noisy = cv2.resize(noisy, (w, h))  # Resize back

            # Update keypoints (as if they were scaled and then rescaled back)
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            kps[:, 0] *= w / new_w
            kps[:, 1] *= h / new_h

        # 4. Shear (horizontal)
        if np.random.rand() < 0.3:
            dx = np.random.randint(-10, 10)
            shear_matrix = np.float32([[1, dx / h, 0], [0, 1, 0]])
            noisy = cv2.warpAffine(noisy, shear_matrix, (w, h), borderValue=255)

            # Apply affine transform to keypoints
            ones = np.ones((kps.shape[0], 1))
            kps_homo = np.hstack([kps, ones])
            kps = (shear_matrix @ kps_homo.T).T

        # Return both
        return np.clip(noisy, 0, 255).astype(np.uint8), kps
