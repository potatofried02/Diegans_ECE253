import cv2
import numpy as np
import os
import glob


def make_improved_glare_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask_bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=12, sigmaY=12)
    high = cv2.subtract(gray, blur)
    _, mask_contrast = cv2.threshold(high, 30, 255, cv2.THRESH_BINARY)

    mask = cv2.max(mask_bright, mask_contrast)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


def process_inpaint(img):
    mask = make_improved_glare_mask(img)

    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return result, mask


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, "data")

    glare_dir = os.path.join(data_dir, "imgur5k_glare")
    out_dir = os.path.join(data_dir, "imgur5k_inpaint")
    os.makedirs(out_dir, exist_ok=True)

    glare_paths = sorted(glob.glob(os.path.join(glare_dir, "*.png")))

    for gpath in glare_paths:
        img = cv2.imread(gpath)
        result, mask = process_inpaint(img)

        fname = os.path.basename(gpath)
        name = fname.replace("_glare", "").replace(".png", "")

        out_path = os.path.join(out_dir, f"{name}_inpaint.png")
        cv2.imwrite(out_path, result)

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
