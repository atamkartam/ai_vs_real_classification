import cv2
import numpy as np

def extract_rgb_histogram(images):
    features = []
    for img in images:
        hist_r = cv2.calcHist([img], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [32], [0, 256])

        hist = np.hstack([
            cv2.normalize(hist_r, hist_r).flatten(),
            cv2.normalize(hist_g, hist_g).flatten(),
            cv2.normalize(hist_b, hist_b).flatten()
        ])
        features.append(hist)

    return np.array(features)


def preprocess_image(image, size=(128, 128)):
    image = cv2.resize(image, size)
    return image


def build_feature(image):
    img = preprocess_image(image)
    feature = extract_rgb_histogram([img])
    return feature
