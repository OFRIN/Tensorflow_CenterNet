
import cv2
import numpy as np

def normalize_to_heatmap(image):
    image = cv2.applyColorMap((image * 255.).astype(np.uint8), cv2.COLORMAP_JET)
    return image