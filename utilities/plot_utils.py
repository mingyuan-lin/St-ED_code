import os
import cv2
import numpy as np


def ensure_dir(s):
    if not os.path.exists(s):
        os.makedirs(s)

def show_img(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (0,2,3,1))
    img = np.squeeze(img)
    img = img.clip(0,1) * 255.
    #img = cv2.cvtColor(img.clip(0,1) * 255., cv2.COLOR_BGR2GRAY)
    return img

def save_disp(img):
    img = img.cpu().detach().numpy()
    img = np.squeeze(img)
    img = img * 256.
    img = img.astype(np.uint16)
    return img
