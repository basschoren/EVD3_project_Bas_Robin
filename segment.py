import os
import glob
import cv2 as cv
import numpy as np

def maskSignBG(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    light_red = (70,0,0)
    dark_red = (255,255,255)

    mask = ~cv.inRange(img, light_red, dark_red)
    return mask


if __name__ == "__main__":
    """ Test segmentation functions"""
    data_path = r'C:\Users\bassc\PycharmProjects\sign_recognition\data\sign50'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:

        # load image and blur a bit
        img = cv.imread(filename)
        img = cv.blur(img, (3, 3))

        # mask background
        mask = maskSignBG(img)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # show result and wait a bit
        cv.imshow("Masked image", masked_img)
        k = cv.waitKey(0) & 0xFF

        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break