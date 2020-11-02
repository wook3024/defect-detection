import dip.image as im
import numpy as np
import cv2

# def image_preprocessor(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = im.threshold(image, min_limit=127)
#     return cv2.bitwise_not(image)
    
# def label_preprocessor(label):
#     label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#     return im.threshold(label, min_limit=127)

def posprocessor(image):
    return im.threshold(image)



def image_preprocessor(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = im.edges(image, threshold1=250, threshold2=200, kernel=3)

    ############### use image preprocessing ################
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ########################################################
    return image
    
def label_preprocessor(label):
    # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    return label