import cv2
import classification as cl


test_image =  cv2.imread("modified_images/m_17.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
