import cv2
import numpy as np
img_1 = np.zeros([500,500,1],dtype=np.uint8)
img_1.fill(255)
# or img[:] = 255
cv2.imshow('Single Channel Window', img_1)
print("image shape: ", img_1.shape)
img_3 = np.zeros([512,512,3],dtype=np.uint8)
img_3.fill(255)
# or img[:] = 255
cv2.imwrite('white.jpg', img_3)
cv2.waitKey(0)
cv2.destroyAllWindows()