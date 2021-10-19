import numpy as np

import cv2

# Create a black image

img = np.zeros((500,500,3), np.uint8)
img.fill(255)
#Diagonal Blue line
#img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

#Cicle
img = cv2.circle(img,(150,200),10,(0,0,255),-1)


#Ellipse
#cv2.ellipse(image, centerpoint, axes, angle, startAngle, endAngle, color,thickness)  
#img = cv2.ellipse(img,(400,450),(50,25),0,0,180,(0,255,0),-1)

#Polyline1
pts = np.array([[200,100],[270,160],[230,140],[270,50]], np.int32)
pts = pts.reshape((-1,1,2))
#img = cv2.polylines(img,[pts],True,(0,0,0),)

cv2.rectangle(img, (60, 80), (90, 120), (255,0,0), 2)
cv2.rectangle(img, (100, 300), (150, 350), (255,0,0), 2)
cv2.rectangle(img, (300, 400), (350, 450), (255,0,0), 2)
cv2.rectangle(img, (150, 100), (180, 150), (255,0,0), 2)


cv2.imwrite('img.png', img)



'''
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''