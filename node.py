# import the necessary packages
from numpy.lib.npyio import genfromtxt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import random
import math

#number of node
node =[]
obs =[]

def start_goal():
    start = [15, 20]
    goal = [490, 490]
    return (start, goal)

def create_node(r):
    for i in range(0, 250):
        value = (random.randint(0, r), random.randint(0, r))
        node.append(value)
    return node

def read_obstacles():
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread("img.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    orig = image.copy()

    # loop over the contours individually
    z = 0
    for c in cnts:
        z = z+1
        if cv2.contourArea(c) < 100:
                continue
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        (tl, tr, br, bl) = box.astype("int")
        
        pixel_x = [tl[0], tr[0], br[0], bl[0]]
        pixel_y = [tl[1], tr[1], br[1], bl[1]]
        max_x = max(pixel_x)
        min_x = min(pixel_x)
        max_y = max(pixel_y)
        min_y = min(pixel_y)
        
        # width & height of obstacles
        k=0
        l=0
        d1 = max_x - min_x
        d2 = max_y - min_y

        x1, y1 = min_x, min_y
        for m in range(-5, int(d1+5)):
            k=x1+m
            for n in range(-5, int(d2+5)):
                l=y1+n
                obs.append((k,l))
    return (orig, obs)

def filter_node(node, obs):
    final=[]
    for j in node:
        for i in obs:
            if j == i:
                final.append(j)
    final = list(set(final))
    for k in final:
        node.remove(k)
    return node


def draw_node(orig, node):          
    for i in node:
        cv2.circle(orig, (i), 2, (155, 155, 155), -1)
    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)

def joint_node(orig, node):
    #cv2.line(image, start_point, end_point, color, thickness)
    a = np.array(node)
    cv2.drawContours(orig, [a], 0, (255,0,0), 2) 
    return orig

def joint_byStart(orig, points, g):
    (sx, sy) = points[0]
    (gx, gy) = g
    act_path = ((gx - sx)**2 + (gy - sy)**2)**0.5
    i = 0
    def filtering(x, g):
        (gx, gy) = g
        x1, y1 = x
        points.remove(x)
        z = len(points)
        print(z)
        if x == g:
            return orig
        else:
            dist = []
            p = []
            if (z !=1):
                for j in range(0, z -1):
                    x2, y2 = points[j]
                    a = (x2 - x1) **2
                    b = (y2 - y1)**2
                    d1 = (a+b)**(0.5)
                    u = (gx - x2)**2
                    v = (gy - y2)**2
                    d2 = (u+v)**(0.5)
                    d = (d1+d2)*(0.5)
                    dist.append(d)
                    p.append(points[j]) 
                    m = min(dist) 
                    k = dist.index(m)
                    cv2.line(orig, x, p[k], (255, 0, 255), thickness=1, lineType=8)
                    return filtering(p[k], g)    
            else:
               return orig 
    orig = filtering(points[0], g)
    return orig 




#Get Start point & goal
(start, goal) = start_goal()
(sx, sy) = start
(gx, gy) = goal



# Get obstacles
(orig, obs) = read_obstacles()

# Get node for first 500 radius
node = create_node(500)

# filter node
node = filter_node(node, obs)

# draw start & goal
orig = cv2.circle(orig,(sx, sy),10,(0,0,0),-1)
orig = cv2.circle(orig,(gx, gy),10,(0,0,0),-1)

# joint Points
#orig = joint_node(orig, node)

points = []
points.append((sx, sy))
for i in node:
    points.append(i)
points.append((gx, gy))
# start point one by one
orig = joint_byStart(orig, points, goal)

# show in image
draw_node(orig, node)







	# loop over the original points and draw them
	#for (x, y) in box:
		#cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)