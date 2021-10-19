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
    start = [20, 20]
    goal = [400, 400]
    return (start, goal)

def distance_bw_sg(start, goal, node):
    (sx, sy) = start
    (gx, gy) = goal
    val = []
    points = []
    for i in node:
        (x, y) = i
        a = 2**0.5
        m = (x - y)/a
        if m <0:
            m = m*-1
        points.append(i)
        val.append(m)
    final = []
    k = -1
    for i in val:
        k = k +1
        if i<40:
            final.append(points[k])
    return final

def find_nextnode( orig, start, node, obs):
    (x1, y1) = start
    (gx, gy) = node[len(node)-1]
    for j in range (0, len(node)):
        k = []
        x2, y2 = node[j]
        if (x2 - x1) >0 and (y2-y1) >0:
            a = (x2 - x1) **2
            b = (y2 - y1)**2
            d = (a+b)**(0.5)
            n = 2**0.5
            m = (x2 - y2)/n
            if m < 0:
                m= m*-1
            if m < 30 and d < 100:
                for i in obs:
                    (x , y) = i
                    f = ((y - y1)*(x2 - x1)) - ((x - x1)*(y2 - y1))
                    k.append(f)
                print(min(k))
                if min(k) != 0:
                    cv2.line(orig, (x1, y1), (x2, y2), (255, 0, 255), thickness=1, lineType=8)
                    (x1, y1) = (x2, y2)
    return orig
        
def find_path( orig, node, start, goal, obs):
    (sx, sy) = start
    (gx, gy) = goal
    node.append(goal)
    orig = find_nextnode( orig, start, node, obs)
    return orig

def create_node(x , y):
    n = x*y/(x/10)
    for i in range(0, int(n)):
        value = (random.randint(0, x), random.randint(0, y))
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
    count = 10
    
    def filtering(l, g, orig, count):
        ul = []
        change = 0
        (gx, gy) = g
        p = []
        for x in l:
            if x in points:
                points.remove(x)
            x1, y1 = x
            z = len(points)
            if x == g:
                return orig
            else:
                dist = []
                if z !=1:
                    for j in range(0, z -1):
                        x2, y2 = points[j]
                        a = (x2 - x1) **2
                        b = (y2 - y1)**2
                        d1 = (a+b)**(0.5)
                        u = (gx - x2)**2
                        v = (gy - y2)**2
                        d2 = (u+v)**(0.5)
                        d = (d1+d2)*0.5
                        if d1<count:
                            dist.append(d1)
                            p.append(points[j]) 
                    if len(dist):
                        change = 0
                        for m in p:
                            cv2.line(orig, x, m, (255, 0, 255), thickness=1, lineType=8)
                        if count>10:
                            count = 10
                        return filtering(p, g, orig, count)   
                    else:
                        change = 1
                        ul.append(x)
                else:
                    return orig 
        if change ==1 and len(l) == len(ul):
            count = count+10
            return filtering(ul, g, orig, count)
        return orig
    l = []
    l.append(points[0])
    orig = filtering(l, g, orig, count)
    return orig 




#Get Start point & goal
(start, goal) = start_goal()
(sx, sy) = start
(gx, gy) = goal



# Get obstacles
(orig, obs) = read_obstacles()

# Get node for first 500 radius
node = create_node(gx, gy)

# filter node
node = filter_node(node, obs)

# draw start & goal
orig = cv2.circle(orig,(sx, sy),10,(0,0,0),-1)
orig = cv2.circle(orig,(gx, gy),10,(0,0,0),-1)

# filter near by value
node = distance_bw_sg(start, goal, node)

# joint one by one
orig = find_path(orig, node,  start, goal, obs)

# show in image
draw_node(orig, node)
