################################################### 
#                   AUTHORS                       #
#         Advait Deshmukh - 5029XXXX              #
#          Priyanka Pai   - 5029XXXX              #
#                                                 #
#              README before running              #
#         To run via command line,                #
#       delete outputs in folders and run         #
#  python[space]stitch1.py[space]../img_directory #
#                                                 #
###################################################

import cv2 #Make sure it's OpenCV 3.4.2 (using cv2.__version__) or lower for SIFT Feature Extraction to work
# If not, in terminal pip/pip3 uninstall opencv-python -> pip install opencv-python==3.4.2.17 and pip install opencv-contrib-python==3.4.2.17
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import glob
import os
import sys

def get_corners(image):
  corners = np.zeros((4, 1, 2), dtype=np.float32)
  shape = image.shape
  y = shape[0]
  x = shape[1]
  corners[0] = [0, 0]
  corners[1] = [0, y]
  corners[2] = [x, y]
  corners[3] = [x, 0]
  return corners

def get_min(result_dims, axis):
  if axis == 'x':
    x_min = int(result_dims.min(axis=0).reshape(-1, order='A')[0])
    return x_min
  if axis == 'y':
    y_min = int(result_dims.min(axis=0).reshape(-1, order='A')[1])
    return y_min

def get_max(result_dims, axis):
  if axis == 'x':
    x_max = int(result_dims.max(axis=0).reshape(-1, order='A')[0])
    return x_max
  if axis == 'y':
    y_max = int(result_dims.max(axis=0).reshape(-1, order='A')[1])
    return y_max


def get_matches(img1, img2, t):
  sift=cv2.xfeatures2d.SIFT_create(t)
  k1, d1 = sift.detectAndCompute(img1, None)
  k2, d2 = sift.detectAndCompute(img2, None)
  matches = []
  pt1 = 0
  pt2 = 0

  for i in range(len(d1)):
      least = 100
      for j in range(len(d2)):
          n = np.linalg.norm(d1[i]-d2[j]) #Compute Euclidean Distance
          if n < least: 
              least = n
              pt1 = i
              pt2 = j
      matches.append([k1[pt1].pt,k2[pt2].pt])
  return matches


def findHomography(matches):
  maxliner=[]
  finalH=None

  #r=np.matrix([[0],[0],[0],[0],[0],[0],[0],[0],[1]])

  for i in range(1000):
      c = matches[random.randrange(0,len(matches))]
      c1 = matches[random.randrange(0,len(matches))]
      c2 = matches[random.randrange(0,len(matches))]
      c3 = matches[random.randrange(0,len(matches))]

      #Stack four arrays of 2x9 size
      p1 = np.array([-c[0][0], -c[0][1], -1,0,0,0,c[0][0]*c[1][0],c[0][1]*c[1][0],c[1][0]])
      p1_1 = np.array([0,0,0,-c[0][0], -c[0][1], -1,c[0][0]*c[1][1],c[0][1]*c[1][1],c[1][1]])
      y = np.vstack((p1,p1_1))
      
      p2 = np.array([-c1[0][0], -c1[0][1], -1,0,0,0,c1[0][0]*c1[1][0],c1[0][1]*c1[1][0],c1[1][0]])
      p2_1 = np.array([0,0,0,-c1[0][0], -c1[0][1], -1,c1[0][0]*c1[1][1],c1[0][1]*c1[1][1],c1[1][1]])
      y_1=np.vstack((p2,p2_1))

      p3 = np.array([-c2[0][0], -c2[0][1], -1,0,0,0,c2[0][0]*c2[1][0],c2[0][1]*c2[1][0],c2[1][0]])
      p3_1 = np.array([0,0,0,-c2[0][0], -c2[0][1], -1,c2[0][0]*c2[1][1],c2[0][1]*c2[1][1],c2[1][1]])
      y_2 = np.vstack((p3,p3_1))
      
      p4 = np.array([-c3[0][0], -c3[0][1], -1,0,0,0,c3[0][0]*c3[1][0],c3[0][1]*c3[1][0],c3[1][0]])
      p4_1 = np.array([0,0,0,-c3[0][0], -c3[0][1], -1,c3[0][0]*c3[1][1],c3[0][1]*c3[1][1],c3[1][1]])
      y_3 = np.vstack((p4,p4_1))
      
      y_4 = np.array([0,0,0,0,0,0,0,0,1])
      M = np.vstack((y,y_1,y_2,y_3))
      #M=np.matrix(M)

      u, s, v = np.linalg.svd(M)
      H = np.reshape(v[8], (3, 3)) #Get the 3x3 homography matrix

      H= (1/H.item(8)) * H 

      aliner = []

      for i in range(len(matches)):
        coordinates_H = matches[i]
        original_pt = np.transpose(np.matrix([coordinates_H[0][0],coordinates_H[0][1],1]))
        calculated_pt = np.dot(H,original_pt)
        calculated_pt = (1 / calculated_pt.item(2)) * calculated_pt
        p2 = np.transpose(np.matrix([coordinates_H[1][0],coordinates_H[1][1],1]))
        distance = p2 - calculated_pt
        distance = np.linalg.norm(distance)
        if distance < 10:
          aliner.append(coordinates_H)

      if len(aliner)>len(maxliner):   #Keep Best Homography Matrix and Normalize
        maxliner = aliner
        homography = H
        homography = np.linalg.inv(homography)
        homography = (1 / homography[2][2]) * homography #Normalize the homography matrix by dividing with last element
  return homography



def stitch(img1,img2,H):
    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

    #dimensions
    dimension1 = get_corners(img1).reshape(-1, 1, 2)
    dimension2_temp = get_corners(img2).reshape(-1, 1, 2)
    dimension2 = cv2.perspectiveTransform(dimension2_temp, H)
    result_dimension = np.concatenate((dimension1, dimension2))

    x_min = get_min(result_dimension, 'x')
    y_min = get_min(result_dimension, 'y')
    x_max = get_max(result_dimension, 'x')
    y_max = get_max(result_dimension, 'y')

    b = [-x_min,-y_min]
    H_ = np.array([[1, 0, b[0]], [0, 1, b[1]], [0,0,1]])

    #im1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)  
    result_img = cv2.warpPerspective(img2, H_.dot(H),(x_max-x_min, y_max-y_min), flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_TRANSPARENT)
    result_img[-y_min: img1.shape[:2][0] + -y_min, -x_min: img1.shape[:2][1] + -x_min] = img1
    return result_img
    
def process(img1, img2):
	print('Getting matches...')
	matches = get_matches(img1, img2, 3000)	#Modify the number of features depending on the size/type of your images
	print('Finding Homography...')
	homography = findHomography(matches)
	return homography


def main():
    directory=sys.argv[1]
    print([file for file in sorted(glob.glob((os.path.join(str(directory),"*.jpg"))))])
    images = [cv2.imread(file) for file in sorted(glob.glob(os.path.join(str(directory),"*.jpg")))]
    #images = []
    #images.append(cv2.imread('94.jpg'))
    #images.append(cv2.imread('95.jpg'))
    #images.append(cv2.imread('96.jpg'))
    #images=os.listdir(path)
    if len(images)==2:
    	homography = process(images[0], images[1])
    	pan1 = stitch(images[0], images[1], homography)
    	cv2.imwrite("../ubdata1/panorama2.jpg", pan1)
    else:
    	homography = process(images[0], images[1])
    	print('Stitching...')
    	pan1 = stitch(images[1],images[0],np.linalg.inv(homography))
    	result = pan1.copy()
    	homography1 = process(result, images[2])
    	print('Stitching...')
    	final=stitch(result, images[2], homography1)
    	cv2.imwrite("../ubdata/panorama.jpg",final)
    	print('File written')

if __name__ == "__main__":
    main()