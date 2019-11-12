#!/usr/bin/env python
# coding: utf-8

# ### AI 第五期 CV 课 Classic CV project 1： Image Stitching
# #### Student: Chen Yaofeng

# In[ ]:


"""
The image stitching algorithm consists of four steps:

Step #1: Detect keypoints (DoG, Harris, etc.) and extract local invariant descriptors (SIFT, SURF, etc.) from the two input images.
Step #2: Match the descriptors between the two images.
Step #3: Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
Step #4: Apply a warping transformation using the homography matrix obtained from Step #3.

"""


# In[1]:


import numpy as np
import cv2


# In[33]:


from matplotlib import pyplot as plt
import random


# In[13]:


def is_CV_correct(major):
    openCV_majorVersion = int(cv2.__version__.split(".")[0])
    if openCV_majorVersion == major:
        print('This program need to use OpenCV3.')
        print('Your OpenCV is OpenCV3.')
    else:
        print('This program need to use OpenCV3.')
        print('Your OpenCV is not OpenCV3. Sorry.')
    print('cv2.__version__ is:', cv2.__version__)
    return


# In[14]:


is_CV_correct(3)


# In[15]:


# read two images for stitching
img_ = cv2.imread('original_image_right.jpg')   
# img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread('original_image_left.jpg')
# img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[16]:


cv2.imshow('Left Image',img)
cv2.imshow('Right Image',img_)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[17]:


"""
Step #1: Detect keypoints (DoG, Harris, etc.) and extract local invariant descriptors (SIFT, SURF, etc.) 
         from the two input images
"""
## 1. 对两幅图像分别进行关键点检测，比如用SIFT. 【大家完全可以尝试别的关键点】
## find SIFT Keypoints and Descriptors
## https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
sift = cv2.xfeatures2d.SIFT_create()
# find the key points and descriptors with SIFT
# Here kp will be a list of keypoints and des is a numpy array of shape Number_of_Keypoints×128.
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# In[18]:


## """
## show the keypoints on the two images

img_sift_right  = cv2.drawKeypoints(img_, kp1, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift_left = cv2.drawKeypoints(img,  kp2, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('left image',img_sift_left)
cv2.imshow('right image',img_sift_right)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
### """


# In[26]:


"""
Step #2: Match the descriptors between the two images.
"""
FLANN_INDEX_KDTREE = 1  ## FLANN_INDEX_KDTREE = 0  0 or 1 does not change the results.
### FLANN_INDEX_KDTREE:
### (The upper case should have been a hint that these are meant as descriptive labels of fixed integer values.)
### https://stackoverflow.com/questions/42397009/what-values-does-the-algorithm-parametre-take-in-opencvs-flannbasedmatcher-cons
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(des1,des2,k=2)

## Select the top best matches for each descriptor of an image.
## filter out through all the matches to obtain the best ones by applying ratio test using the top 2 matches obtained above.
good = []
for m,n in matches:
    if m.distance < 0.03*n.distance:
        good.append(m)



# In[27]:


"""
Step #3: Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
"""

"""
Now we set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object.
Otherwise simply show a message saying not enough matches are present.
If enough matches are found, we extract the locations of matched keypoints in both the images. 
They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
we use it to transform the corners of queryImage to corresponding points in trainImage. 
"""

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)    
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    cv2.imshow("original_image_overlapping.jpg", img2)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))


# In[28]:


"""
Step #4: Apply a warping transformation using the homography matrix obtained from Step #3.
"""

dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
cv2.imshow("dst", dst)
dst[0:img.shape[0], 0:img.shape[1]] = img

cv2.imshow("original_image_stitched.jpg", dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[29]:


"""
Step #5: trim the stitched image and output the final image. 
"""

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


# In[32]:


## Save the image
trimed_output = trim(dst)
cv2.imwrite("output_stitched.jpg", trimed_output)


# In[31]:


## Display the output image
cv2.imshow("original_image_stitched_crop.jpg", trimed_output)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[ ]:




