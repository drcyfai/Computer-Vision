# ### AI 第五期 CV 课 Classic CV project 1： Image Stitching
# #### Student: Chen Yaofeng

"""
The image stitching algorithm consists of four steps:

Step #1: Detect keypoints (DoG, Harris, etc.) and extract local invariant descriptors (SIFT, SURF, etc.) from the two input images.
Step #2: Match the descriptors between the two images.
Step #3: Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
Step #4: Apply a warping transformation using the homography matrix obtained from Step #3.

"""
import numpy as np
import cv2 as cv

class FindKeyPointsAndMatching:
    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create()
        self.brute = cv.BFMatcher()
        
    def is_CV_correct(major):
        openCV_majorVersion = int(cv.__version__.split(".")[0])
        if openCV_majorVersion == major:
            print('This program need to use OpenCV3.')
            print('Your OpenCV is OpenCV3.')
        else:
            print('This program need to use OpenCV3.')
            print('Your OpenCV is not OpenCV3. Sorry.')
        print('cv2.__version__ is:', cv.__version__)
        return
                
        """
        Step #1: Detect keypoints (DoG, Harris, etc.) and extract local invariant descriptors (SIFT, SURF, etc.) 
         from the two input images
         
        ## 1. 对两幅图像分别进行关键点检测，比如用SIFT. 【大家完全可以尝试别的关键点】
        ## find SIFT Keypoints and Descriptors
        ## https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
        """
    def get_key_points(self, img_left, img_right):
        img1 = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)  ## Note: here img1=img_right, but in the main.py image1=img_left 
        img2 = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)   ##            img2=img_left
        
        # find the key points and descriptors with SIFT
        ## sift = cv.xfeatures2d.SIFT_create(), use self.sift
        
        # Here kp will be a list of keypoints and des is a numpy array of shape Number_of_Keypoints×128.
        kp1, kp2 = {}, {}
        print('=======>Detecting key points.')
        # kp1, des1 = self.sift.detectAndCompute(img1,None)
        # kp2, des2 = self.sift.detectAndCompute(img2,None)        
        kp1['kp'], kp1['des'] = self.sift.detectAndCompute(img1,None)
        kp2['kp'], kp2['des'] = self.sift.detectAndCompute(img2,None)
        return kp1, kp2
    

       
        """
        Step #2: Match the descriptors between the two images.
        """

    def match(self, kp1, kp2):
        print('=======>Matching key points.')
        
        FLANN_INDEX_KDTREE = 1  ## FLANN_INDEX_KDTREE = 0  0 or 1 does not change the results.
         ### FLANN_INDEX_KDTREE:
         ### (The upper case should have been a hint that these are meant as descriptive labels of fixed integer values.)
         ### https://stackoverflow.com/questions/42397009/what-values-does-the-algorithm-parametre-take-in-opencvs-flannbasedmatcher-cons
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        match = cv.FlannBasedMatcher(index_params, search_params)
        matches = match.knnMatch(kp1['des'],kp2['des'], k=2)

        ## Select the top best matches for each descriptor of an image.
        ## filter out through all the matches to obtain the best ones by applying ratio test using the top 2 matches obtained above.
        good_matches = []
        for m,n in matches:
            if m.distance < 0.03*n.distance:
                good_matches.append(m)
                                
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
        MIN_MATCH_COUNT = 10  ## The minimum number should be 4, because we need at least 4 points to find homo_matrix
        
        if len(good_matches) > MIN_MATCH_COUNT:
            
            keypoints1 = kp1['kp']
            keypoints2 = kp2['kp']
            src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            
            print("=======> Random sampling and computing the homography matrix.")
            """
            # ransacReprojThreshold: 
            #Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only).
            # If srcPoints and dstPoints are measured in pixels, it usually makes sense to set this parameter 
              somewhere in the range of 1 to 10.
            # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography
            """            
            homo_matrix, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            return homo_matrix
            # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#             h,w = img1.shape
#             pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#             dst = cv2.perspectiveTransform(pts,M)
#             img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)            
        else:
            print ("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))
            return None
            

        """
        Step #4: Apply a warping transformation using the homography matrix obtained from Step #3.
        """
class StitchTwoImages:
    def __init__(self):
        pass
    
    def stitch(self, img_left, img_right, homo_matrix):
        dst = cv.warpPerspective(img_right, homo_matrix ,(img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
        ## cv.imshow("dst", dst)
        dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        
        #print("type(dst):", type(dst))
        #print(dst)
        #cv.imshow("original_image_stitched.jpg", dst)
        #key = cv.waitKey()
        #if key == 27:
        #    cv.destroyAllWindows()

        return dst

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

        #cv.imshow("original_image_stitched.jpg", frame)
        #key = cv.waitKey()
        #if key == 27:
            #cv.destroyAllWindows()

        return frame






