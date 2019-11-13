import cv2 as cv
import sys
import ImageStitch as imgstitch   # ImageStitch is the code I wrote to define all the related functions. utils

if __name__ == '__main__':

    ## is_CV_correct(3) ???
    
    
# # read two images for stitching
# img_ = cv2.imread('original_image_right.jpg')   
# # img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
# img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
# img = cv2.imread('original_image_left.jpg')
# # img = cv2.resize(img, (0,0), fx=1, fy=1)
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# cv2.imshow('Left Image',img)
# cv2.imshow('Right Image',img_)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
    
    if sys.argv[1] and sys.argv[2]:   
        
        ## https://www.pythonforbeginners.com/argv/more-fun-with-sys-argv
        
        image1 = cv.imread(sys.argv[1])
        image2 = cv.imread(sys.argv[2])
        stitch_match = imgstitch.FindKeyPointsAndMatching()
        kp1, kp2 = stitch_match.get_key_points(img_left = image1, img_right = image2)
        homo_matrix = stitch_match.match(kp1, kp2)
        stitch_merge = imgstitch.StitchTwoImages()
        dst = stitch_merge.stitch(img_left = image1, img_right = image2, homo_matrix = homo_matrix)
        #print("dst.shape()", dst.info)
        trim_image = stitch_merge.trim(frame=dst)
        print("type(trim_image):", type(trim_image))
        print(trim_image)

        #cv.namedWindow('output', 0)
        cv.imshow("original_image_stitched.jpg", trim_image)
        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()
        cv.imwrite(sys.argv[1][0] + '-output.jpg', trim_image)
        print('\n=======>Output saved!')
    else:
        print('input images location!')
        
        
## Save the image
# trimed_output = trim(dst)
# cv2.imwrite("output_stitched.jpg", trimed_output)

## Display the output image
# cv2.imshow("original_image_stitched_crop.jpg", trimed_output)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()