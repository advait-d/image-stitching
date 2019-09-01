# Image Stitching
As a part of the Computer Vision and Image Processing course, we performed image stitching without the use of OpenCV's library functions such as cv2.BFMatcher or cv2.findHomography

To run via command line, delete outputs if they exist as panorama.jpg (in case of two images) in folders and perform

'python[space]stitch1.py[space]../img_directory'

The final output will be saved as 'panorama.jpg' in the folder your original images were stored in.  You can change this on line #183 in the cv2.imwrite method.

Since we have used a fixed number of Keypoints to be mamtched, the output may vary depending on your input images. In this case, you may change the number of matches on line #156.
These matches will usually be approximately equal to the less than half of the number of pixels in the overlapping regions.
We wouldn't recommend setting the number of matches too high as it may take a lot of time depending on your machine.
