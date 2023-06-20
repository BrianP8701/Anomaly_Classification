'''
This script is used to convert a video to a series of images.

Simply select the path to the video and the path to the folder you want to save the images to.
If the path to the folder doesen't exist, it will be created in the same directory as this script.

If you want to collect frames from a video for some data, it may be much faster to use the extract_frames_gui.py script instead.
'''
import cv2
import os
import numpy as np
import math
 
 
video_path = '/Users/brianprzezdziecki/Downloads/Run1.mov'
output_folder = 'Run1'


# Read the video from specified path
cam = cv2.VideoCapture(video_path)

try:
    # creating a folder named data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 0

while(True):
      
    # reading from frame
    ret,frame = cam.read()

    # Turns the image to grayscale
    frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])

    if ret:
        # if video is still left continue creating images
        name = output_folder + '/frame' + str(currentframe) + '.jpg'
        
        # Include this line if you want to see progress. It runs faster without this line
        # print ('Creating...' + name)
    
        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()


print("Done!")