'''
Are you trying to look through a large video and save some indivdual frames?

This tool lets you move forwards, and backards through a video at different speeds and save the frames you want.

Simply select the path to the video and the paths to the folders you want to save the images to.

    1. Select your video
    2. Select the folder you want to save to. If you have a different amount of folders folders:
    
        Create a variable string for each one
        Add that variable to classes list
        In the series of if statements, add a new elif statement for each folder
        
    3. Run the script
    4. There are multiple keys you can press to move through the video:
        'q' - Go forward 1 frame
        'w' - Go forward 10 frames
        'e' - Go forward 50 frames
        'r' - Go forward 150 frames

        'a' - Go backward 1 frame
        's' - Go backward 10 frames
        'd' - Go backward 50 frames
        'f' - Go backward 150 frames


    You can change these if you have more or less folders:
    
        'z' - Save the frame to the first folder
        'x' - Save the frame to the second folder
        'c' - Save the frame to the third folder
        
        'p' - Exit        
'''
import cv2
import os

video = 13

# Path to the video
video_path = f'/Users/brianprzezdziecki/Downloads/Run{video}.mov'

# Paths to the folders you want to save the images to.
under_folder = f'data/data{video}/under'
over_folder = f'data/data{video}/over'
normal_folder = f'data/data{video}/normal'

# If you want to save all images to one folder, just put that one folder in this list
classes = [under_folder, over_folder, normal_folder]

# Read video from specified path
cam = cv2.VideoCapture(video_path)
frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

# Try to create folders if they don't exist
for folder in classes:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print ('Error: Creating directory of data')
    
currentframe = 0

while(True):
    cam.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
    ret, frame = cam.read()
    
    if ret:
        cv2.imshow('frame', frame)
        # Wait for the user to press keys and perform corresponding actions
        key = cv2.waitKey(0) & 0xFF
        
        # Save frames to the corresponding folders
        # Add or remove elif statements if you have more or less folders
        if key == ord('z'):
            name = under_folder + '/frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
        elif key == ord('x'):
            name = normal_folder + '/frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
        elif key == ord('c'):
            name = over_folder + '/frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
        
        # Go ahead frames
        elif key == ord('q'):  # go forward 1 frame
            currentframe = min(currentframe + 1, frame_count - 1)
        elif key == ord('w'):  # go forward 10 frames
            currentframe = min(currentframe + 10, frame_count - 1)
        elif key == ord('e'):  # go forward 50 frames
            currentframe = min(currentframe + 50, frame_count - 1)
        elif key == ord('r'):  # go forward 150 frames
            currentframe = min(currentframe + 150, frame_count - 1)
        
        # Go backward frames
        elif key == ord('a'):  # go backward 1 frame
            currentframe = max(currentframe - 1, 0)
        elif key == ord('s'):  # go backward 10 frames
            currentframe = max(currentframe - 10, 0)
        elif key == ord('d'):  # go backward 50 frames
            currentframe = max(currentframe - 50, 0)
        elif key == ord('f'):  # go backward 150 frames
            currentframe = max(currentframe - 150, 0)
        
        # Exit
        elif key == ord('p'):  # exit
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()
