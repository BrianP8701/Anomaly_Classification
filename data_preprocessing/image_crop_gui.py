'''
    This file provides a specific preprocessing tool to crop images.
    A user may select the size of a crop box with the crop_box_size variable. 
    
    Modify the main method to change the input and output paths.

        1. Call the main method with your image path and destination path to use the tool.

        2. Left click to select the bottom right corner of what you want to crop.

    Note: There's this annoying thing where for a moment after clicking, the crop box will 
    continue to follow your mouse. So just keep your mouse still for like, half a second after clicking.

        3. Click c twice to select and exit.
'''

import cv2

# Initialize the top right corner of the rectangle to be drawn
bottom_right_corner = (0, 0)
mouse_position = (0, 0)
crop_box_size = 640  # Size of the crop box
        
def on_mouse(event, x, y, flags, params):
    global bottom_right_corner, mouse_position

    # Start to Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        bottom_right_corner = (x, y)  # top right corner of rectangle

    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = (x, y)

def main(image_path, destination_path, image_name):
    # Set path of image
    
    image = cv2.imread(image_path)

    cv2.namedWindow(image_name)
    cv2.setMouseCallback(image_name, on_mouse)

    while True:
        img = image.copy()
        cv2.rectangle(img, (bottom_right_corner[0] - crop_box_size, bottom_right_corner[1]), (bottom_right_corner[0], bottom_right_corner[1]-crop_box_size), (255, 255, 255), 2)
        cv2.rectangle(img, (mouse_position[0] - crop_box_size, mouse_position[1]), (mouse_position[0], mouse_position[1]-crop_box_size), (0, 255, 0), 2)

        cv2.imshow(image_name, img)

        key = cv2.waitKey(10) & 0xFF

        # Exit cropping mode on pressing 'c'
        if key == ord('c'):
            break
        
        # Return without cropping on pressing 'r'
        if key == ord('r'):
            cv2.destroyAllWindows()
            return

    # Crop the image
    cropped_img = image[bottom_right_corner[1] - crop_box_size:bottom_right_corner[1], bottom_right_corner[0] - crop_box_size:bottom_right_corner[0]]
    
    # Save the cropped image
    cv2.imwrite(destination_path, cropped_img)

    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    
    # Select your image destination and output paths here
    input_path = 'datasets/whole/dataset1/normal'
    output_path = 'tipdatasetcropped'
    
    # I'm using it to loop through a bunch of images and crop them all one by one
    frame = 0
    while True:
        image_path = f'{input_path}/frame{frame}.jpg'  # Replace this with your image path
        destination_path = f'{output_path}/frame{frame}.jpg'
        main(image_path, destination_path, f'frame{frame}')
        frame += 1