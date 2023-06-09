'''
This file contains functions for preprocessing images.

    - Add padding to square images to make them larger.
    - Resize square images.
    - Simplify images by emphasizing edges and reducing noise through color simplification based on pixel color variation.
        Before inputting image, make sure to make it black and white. For example, you can do:
        image = cv2.imread(image_path)[:,:,0]

'''

from PIL import Image, ImageOps
import numpy as np
import cv2

# Simply rescales images to a square of size x size pixels. Meant to work with square images.
def resize_image(input_path, output_path, size):
    with Image.open(input_path) as img:
        img_resized = img.resize((size, size))
        img_resized.save(output_path)
        
# Pads images to a square of size x size pixels. Meant to work with square images.
def pad_image(input_path, output_path, final_size):
    with Image.open(input_path) as img:
        width, height = img.size

        new_width = final_size
        new_height = final_size

        left_padding = (new_width - width) // 2
        top_padding = (new_height - height) // 2
        right_padding = new_width - width - left_padding
        bottom_padding = new_height - height - top_padding

        img_with_border = ImageOps.expand(img, (left_padding, top_padding, right_padding, bottom_padding), fill='black')
        img_with_border.save(output_path)

# The method simplifies the image by emphasizing edges and reducing noise through color simplification based on pixel color variation.
def simplify(img: np.ndarray):
    # Get standard deviation across all pixels in image
    x = np.std(img)
    
    # Compute the 'range' threshold based on the standard deviation
    rang = 2.04023 * x - 4.78237
    if(x < 20): rang = 5
    print(rang)
    
    # Find the maximum pixel intensity in the image
    whitestPixel = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel): whitestPixel = img[i][j]
            
    # Set all pixels with intensities greater than 'whitestPixel - range' to 255 (white)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel - rang): img[i][j] = 255
            
    # Set cutoff to '255 - range'
    cutoff = 255 - rang
    
    # Loop until all pixels have been categorized
    while(True):    
        whitestPixel = 0
        
        # Find the maximum pixel intensity that's less than 'cutoff' and greater than the most intense pixel in this range
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] < cutoff and img[i][j] > whitestPixel): whitestPixel = img[i][j]
                
        # Break the loop if no such pixel is found
        if whitestPixel == 0: break
        
        # Set all pixels with intensities greater than 'whitestPixel - range' and less than 'cutoff' to 'whitestPixel'
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] > whitestPixel - rang and img[i][j] < cutoff): img[i][j] = whitestPixel
                
        # Reduce 'cutoff' by 'range'
        cutoff = whitestPixel - rang
    return img


frame = 0
while True:
    print(frame)
    input_path = f'datasets/resize_datasets/normal/frame{frame}.jpg'
    output_path = f'whiten1_224/frame{frame}.jpg'
    
    image = cv2.imread(input_path)[:,:,0]
    image = simplify(image)
    cv2.imwrite(output_path, image)
    print()
    
    frame += 1

