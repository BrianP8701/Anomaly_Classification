'''
    This file contains functions for preprocessing images.

    1. Add padding to square images to make them larger.
    2. Resize square images.
    3. Simplify images by emphasizing edges and reducing noise through color simplification based on pixel color variation.
'''

from PIL import Image, ImageOps, ImageFilter
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
def simplify(img: str):
    img = cv2.imread(input_path)[:,:,0]
    # Get standard deviation across all pixels in image
    x = np.std(img)
    print(x)
    
    # Compute the 'range' threshold based on the standard deviation
    threshold = 2.04023 * x - 4.78237
    print(threshold)
    if(x < 20): threshold = 5
    print(threshold)
    threshold = 5
    
    # Find the maximum pixel intensity in the image
    whitestPixel = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel): whitestPixel = img[i][j]
            
    # Set all pixels with intensities greater than 'whitestPixel - range' to 255 (white)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel - threshold): img[i][j] = 255
            
    # Cutoff is the minimum pixel intensity we have already simplified
    cutoff = 255 - threshold
    
    # Loop until all pixels have been categorized
    while(True):    
        whitestPixel = 0
        
        # Find the maximum pixel intensity that's less than 'cutoff'
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] < cutoff and img[i][j] > whitestPixel): whitestPixel = img[i][j]
                
        # Break the loop if no such pixel is found
        if whitestPixel == 0: break
        
        # Set all pixels with intensities greater than 'whitestPixel - range' and less than 'cutoff' to 'whitestPixel'
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] > whitestPixel - threshold and img[i][j] < cutoff): img[i][j] = whitestPixel
                
        # Update cutoff
        cutoff = whitestPixel - threshold
    return img

def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

    # Denoise
    img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert to PIL Image for edge enhancement
    #img = Image.fromarray(img)

    # Sharpen edges using Unsharp Mask
    #img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=1000))

    return np.array(img)


frame = 0
while True:
    input_path = f'whiten_testing/frame{frame}.jpg'
    output_path = f'whiten_output/frame{frame}.jpg'
    
    img = preprocess_image(input_path)
    cv2.imwrite(output_path, img)
    
    frame += 1

