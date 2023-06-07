'''
This file contains functions for preprocessing images.

    - Add padding to square images to make them larger.
    - Resize square images.
'''

from PIL import Image, ImageOps

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


frame = 0
size = 224  # size of the output image in pixels
while True:
    input_path = f'bubble_datasets/bubble/frame{frame}.jpg'
    output_path = f'bubble_resize_datasets/bubble/frame{frame}.jpg'
    resize_image(input_path, output_path, size)
    frame += 1


