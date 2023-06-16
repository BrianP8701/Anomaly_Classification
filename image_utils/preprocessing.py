'''
    This file contains functions for preprocessing images.

    1. Add padding to square images to make them larger.
    2. Resize square images.
    3. Simplify images by emphasizing edges and reducing noise through color simplification based on pixel color variation.
'''

from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time

def gmms_preprocess_image(img_path, num_components):
    data = cv2.imread(img_path)[:,:,0]
    original_shape = data.shape
    data = flatten(data)
    
    # Get means and variances assuming 3 clusters
    means, stdvs = gmm_parameters(data, 3)    

    # Get information of the cluster with the highest mean
    means = np.sort(means)
    stdvs = np.sqrt(np.sort(stdvs))
    max_mean = means[-1]
    max_stdv = stdvs[-1] * 2
    if(max_stdv > 40): max_stdv = 50
    
    # This is the range of intensities we believe make up the recently extruded material
    material_range = (max_mean - max_stdv, max_mean + max_stdv)
    material_indices = get_indices_within_range(data, material_range)
    material_data = data[material_indices]
    
    # Split the material data into num_components clusters
    material_means, material_stdvs = gmm_parameters(material_data, num_components)
    material_means = np.sort(material_means)
    material_stdvs = np.sqrt(np.sort(material_stdvs))
    
    # Combine information from first cluster, and the material clusters
    total_means = np.insert(material_means, 0, means[0])
    total_stdvs = np.insert(material_stdvs, 0, stdvs[0])
        
    # Given the means and standard deviations, split the range 0-255 into 4 intervals
    ranges = get_ranges(total_means, total_stdvs)
    
    # Create a list of colors to use for each range
    colors = []
    for i in range(num_components+1):
        colors.append((255/(num_components+1))*i)
    
    # Replace all values in the data that fall within the ranges with the corresponding color
    for i in range(num_components+1):
        data = replace_values_within_range(data, ranges[i], colors[i])
    
    data = unflatten(data, original_shape)
    return data
    
def gmm_preprocess_image(img_path, num_components):
    data = cv2.imread(img_path)[:,:,0]
    original_shape = data.shape
    data = flatten(data)
    
    # Get means and variances assuming num_components clusters
    means, stdvs = gmm_parameters(data, num_components)    
    means = np.sort(means)
    stdvs = np.sqrt(np.sort(stdvs))
    
    if(means[2] - means[0] > 80):
        black_range = np.array([0, means[0] + 60])
    
    # Given the means and standard deviations, split the range 0-255 into num_components intervals
    ranges = get_ranges(means, stdvs)
    
    # Create a list of colors to use for each range
    colors = []
    for i in range(num_components):
        colors.append((255/num_components)*i)
        
    # Replace all values in the data that fall within the ranges with the corresponding color
    for i in range(num_components):
        data = replace_values_within_range(data, ranges[i], colors[i])
    
    data = unflatten(data, original_shape)
    return data
    
def get_ranges(means, std_devs):
    # Calculate the raw ranges
    raw_ranges = list(zip(means - std_devs, means + std_devs))
    
    # Fix overlaps and extend to 0 and 255
    ranges = []
    for i, (low, high) in enumerate(raw_ranges):
        # If this is the first range, extend the lower bound to 0
        if i == 0:
            low = 0
        # Otherwise, adjust the lower bound to be the midpoint between this range's lower bound and the previous range's upper bound
        else:
            low = (low + ranges[-1][1]) / 2

        # If this is the last range, extend the upper bound to 255
        if i == len(raw_ranges) - 1:
            high = 255
        # Otherwise, adjust the upper bound to be the midpoint between this range's upper bound and the next range's lower bound
        else:
            high = (high + raw_ranges[i+1][0]) / 2
        
        ranges.append((low, high))
    
    return ranges
    
# Return a list of indices in the array that fall within the provided range.
def get_indices_within_range(array, range):
    lower, upper = range
    indices = np.where((array >= lower) & (array <= upper))
    return indices[0].tolist()

# Replace all values in the array that fall within the provided range with the given number.
def replace_values_within_range(array, range, replacement_value):
    lower, upper = range
    array[(array >= lower) & (array <= upper)] = replacement_value
    return array

'''
Gaussian Mixture Model

    Params: 
        img_path: path to image
        num_components: number of clusters to fit
        
    Returns:
        means: list of means for each cluster
        variances: list of variances for each cluster
'''
def gmm_parameters(data, num_components):
    # Reshape data to fit the GMM input requirements (should be 2D)
    data = data.reshape(-1, 1)

    # Initialize Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_components, random_state=0)

    # Fit the GMM to the data
    gmm.fit(data)

    # Extract means and variances
    means = gmm.means_.flatten()  # Flatten to convert to 1D
    variances = gmm.covariances_.flatten()

    # Return the parameters as a list
    return list(means), list(variances)


# Given an image path and number of clusters, returns cluster assignments and cluster centers
def perform_kmeans(data, k):
    # Reshape the data to the shape that scikit-learn expects
    data = data.reshape(-1, 1)

    # Initialize and fit a KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

    # Get the cluster assignments for each data point
    cluster_assignments = kmeans.predict(data)

    # Get the cluster centers (mean of each cluster)
    cluster_centers = kmeans.cluster_centers_

    return cluster_assignments, cluster_centers

'''
Creates a bar chart with the pixel intensities on the x-axis and the counts on the y-axis.
    
    Params: 
        img_path: path to image
'''
def create_bar_chart(img_path):
    data = cv2.imread(img_path)[:,:,0]
    data = flatten(data)
    
    # Create a list of zeros with a length of 256 (for each possible pixel intensity)
    pixel_counts = [0]*256

    # Iterate through the data and increment the count for each pixel intensity
    for pixel in data:
        pixel_counts[int(pixel)] += 1

    # Create the bar chart
    plt.bar(range(256), pixel_counts, color='gray')
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.show()
    
'''
Creates a bar chart with the pixel intensities on the x-axis and the counts on the y-axis.
    - Adds a red line at means
    - Adds blue lines at means +/- standard deviations
    
    Params: 
        img_path: path to image
        num_components: number of clusters to fit
'''
def create_bar_chart_with_gmm(img_path, num_components):
    data = cv2.imread(img_path)[:,:,0]
    data = flatten(data)
    
    # Create a list of zeros with a length of 256 (for each possible pixel intensity)
    pixel_counts = [0]*256

    # Iterate through the data and increment the count for each pixel intensity
    for pixel in data:
        pixel_counts[int(pixel)] += 1

    # Create the bar chart
    plt.bar(range(256), pixel_counts, color='gray')

    # Fit a GMM to the data and get the means and variances
    means, variances = gmm_parameters(data, num_components)

    # Convert variances to standard deviations
    std_devs = [np.sqrt(variance) for variance in variances]

    # Add lines for the means and standard deviations
    for mean, std_dev in zip(means, std_devs):
        plt.axvline(x=mean, color='red')  # Mean
        plt.axvline(x=mean - std_dev, color='blue')  # One standard deviation below the mean
        plt.axvline(x=mean + std_dev, color='blue')  # One standard deviation above the mean

    # Add labels and show the plot
    plt.title('Pixel Intensity Distribution with GMM components')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.show()


# Turns 2d matrix into 1d vector
def flatten(matrix):
    # Use numpy's ravel function to flatten the matrix
    return matrix.ravel()

# Turns 1d vector into 2d matrix
def unflatten(vector, original_shape):
    # Use numpy's reshape function to convert the vector back to the original matrix shape
    return vector.reshape(original_shape)

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
def simplify(input_path: str):
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

# Given an image path, removes noise and emphasizes edges. Returns the processed image as a numpy array.
def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

    # Denoise
    img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert to PIL Image for edge enhancement
    img = Image.fromarray(img)

    # Sharpen edges using Unsharp Mask
    img = img.filter(ImageFilter.UnsharpMask(radius=8, percent=100))

    return np.array(img)

frame = 0
while True:
    print(f'Frame: {frame}')
    print()
    print()
    img = gmms_preprocess_image(f'whiten/whiten_test/frame{frame}.jpg', 4)
    cv2.imwrite(f'whiten/gmms4/frame{frame}.jpg', img)
    img = gmms_preprocess_image(f'whiten/whiten_test/frame{frame}.jpg', 6)
    cv2.imwrite(f'whiten/gmms6/frame{frame}.jpg', img)
    img = gmms_preprocess_image(f'whiten/whiten_test/frame{frame}.jpg', 8)
    cv2.imwrite(f'whiten/gmms8/frame{frame}.jpg', img)
    frame += 1




