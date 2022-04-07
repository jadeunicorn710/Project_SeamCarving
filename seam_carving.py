import cv2
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import njit


# ------------------------------------------------------------------------
#    Seam carving using Backward Energy method
# ------------------------------------------------------------------------


# *********************
#    Seam Removal
# *********************

# Extract the color channels of the image
def extract_channels(image):
    # Modify array shape from (row, col, channel) to (channel, row, col)
    channels = blue, green, red = cv2.split(image)
    return channels


# Compute the first x- and y- image derivatives using Scharr operator
def image_derivative(image):
    dx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    return dx, dy


# Define the energy function for each pixel in the image
# Sum the absolute value of image derivatives in all three channels
def energy_function (image):
    # Extract the image channels
    channels = extract_channels(image)
    # Get the pixel energy for each channel
    blue_dx, blue_dy = image_derivative(channels[0])
    blue_energy = np.abs(blue_dx) + np.abs(blue_dy)
    red_dx, red_dy = image_derivative(channels[1])
    red_energy = np.abs(red_dx) + np.abs(red_dy)
    green_dx, green_dy = image_derivative(channels[2])
    green_energy = np.abs(green_dx) + np.abs(green_dy)
    # Get the energy map
    energy_map = blue_energy + red_energy + green_energy
    return energy_map



# Determine the cumulative minimum energy (or the minimum path cost) M
# Use Numba to improve runtime
@njit
def min_path_cost(energy_map):
    # Initialize the cumulative minimum energy array M
    M = energy_map[:, :]
    # Traverse from second row to last row
    row = energy_map.shape[0]
    col = energy_map.shape[1]
    for i in range(1, row):
        for j in range(0, col):
            # Determine the cumulative minimum energy till last row
            # Consider the left edge and right edge first
            if j == 0:
                last_min_energy = np.amin(M[i - 1, j:j + 2])

            elif j == (col - 1):
                last_min_energy = np.amin(M[i - 1, j - 1:j + 1])

            # Next consider the interior pixels
            else:
                last_min_energy = np.amin(M[i - 1, j - 1:j + 2])

            # Add the last cumulative minimum energy to current energy of the pixel to get the minimum path cost at the pixel
            M[i, j] = energy_map[i, j] + last_min_energy

    return M



# Backtrack from the minimum entry in the last row of M to find the optimal seam
# Use Numba to improve runtime
@njit
def backtrack(M):
    # Locate the minimum entry in the last row of M to start the optimal seam path
    j = np.argmin(M[-1])
    # Define an empty array to record location of optimal seam in each row
    index = np.zeros((M.shape[0],))
    # Store j as last entry in the array as starting point
    index[-1] = j
    # Traverse from last row to top
    row = M.shape[0]
    col = M.shape[1]
    for i in range(row - 1, 0, -1):
        # Consider the left edge and right edge first
        if j == 0:
            index[i - 1] = j + np.argmin(M[i - 1, j: j + 2])
        elif j == (col - 1):
            index[i - 1] = j - 1 + np.argmin(M[i - 1, j - 1: j + 1])
        # Next consider the interior pixels
        else:
            index[i - 1] = j - 1 + np.argmin(M[i - 1, j - 1: j + 2])
        # Update index j to the row above
        j = index[i - 1]
    return index




# Remove the single optimal seam from image and get the reduced image
# Use Numba to improve runtime
@njit
def remove_seam(image, index):
    # Define an empty array to take the reduced image after removing the seam
    image_reduced = np.zeros((image.shape[0], image.shape[1] - 1, 3))

    # Traverse from top row to bottom to remove seam pixels in the image
    row = image.shape[0]
    col = image.shape[1]
    for i in range(0, row):
        # Get the optimal seam index in each row
        j = int(index[i])
        # Get the reduced image for each channel
        for channel in range(0, 3):
            # image_reduced[i, :, channel] = np.delete(image[i, :, channel], [j])
            image_reduced[i, : j, channel] = image[i, : j, channel]
            image_reduced[i, j:, channel] = image[i, j + 1:, channel]

    # Update the reduced image
    image = image_reduced[:, :, :]
    return image



# Repeat the above seam removing process until the retargeted image width
def remove_seams(image, width_ratio):
    # Duplicate the image to process seam removal
    image_reduced = np.copy(image)
    image_seam = np.copy(image)
    # Define an empty list to store the optimal seams
    optimal_seams = []
    # Determine the new image width
    col_new = int((image.shape[1]) * float(width_ratio))
    # Loop until new row width is satisfied
    loop = image.shape[1] - col_new
    for i in range(loop):
        # Get the energy map
        energy_map = energy_function(image_reduced)
        # Get the cumulative minimum energy array
        M = min_path_cost(energy_map)
        # Get the index array for optimal seam
        index = backtrack(M)
        # Record the optimal seam in order before removal
        optimal_seams.append(index)
        # Remove optimal seam
        image_reduced = remove_seam(image_reduced, index)

    # Loop again to get the image with seams in red color
    # Define an array for red color BGR pixel [0, 1, 255]
    redpixel = np.array([0, 1, 255]).astype(np.float64)

    for i in range(loop):
        # Get the first optimal seam in the list, then remove it from the seam list
        optimal_seam = optimal_seams.pop(0)

        for i in range(0, image.shape[0]):
            # Get the optimal seam index in each row
            j = int(optimal_seam[i])
            # Get the image with red seams for each channel
            for channel in range(0, 3):
                image_seam[i, j, channel] = redpixel[channel]

        # Update the indices of next seam in the optimal seams array
        # to accommodate for changes from reduced image

        # Determine if the remaining optimal seams array is empty
        if len(optimal_seams) > 0:
            # Loop through each seam
            for next_seam in optimal_seams:
                # Loop through each indices in the next seam and compare with corresponding indices in the last seam
                for k in range(len(next_seam)):
                    # If the index in the next seam was on the right hand side of that in the last seam,
                    # need to shift the index by 1 to the right, to account for image reduction
                    if next_seam[k] >= optimal_seam[k]:
                        next_seam[k] = next_seam[k] + 1

    # Update the reduced image
    image = image_reduced[:, :, :]
    image_seams = image_seam[:, :, :]
    return image, image_seams


# *********************
#     Seam Insertion
# *********************

# Add the single optimal seam to get the enlarged image
# Use Numba to improve runtime
@njit
def add_seam(image, image_seams, index):
    # Define an empty array to take the enlarged image after adding the seam
    image_enlarged = np.zeros((image.shape[0], image.shape[1] + 1, 3))
    # Make a duplicate to show the inserted seam in red color
    image_seam = np.copy(image_enlarged)
    # Make a duplicate of the input image for processing the image with seam
    image_duplicate = np.copy(image)
    # Define an array for red color BGR pixel [0, 1, 255]
    redpixel = np.array([0, 1, 255]).astype(np.float64)
    # Traverse from top row to bottom to insert seam pixels in the image at the index location
    # Shift the original pixels at the right hand side of index (inclusive) to the right by one
    row = image.shape[0]
    col = image.shape[1]
    for i in range(0, row):
        # Get the index of the optimal seam in each row
        j = int(index[i])
        # Get the enlarged image for each channel
        for channel in range(3):
            # Get the 'artificial' pixel by averaging it with its left and right neighbors
            # Consider the left edge and right edge first
            if j == 0:
                # Determine the 'artificial' pixel
                artificial_pixel = np.sum(image[i, j: j + 2, channel]) / 2
                # Insert the 'artificial' pixel at the index location of the row
                # image_enlarged[i, : , channel] = np.insert(image[i, :, channel], [j], [artificial_pixel])
                image_enlarged[i, 0, channel] = artificial_pixel
                image_enlarged[i, 1:, channel] = image[i, :, channel]
                # Similarly update the enlarged image with seam in red color
                image_seam[i, 0, channel] = redpixel[channel]
                image_seam[i, 1:, channel] = image_seams[i, :, channel]
            elif j == (col - 1):
                # Determine the 'artificial' pixel
                artificial_pixel = np.sum(image[i, j - 1: j + 1, channel]) / 2
                # Insert the 'artificial' pixel at the index location of the row
                # image_enlarged[i, :, channel] = np.insert(image[i, :, channel], [j], [artificial_pixel])
                image_enlarged[i, : j, channel] = image[i, : j, channel]
                image_enlarged[i, j, channel] = artificial_pixel
                image_enlarged[i, j + 1, channel] = image[i, j, channel]
                # Similarly update the enlarged image with seam in red color
                image_seam[i, : j, channel] = image_seams[i, : j, channel]
                image_seam[i, j, channel] = redpixel[channel]
                image_seam[i, j + 1, channel] = image_seams[i, j, channel]
            # Next consider interior cases
            else:
                # Determine the 'artificial' pixel
                artificial_pixel = np.sum(image[i, j - 1: j + 2, channel]) / 3
                # Insert the 'artificial' pixel at the index location of the row
                # image_enlarged[i, :, channel] = np.insert(image[i, :, channel], [j], [artificial_pixel])
                image_enlarged[i, : j, channel] = image[i, : j, channel]
                image_enlarged[i, j, channel] = artificial_pixel
                image_enlarged[i, j + 1:, channel] = image[i, j:, channel]
                # Similarly update the enlarged image with seam in red color
                image_seam[i, : j, channel] = image_seams[i, : j, channel]
                image_seam[i, j, channel] = redpixel[channel]
                image_seam[i, j + 1:, channel] = image_seams[i, j:, channel]

    # Update the enlarged image
    image = image_enlarged[:, :, :]
    image_seams = image_seam[:, : , :]
    return image, image_seams



# Repeat the seam addition process until the retargeted image width
# Use Numba to improve runtime
# @njit
def add_seams(image, width_ratio):
    # Determine the new image width
    col_new = int((image.shape[1]) * float(width_ratio))
    # Determine the number of loops to process seam removal and seam addition
    loop = col_new - image.shape[1]
    # Duplicate the image to process seam removal and get the seam locations
    image_reduced = np.copy(image)
    # Define an empty list to store the optimal seams
    optimal_seams = []
    # Loop through the duplicated image to remove seams first
    for i in range(loop):
        # Get the energy map
        energy_map = energy_function(image_reduced)
        # Get the cumulative minimum energy array
        M = min_path_cost(energy_map)
        # Get the index array for optimal seam
        index = backtrack(M)
        # Record the optimal seam in order before removal
        optimal_seams.append(index)
        # Remove optimal seam
        image_reduced = remove_seam(image_reduced, index)

    # Duplicate the image again to process seam addition
    image_enlarged = np.copy(image)
    image_seam = np.copy(image)
    # Loop and add new seams back to the enlarged image, in the sequence of seam removal
    for i in range(loop):
        # Get the first optimal seam in the list for seam addition, then remove it from the seam list
        optimal_seam = optimal_seams.pop(0)
        image_enlarged, image_seam = add_seam(image_enlarged, image_seam, optimal_seam)
        # Update the indices of next seam in the optimal seams array
        # to accommodate for changes from reduced image to enlarged image

        # Determine if the remaining optimal seams array is empty
        if len(optimal_seams) > 0:
            # Loop through each seam
            for next_seam in optimal_seams:
                # Loop through each indices in the next seam and compare with corresponding indices in the last seam
                for k in range(len(next_seam)):
                    # If the index in the next seam was on the right hand side of that in the last seam,
                    # need to shift the index by 2 to the right, 1 to account for image reduction, 1 to account for image
                    # enlarging
                    if next_seam[k] >= optimal_seam[k]:
                        next_seam[k] = next_seam[k] + 2


    # Update the enlarged images
    image = image_enlarged[:, :, :]
    image_seams = image_seam[:, :, :]
    return image, image_seams


# Define a function to determine either to perform seam removal or seam insertion,
def seam_processing(image, width_ratio):
    if float(width_ratio) < 1:
        # Perform seam removal
        image, image_seams = remove_seams(image, width_ratio)
    elif float(width_ratio) > 1:
        # Perform seam insertion
        image, image_seams = add_seams(image, width_ratio)
    return image, image_seams



# ------------------------------------------------------------------------
#    Seam carving using Forward Energy method
# ------------------------------------------------------------------------



# Define the cost for each of the three possible cases
def cost_function(image):
    # Define the image filtering kernels (3-tap) to perform linear operations to image pixels as required in the three costs
    # Based on I(i, j), three 3-tap kernels centered on it will be required to perform the operations in the three costs

    # (1). I(i, j + 1) - I(i, j - 1):
    filter1 = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
    # (2). I(i - 1, j) - I(i, j - 1):
    filter2 = np.array([[0., 1., 0.], [-1., 0., -0.], [0., 0., 0.]], dtype=np.float64)
    # (3). I(i - 1, j) - I(i, j + 1):
    filter3 = np.array([[0., 1., 0.], [0., 0., -1.], [0., 0., 0.]], dtype=np.float64)

    # Extract the image channels
    channels = extract_channels(image)
    # Define the three absolute elements that constitutes the three costs
    # Use cv2.filter2D assuming padding type BORDER_REFLECT101
    # Determine the first cost element
    element1 = np.absolute(cv2.filter2D(channels[0], -1, filter1, cv2.BORDER_REFLECT101)) + \
               np.absolute(cv2.filter2D(channels[1], -1, filter1, cv2.BORDER_REFLECT101)) + \
               np.absolute(cv2.filter2D(channels[2], -1, filter1, cv2.BORDER_REFLECT101))

    # Determine the second cost element
    element2 = np.absolute(cv2.filter2D(channels[0], -1, filter2, cv2.BORDER_REFLECT101)) + \
               np.absolute(cv2.filter2D(channels[1], -1, filter2, cv2.BORDER_REFLECT101)) + \
               np.absolute(cv2.filter2D(channels[2], -1, filter2, cv2.BORDER_REFLECT101))

    # Determine the third cost element
    element3 = np.absolute(cv2.filter2D(channels[0], -1, filter3, cv2.BORDER_REFLECT101)) + \
               np.absolute(cv2.filter2D(channels[1], -1, filter3, cv2.BORDER_REFLECT101)) + \
               np.absolute(cv2.filter2D(channels[2], -1, filter3, cv2.BORDER_REFLECT101))

    # Determine the three cost arrays
    # Cl
    Cl = element1 + element2
    # Cu
    Cu = element1
    # Cr
    Cr = element1 + element3

    return Cl, Cu, Cr

# Define the new accumulative cost matrix M
# Use Numba to improve runtime
@njit
def accumu_cost_forward(energy_map, Cl, Cu, Cr):
    # Initialize the cumulative minimum energy array M
    M = energy_map[:, :]
    row = energy_map.shape[0]
    col = energy_map.shape[1]

    # Determine the first row first
    for j in range(0, col):
        Mup = Cu[0, j]
        M[0, j] = energy_map[0, j] + Mup


    # Traverse from second row to last row

    for i in range(1, row):
        for j in range(0, col):
            # Determine the new accumulative cost matrix till last row
            # For each row, consider 3 cases from the last row
            # Consider the left edge and right edge first
            if j == 0:
                Mup = M[i - 1, j] + Cu[i, j]
                Mur = M[i - 1, j + 1] + Cr[i, j]
                M[i, j] = energy_map[i, j] + min(Mup, Mur)
            elif j == col -1:
                Mul = M[i - 1, j - 1] + Cl[i, j]
                Mup = M[i - 1, j] + Cu[i, j]
                M[i, j] = energy_map[i, j] + min(Mul, Mup)
            # Next consider the interior pixels
            else:
                Mul = M[i - 1, j - 1] + Cl[i, j]
                Mup = M[i - 1, j] + Cu[i, j]
                Mur = M[i - 1, j + 1] + Cr[i, j]
                M[i, j] = energy_map[i, j] + min(Mul, Mup, Mur)

    return M


# *********************
#     Seam Removal
# *********************


def remove_seams_forward(image, width_ratio):
    # Duplicate the image to process seam removal
    image_reduced = np.copy(image)
    image_seam = np.copy(image)
    # Define an empty list to store the optimal seams
    optimal_seams = []
    # Determine the new image width
    col_new = int((image.shape[1]) * float(width_ratio))
    # Loop until new row width is satisfied
    loop = image.shape[1] - col_new
    for i in range(loop):
        # Get the energy map
        energy_map = energy_function(image_reduced)
        # Get the three cost arrays
        Cl, Cu, Cr = cost_function(image_reduced)
        # Get the new  accumulative cost matrix M
        M = accumu_cost_forward(energy_map, Cl, Cu, Cr)
        # Get the index array for optimal seam
        index = backtrack(M)
        # Record the optimal seam in order before removal
        optimal_seams.append(index)
        # Remove optimal seam
        image_reduced = remove_seam(image_reduced, index)

    # Loop again to get the image with seams in red color
    # Define an array for red color BGR pixel [0, 1, 255]
    redpixel = np.array([0, 1, 255]).astype(np.float64)

    for i in range(loop):
        # Get the first optimal seam in the list, then remove it from the seam list
        optimal_seam = optimal_seams.pop(0)

        for i in range(0, image.shape[0]):
            # Get the optimal seam index in each row
            j = int(optimal_seam[i])
            # Get the image with red seams for each channel
            for channel in range(0, 3):
                image_seam[i, j, channel] = redpixel[channel]

        # Update the indices of next seam in the optimal seams array
        # to accommodate for changes from reduced image

        # Determine if the remaining optimal seams array is empty
        if len(optimal_seams) > 0:
            # Loop through each seam
            for next_seam in optimal_seams:
                # Loop through each indices in the next seam and compare with corresponding indices in the last seam
                for k in range(len(next_seam)):
                    # If the index in the next seam was on the right hand side of that in the last seam,
                    # need to shift the index by 1 to the right, to account for image reduction
                    if next_seam[k] >= optimal_seam[k]:
                        next_seam[k] = next_seam[k] + 1

    # Update the reduced image
    image = image_reduced[:, :, :]
    image_seams = image_seam[:, :, :]
    return image, image_seams





# *********************
#     Seam Insertion
# *********************


# Determine the new add seams function to repeat the seam addition process until the retargeted image width
# Use Numba to improve runtime
# @njit
def add_seams_forward(image, width_ratio):
    # Determine the new image width
    col_new = int((image.shape[1]) * float(width_ratio))
    # Determine the number of loops to process seam removal and seam addition
    loop = col_new - image.shape[1]
    # Duplicate the image to process seam removal and get the seam locations
    image_reduced = np.copy(image)

    # Define an empty list to store the optimal seams
    optimal_seams = []
    # Loop through the duplicated image to remove seams first
    for i in range(loop):
        # Get the energy map
        energy_map = energy_function(image_reduced)
        # Get the three cost arrays
        Cl, Cu, Cr = cost_function(image_reduced)
        # Get the new  accumulative cost matrix M
        M = accumu_cost_forward(energy_map, Cl, Cu, Cr)
        # Get the index array for optimal seam
        index = backtrack(M)
        # Record the optimal seam in order before removal
        optimal_seams.append(index)
        # Remove optimal seam
        image_reduced = remove_seam(image_reduced, index)

    # Duplicate the image again to process seam addition
    image_enlarged = np.copy(image)
    image_seam = np.copy(image)
    # Loop and add new seams back to the enlarged image, in the sequence of seam removal
    for i in range(loop):
        # Get the first optimal seam in the list for seam addition, then remove it from the seam list
        optimal_seam = optimal_seams.pop(0)
        image_enlarged, image_seam = add_seam(image_enlarged, image_seam, optimal_seam)
        # Update the indices of next seam in the optimal seams array
        # to accommodate for changes from reduced image to enlarged image

        # Determine if the remaining optimal seams array is empty
        if len(optimal_seams) > 0:
            # Loop through each seam
            for next_seam in optimal_seams:
                # Loop through each indices in the next seam and compare with corresponding indices in the last seam
                for k in range(len(next_seam)):
                    # If the index in the next seam was on the right hand side of that in the last seam,
                    # need to shift the index by 2 to the right, 1 to account for image reduction, 1 to account for image
                    # enlarging
                    if next_seam[k] >= optimal_seam[k]:
                        next_seam[k] = next_seam[k] + 2


    # Update the enlarged images
    image = image_enlarged[:, :, :]
    image_seams = image_seam[:, :, :]
    return image, image_seams



# Define a similar function to determine either to perform seam removal or seam insertion using the forward energy,
def seam_processing_forward(image, width_ratio):
    if float(width_ratio) < 1:
        # Perform seam removal
        image, image_seams = remove_seams_forward(image, width_ratio)
    elif float(width_ratio) > 1:
        # Perform seam insertion
        image, image_seams = add_seams_forward(image, width_ratio)
    return image, image_seams