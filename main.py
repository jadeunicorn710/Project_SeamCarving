import sys
import cv2
from seam_carving import *

# ==================================================================================================================
# All the original images files need to be in the same folder with seam_carving.py and main.py
# Activate the virtual environment first
#     $ source activate CS6475
# It is run as follows:
#     python main.py <method> <input_image> <output_image> <output_image_seams> width_ratio <energy_color>
# ==================================================================================================================

# ------------------------------------------------------------------------------------------------------------------
# For the 2007 paper - (backward energy)
# ------------------------------------------------------------------------------------------------------------------

# For instance, to reduce fig5.png to 50% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py backward fig5.png fig5result.png fig5result_seams.png 0.5 fig5_color.png

# For instance, to enlarge fig8.png to 150% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py backward fig8.png fig8resultd.png fig8resultd_seams.png 1.5 fig8_color.png

# To achieve part f in the 2007 paper, perform another seam insertion by using the following:
#     python main.py backward fig8resultd.png fig8resultf.png fig8resultf_seams.png 1.5 fig8d_color.png

# ------------------------------------------------------------------------------------------------------------------
# For the 2008 paper - (backward and forward engergy)
# ------------------------------------------------------------------------------------------------------------------

# To use the backward energy

# For instance, to reduce fig8-2008.png to 50% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py backward fig8-2008.png fig8-2008resultb.png fig8-2008resultb_seams.png 0.5 fig8-2008_color.png


# For instance, to reduce fig9-2008.png to 50% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py backward fig9-2008.png fig9-2008resultrb.png fig9-2008resultrb_seams.png 0.5 fig9-2008_color.png


# For instance, to enlarge fig9-2008.png to 150% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py backward fig9-2008.png fig9-2008resultb.png fig9-2008resultb_seams.png 1.5 fig9-2008_color.png


# To use the forward energy

# For instance, to reduce fig8-2008.png to 50% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py forward fig8-2008.png fig8-2008resultf.png fig8-2008resultf_seams.png 0.5 fig8-2008_color.png

# For instance, to reduce fig9-2008.png to 50% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py forward fig9-2008.png fig9-2008resultrf.png fig9-2008resultrf_seams.png 0.5 fig9-2008_color.png

# For instance, to enlarge fig9-2008.png to 150% and save the output image, the output image with seams, and the energy map,
# use the following:
#     python main.py forward fig9-2008.png fig9-2008resultf.png fig9-2008resultf_seams.png 1.5 fig9-2008_color.png

if len(sys.argv) != 7:
    print ("Syntax:")
    print ("    python main.py <method> <input_image> <output_image> width_ratio <energy_color>")
    exit()


method = sys.argv[1]
input_image = sys.argv[2]
output_image = sys.argv[3]
output_image_seams = sys.argv[4]
width_ratio = sys.argv[5]
energy_color = sys.argv[6]

image_original = cv2.imread(input_image)
# Make a copy of the original image for processing
image = np.copy(image_original.astype(np.float64))
if method == 'forward':
    result, result_seams = seam_processing_forward(image, float(width_ratio))
elif method =='backward':
    result, result_seams = seam_processing(image, float(width_ratio))
else:
    print('Please specify which method to use for seam carving')

# Save the result files
cv2.imwrite(output_image, result)
cv2.imwrite(output_image_seams, result_seams)

energy_map = energy_function(image)
# Define a function to plot the energy map
def color_map(energy_map):
    cv2.imwrite(energy_color, energy_map)
    energy_color_map = cv2.imread(energy_color, cv2.CV_8UC1)
    energy_color_map = cv2.applyColorMap(energy_color_map, cv2.COLORMAP_JET)
    return energy_color_map

# Save the energy color map to file
energy_color_map = color_map(energy_map)
cv2.imwrite(energy_color, energy_color_map)




