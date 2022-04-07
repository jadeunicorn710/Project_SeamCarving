# Project_SeamCarving

==================================================================================================================
1. Before running the code, please ensure all the original images files are in the same folder with seam_carving.py and main.py
2. Activate the virtual environment first
    $ source activate CS6475
3. Please ensure numba library is installed
    $ conda install numba
4. It is run as follows:
    python main.py <method> <input_image> <output_image> <output_image_seams> width_ratio <energy_color>
==================================================================================================================

------------------------------------------------------------------------------------------------------------------
For the 2007 paper - (backward energy)
------------------------------------------------------------------------------------------------------------------

For instance, to reduce fig5.png to 50% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py backward fig5.png fig5result.png fig5result_seams.png 0.5 fig5_color.png

For instance, to enlarge fig8.png to 150% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py backward fig8.png fig8resultd.png fig8resultd_seams.png 1.5 fig8_color.png

To achieve part f in the 2007 paper, perform another seam insertion by using the following:
    python main.py backward fig8resultd.png fig8resultf.png fig8resultf_seams.png 1.5 fig8d_color.png

------------------------------------------------------------------------------------------------------------------
For the 2008 paper - (backward and forward engergy)
------------------------------------------------------------------------------------------------------------------

To use the backward energy

For instance, to reduce fig8-2008.png to 50% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py backward fig8-2008.png fig8-2008resultb.png fig8-2008resultb_seams.png 0.5 fig8-2008_color.png


For instance, to reduce fig9-2008.png to 50% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py backward fig9-2008.png fig9-2008resultrb.png fig9-2008resultrb_seams.png 0.5 fig9-2008_color.png


For instance, to enlarge fig9-2008.png to 150% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py backward fig9-2008.png fig9-2008resultb.png fig9-2008resultb_seams.png 1.5 fig9-2008_color.png


To use the forward energy

For instance, to reduce fig8-2008.png to 50% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py forward fig8-2008.png fig8-2008resultf.png fig8-2008resultf_seams.png 0.5 fig8-2008_color.png

For instance, to reduce fig9-2008.png to 50% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py forward fig9-2008.png fig9-2008resultrf.png fig9-2008resultrf_seams.png 0.5 fig9-2008_color.png

For instance, to enlarge fig9-2008.png to 150% and save the output image, the output image with seams, and the energy map,
use the following:
    python main.py forward fig9-2008.png fig9-2008resultf.png fig9-2008resultf_seams.png 1.5 fig9-2008_color.png





Thanks!
Yunlin Qi
