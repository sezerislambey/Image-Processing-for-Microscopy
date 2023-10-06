## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import color, data, filters, util



## --------- Thresholding and segmentation ---------

# Let's work with images having different kinds of histograms
img_cell = util.img_as_float(data.cell())
img_mitosis = util.img_as_float(data.human_mitosis())
img_retina = util.img_as_float(color.rgb2gray(data.retina()))
img_nuc30_cells3d = util.img_as_float(data.cells3d()[29, 1, :, :])


# Consider the first image of a single cell.... Let's identify the two parts, below ~0.3 and above ~0.3
img_cell_below = img_cell * (img_cell < 0.4)
img_cell_above = img_cell * (img_cell > 0.4)

plt.figure(figsize = (24,6))
plt.gcf().suptitle('Clearly the two parts of the histogram indicate foreground and background!', size = 24, color = 'red')

plt.subplot(141)
plt.gca().set_title('Original image')
plt.imshow(img_cell, cmap = 'nipy_spectral', vmin = 0, vmax = 1)
plt.colorbar()

plt.subplot(142)
plt.hist(img_cell.flatten(), bins = 64, density = True, log = True)
plt.axvline(0.4, color = 'red')

plt.subplot(143)
plt.gca().set_title('Portion below 0.4 brightness')
plt.imshow(img_cell_below, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(144)
plt.gca().set_title('Portion above 0.4 brightness')
plt.imshow(img_cell_above, cmap = 'nipy_spectral', vmin = 0, vmax = 1)


# Likewise, what's in each of the two humps of the histogram in the retinal image?
img_retina_below = img_retina * (img_retina < 0.45) * (img_retina > 0.25)
img_retina_above = img_retina * (img_retina > 0.7)

plt.figure(figsize = (24,6))

plt.subplot(141)
plt.gca().set_title('Original image')
plt.imshow(img_retina, cmap = 'nipy_spectral', vmin = 0, vmax = 1)
plt.colorbar()

plt.subplot(142)
plt.hist(img_retina.flatten(), bins = 64, log = True, density = True)
import matplotlib.patches as pat
plt.gca().add_patch(pat.Rectangle((0.25, 0), 0.2, 10, color = 'red', alpha = 0.5))
plt.axvline(0.7, color = 'blue')

plt.subplot(143)
plt.gca().set_title('Portion in red rectangle peak')
plt.imshow(img_retina_below, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(144)
plt.gca().set_title('Portion in brightest peak')
plt.imshow(img_retina_above, cmap = 'nipy_spectral', vmin = 0, vmax = 1)


# Similarly, how would you separate the nuclei from the background in the mitotic cells image?
# What about the brightest dividing nuclei?
img_mitosis_nuclei = img_mitosis * (img_mitosis > 0.15)
img_mitosis_divs = img_mitosis * (img_mitosis > 0.5)

plt.figure(figsize = (24,6))

plt.subplot(141)
plt.gca().set_title('Original image')
plt.imshow(img_mitosis, cmap = 'nipy_spectral', vmin = 0, vmax = 1)
plt.colorbar()

plt.subplot(142)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = True)
plt.axvline(0.15, color = 'red')
plt.axvline(0.5, color = 'blue')

plt.subplot(143)
plt.gca().set_title('Nuclei')
plt.imshow(img_mitosis_nuclei, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(144)
plt.gca().set_title('Dividers')
plt.imshow(img_mitosis_divs, cmap = 'nipy_spectral', vmin = 0, vmax = 1)


# What's up with the matrices we used to multiply our image and produce the new truncated image?

# This is called a binary mask, and it's a special case of 1-bit image, which can only take Boolean values 0 or 1
img_mitosis_nuclearmask = (img_mitosis > 0.15)
print("The Boolean image contents look like this - \n", img_mitosis_nuclearmask, "\n")

plt.figure(figsize = (16, 16))

plt.subplot(221)
plt.gca().set_title('Original image')
plt.imshow(img_mitosis, cmap = 'nipy_spectral')

plt.subplot(222)
plt.gca().set_title('Boolean image mask')
plt.imshow(img_mitosis_nuclearmask, cmap = 'gray')

# We can also perform logical operations on Boolean / binary images!
# For eg, we can apply a NOT gate to invert it!
plt.subplot(223)
plt.gca().set_title('NOT(Boolean mask)')
plt.imshow(np.logical_not(img_mitosis_nuclearmask), cmap = 'gray')

# For eg, we can also apply an XOR gate between the nuclei and the dividers, to pick out non-dividing cells!
img_mitosis_divmask = (img_mitosis > 0.5)
plt.subplot(224)
plt.gca().set_title('Non-dividers = Nuclei XOR Dividers')
plt.imshow(np.logical_xor(img_mitosis_nuclearmask, img_mitosis_divmask), cmap = 'gray')
# This is not perfect, but it can be manipulated further to get rid of the annular rings


# Let's work with the image of mitotic nuclei and try to segment out the shapes of each and identify the dividing cells
# You will recall we just did a primitive version of this - let's recap it here!
img_mitosis = util.img_as_float(data.human_mitosis())

img_mitosis_nuclei = img_mitosis * (img_mitosis > 0.15)
img_mitosis_divs = img_mitosis * (img_mitosis > 0.5)

plt.figure(figsize = (24,6))
plt.gcf().suptitle('Can we improve this identification?', color='red', size=16)

plt.subplot(141)
plt.gca().set_title('Original image')
plt.imshow(img_mitosis, cmap = 'nipy_spectral', vmin = 0, vmax = 1)
plt.colorbar()

plt.subplot(142)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = True)
plt.axvline(0.15, color = 'red')
plt.axvline(0.5, color = 'blue')

plt.subplot(143)
plt.gca().set_title('Nuclei')
plt.imshow(img_mitosis_nuclei, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(144)
plt.gca().set_title('Dividers')
plt.imshow(img_mitosis_divs, cmap = 'nipy_spectral', vmin = 0, vmax = 1)


# We can automate the calculation of the threshold using various built-in algorithms
filters.try_all_threshold(img_mitosis, figsize=(12,24), verbose=False)


# Where do these different algorithms place the threshold value, and how do they mask out the original image?

# Let's try a few examples... We'll start with the simple Mean thresholding, which does well, but merges some nuclei
print('The threshold value for the Mean thresholding algorithm is ', filters.threshold_mean(img_mitosis))

img_mitosis_mask_mean = (img_mitosis > filters.threshold_mean(img_mitosis))

plt.figure(figsize = (16,16))

plt.subplot(221)
plt.imshow(img_mitosis, 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = False)
plt.axvline(filters.threshold_mean(img_mitosis), color = 'red')

plt.subplot(223)
plt.imshow(img_mitosis_mask_mean, 'gray')
plt.gca().set_title('Thresholded mask - Mean thresholding')

plt.subplot(224)
plt.imshow((img_mitosis * img_mitosis_mask_mean), 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image through mask')


# The Triangle thresholding is a good choice for these kinds of microscopy images
print('The threshold value for the Triangle thresholding algorithm is ', filters.threshold_triangle(img_mitosis))

img_mitosis_mask_triangle = (img_mitosis > filters.threshold_triangle(img_mitosis))

plt.figure(figsize = (16,16))

plt.subplot(221)
plt.imshow(img_mitosis, 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = False)
plt.axvline(filters.threshold_triangle(img_mitosis), color = 'red')

plt.subplot(223)
plt.imshow(img_mitosis_mask_triangle, 'gray')
plt.gca().set_title('Thresholded mask - Triangle thresholding')

plt.subplot(224)
plt.imshow((img_mitosis * img_mitosis_mask_triangle), 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image through mask')


# The popular Otsu thresholding gives an even better separation between nuclei
print('The threshold value for the Otsu thresholding algorithm is ', filters.threshold_otsu(img_mitosis))

img_mitosis_mask_otsu = (img_mitosis > filters.threshold_otsu(img_mitosis))

plt.figure(figsize = (16,16))

plt.subplot(221)
plt.imshow(img_mitosis, 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = False)
plt.axvline(filters.threshold_otsu(img_mitosis), color = 'red')

plt.subplot(223)
plt.imshow(img_mitosis_mask_otsu, 'gray')
plt.gca().set_title('Thresholded mask - Otsu thresholding')

plt.subplot(224)
plt.imshow((img_mitosis * img_mitosis_mask_otsu), 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image through mask')


# What about minimum thresholding, which almost removed all the nuclei?
print('The threshold value for the Minimum thresholding algorithm is ', filters.threshold_minimum(img_mitosis))

img_mitosis_mask_minimum = (img_mitosis > filters.threshold_minimum(img_mitosis))

plt.figure(figsize = (16,16))

plt.subplot(221)
plt.imshow(img_mitosis, 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = True)
plt.axvline(filters.threshold_minimum(img_mitosis), color = 'red')

plt.subplot(223)
plt.imshow(img_mitosis_mask_minimum, 'gray')
plt.gca().set_title('Thresholded mask - Minimum thresholding')

plt.subplot(224)
plt.imshow((img_mitosis * img_mitosis_mask_minimum), 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image through mask')


# Finally, let's look at Yen thresholding, which mostly picked out the brightest dividers!
print('The threshold value for the Yen thresholding algorithm is ', filters.threshold_yen(img_mitosis))

img_mitosis_mask_yen = (img_mitosis > filters.threshold_yen(img_mitosis))

plt.figure(figsize = (16,16))

plt.subplot(221)
plt.imshow(img_mitosis, 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = True)
plt.axvline(filters.threshold_yen(img_mitosis), color = 'red')

plt.subplot(223)
plt.imshow(img_mitosis_mask_yen, 'gray')
plt.gca().set_title('Thresholded mask - Yen thresholding')

plt.subplot(224)
plt.imshow((img_mitosis * img_mitosis_mask_yen), 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Original image through mask')


# Sometimes, global thresholding is insufficient - you need local thresholding
# Let's illustrate this with the example of text segmentation
img_page = data.page()

filters.try_all_threshold(img_page, figsize=(12,12), verbose=False)


# The uneven background makes it very obvious that a single global threshold is not appropriate

# The most popular local thresholding algorithm is Niblack thresholding
# Here, you must specify a kernel size (odd number only!) within which to compute the threshold - you can play around with this value
img_page_niblack = (img_page > filters.threshold_niblack(img_page, 55))

plt.figure(figsize = (16,8))

plt.subplot(121)
plt.imshow(img_page, 'gray')
plt.gca().set_title('Original image')

plt.subplot(122)
plt.imshow(img_page_niblack, 'gray')
plt.gca().set_title('Locally Niblack-thresholded image, kernel footprint 71 pixels')


# Let's now work on a more complicated image of nuclei zoomed-in, which shows the value of filtering before thresholding
img_nuc30_nuclei = img_nuc30_cells3d * (img_nuc30_cells3d > 0.12)

img_nuc30_gauss2 = filters.gaussian(img_nuc30_cells3d, sigma = 2)
img_nuc30_gauss2_nuclei = img_nuc30_gauss2 * (img_nuc30_gauss2 > 0.12)

plt.figure(figsize = (24,16))

plt.subplot(231)
plt.gca().set_title('Original image')
plt.imshow(img_nuc30_cells3d, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(232)
plt.hist(img_nuc30_cells3d.flatten(), bins = 64, density = True, log = True)
plt.axvline(0.12, color = 'red')

plt.subplot(233)
plt.gca().set_title('Nuclei')
plt.imshow(img_nuc30_nuclei, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(234)
plt.gca().set_title('Gaussian-filtered image')
plt.imshow(img_nuc30_gauss2, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(235)
plt.hist(img_nuc30_gauss2.flatten(), bins = 64, density = True, log = True)
plt.axvline(0.12, color = 'red')

plt.subplot(236)
plt.gca().set_title('Filtered nuclei')
plt.imshow(img_nuc30_gauss2_nuclei, cmap = 'nipy_spectral', vmin = 0, vmax = 1)


# What thresholding algorithm do you think should work best for this image?
filters.try_all_threshold(filters.gaussian(img_nuc30_cells3d, sigma =1), figsize=(8,16), verbose = False)
# PLay with the values of sigma, and see what is the best compromise to separate all the nuclei

