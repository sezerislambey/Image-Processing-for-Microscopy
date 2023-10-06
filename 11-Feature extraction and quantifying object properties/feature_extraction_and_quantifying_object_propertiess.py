## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import  data, filters, measure, morphology, util



## --------- Feature extraction and quantifying object propertiess ---------


img_nuc30_cells3d = util.img_as_float(data.cells3d()[29, 1, :, :])

img_nuc30_mask = (img_nuc30_cells3d > filters.threshold_otsu(filters.gaussian(img_nuc30_cells3d, sigma = 1)))

# We can fix this by either closing again, or by using a filter! For eg, a median filter, applied twice!
# We can fix this by area opening!
area_thr = 300

img_nuc30_mask_filt = filters.median(morphology.area_closing(img_nuc30_mask, area_threshold = area_thr, connectivity = 1), morphology.square(4))


# On the left edge, we still have some small bright spots that we can get rid of

img_nuc30_mask_final = filters.median(morphology.area_opening(img_nuc30_mask_filt, area_threshold = area_thr, connectivity = 1), morphology.square(3))


# We've now reached the last part of our pipeline - quantifying the properties of the different nuclei

# The very first thing we want to do is to label our different nuclei as individual connected components
img_nuc30_labeled = measure.label(img_nuc30_mask_final)

plt.figure(figsize=(16,8))

plt.subplot(121)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(122)
plt.imshow(img_nuc30_labeled, 'nipy_spectral', vmin = 0, vmax = np.max(img_nuc30_labeled) + 1)
plt.gca().set_title('Image with labeled components')


# Note that the label 0 corresponds to the background, and each subsequent label corresponds to each object
plt.figure(figsize=(18,6))

plt.subplot(131)
plt.imshow(img_nuc30_labeled == 0, 'nipy_spectral', vmin = 0, vmax = 1)
plt.gca().set_title('Background = label 0')

plt.subplot(132)
plt.imshow((img_nuc30_labeled == 1), 'nipy_spectral', vmin = 0, vmax = np.max(img_nuc30_labeled) + 1)
plt.gca().set_title('Nucleus 1 = label 1')

plt.subplot(133)
plt.imshow(14 * (img_nuc30_labeled == 14), 'nipy_spectral', vmin = 0, vmax = np.max(img_nuc30_labeled) + 1)
plt.gca().set_title('Nucleus 14 = label 14')

print("Number of nuclei = ", np.max(img_nuc30_labeled))


# We will now compute all the properties of each nucleus
propvalues = measure.regionprops(label_image = img_nuc30_labeled, intensity_image = img_nuc30_cells3d)


# Here is a list of all the properties that exist, with values displayed for the k-th nucleus
k = 5

# First, let's rint out the list of all properties that exist
print('For nucleus ', k,' -')
for prop in propvalues[k - 1]:
  print('Property : ', prop, ' = ', propvalues[k - 1][prop])

# Next, let's display an image of the k-th nucleus
plt.figure(figsize=(6,6))
plt.imshow(k * (img_nuc30_labeled == k), 'nipy_spectral', vmin = 0, vmax = np.max(img_nuc30_labeled) + 1)


# Let's display the mask with nuclei colored by various properties
plt.figure(figsize = (16,16))

# Colored by area
plt.subplot(221)
img_nuc30_areas = np.zeros_like(img_nuc30_labeled, dtype='float64') # Initialize a container for the composite image
for k in range(1, np.max(img_nuc30_labeled)): # Go through each of the objects individually
  img_nuc30_areas += propvalues[k-1].area*(img_nuc30_labeled == k) # Multiply its mask by its property value, and add it up into a composite image
plt.imshow(img_nuc30_areas, 'nipy_spectral', vmin = 0, vmax = 1.1*np.max(img_nuc30_areas)) # Display that composite image with appropriate scale
plt.gca().set_title('Colored by area')
plt.colorbar()

# ---------------------- #

# Colored by X-coordinate of centroid
plt.subplot(222)
img_nuc30_centroidX = np.zeros_like(img_nuc30_labeled, dtype='float64')
for k in range(1, np.max(img_nuc30_labeled)):
  img_nuc30_centroidX += propvalues[k-1].centroid[1]*(img_nuc30_labeled == k)
plt.imshow(img_nuc30_centroidX, 'nipy_spectral', vmin = 0, vmax = 1.1*np.max(img_nuc30_centroidX))
plt.gca().set_title('Colored by centroid X-coordinate')
plt.colorbar()

# ---------------------- #

# Colored by mean intensity
plt.subplot(223)
img_nuc30_intmean = np.zeros_like(img_nuc30_labeled, dtype='float64')
for k in range(1, np.max(img_nuc30_labeled)):
  img_nuc30_intmean += propvalues[k-1].mean_intensity*(img_nuc30_labeled == k)
plt.imshow(img_nuc30_intmean, 'nipy_spectral', vmin = 0, vmax = 1.1*np.max(img_nuc30_intmean))
plt.gca().set_title('Colored by mean fluorescence intensity')
plt.colorbar()

# ---------------------- #

# Original image
plt.subplot(224)
plt.imshow(img_nuc30_cells3d, 'inferno')
plt.gca().set_title('Original image')
plt.colorbar()

