## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import  data, exposure, util



## --------- Exposure and Contrast adjustment ---------

# Let's work with an image that will highlight the effects of contrast and exposure
# Load the nuclear channel of the 30th image in the stack cells3d from skimage.data on a 0 to 1 scale
img_nuc30_cells3d = data.cells3d()
img_nuc30_cells3d= util.img_as_float(img_nuc30_cells3d[30, 1, :, :])

# Let's check first whether the image is considered low contrast?
print('Is the image low contrast? ', exposure.is_low_contrast(img_nuc30_cells3d))


# Now we'll display the image and look for ourselves
plt.figure(figsize = (8,8))
plt.imshow(img_nuc30_cells3d, cmap='gray')
plt.colorbar()


# Perform the following operations on this image and see how they affect the appearance of the image
# Display all the images in a 3 X 2 format, with the original image at the top left
plt.figure(figsize = (24, 16))
plt.gcf().suptitle('Contrast adjustment methods', size = 24)

# 0. Show original image for reference
plt.subplot(231)
plt.gca().set_title('Original image')
plt.imshow(img_nuc30_cells3d, cmap = 'gray', vmin = 0, vmax = 1)
plt.colorbar()

# 1. Set gamma = 2
plt.subplot(232)
plt.gca().set_title('gamma = 2')
plt.imshow(exposure.adjust_gamma(img_nuc30_cells3d, 2), cmap = 'gray', vmin = 0, vmax = 1)
plt.colorbar()

# 2. Set gamma = 0.5
plt.subplot(233)
plt.gca().set_title('gamma = 0.5')
plt.imshow(exposure.adjust_gamma(img_nuc30_cells3d, 0.5), cmap = 'gray', vmin = 0, vmax = 1)
plt.colorbar()

# 3. Apply logarithmic correction
plt.subplot(234)
plt.gca().set_title('Log correction')
plt.imshow(exposure.adjust_log(img_nuc30_cells3d), cmap = 'gray', vmin = 0, vmax = 1)
plt.colorbar()

# 4. Apply sigmoidal correction
plt.subplot(235)
plt.gca().set_title('Sigmoid correction')
plt.imshow(exposure.adjust_sigmoid(img_nuc30_cells3d), cmap = 'gray', vmin = 0, vmax = 1)
plt.colorbar()

# 5. Apply intensity rescaling from 0.4 to 0.6
plt.subplot(236)
plt.gca().set_title('Rescaled intensity')
plt.imshow(exposure.rescale_intensity(img_nuc30_cells3d, (img_nuc30_cells3d.min(), img_nuc30_cells3d.max()), (0.4,0.6)), cmap = 'gray', vmin = 0, vmax = 1)
plt.colorbar()


# We can perform contrast enhancement operations like histogram equalization
img_nuc30_eqhist = exposure.equalize_hist(img_nuc30_cells3d)

plt.figure(figsize = (12, 12))

plt.subplot(221)
plt.gca().set_title('Original image')
plt.imshow(img_nuc30_cells3d, cmap = 'gray', vmin = 0, vmax = 1)

plt.subplot(222)
plt.gca().set_title('Histogram equalized image')
plt.imshow(img_nuc30_eqhist, cmap = 'gray', vmin = 0, vmax = 1)

plt.subplot(223)
plt.hist(img_nuc30_cells3d.flatten(), bins=64, density = True)
img_cdf, bins = exposure.cumulative_distribution(img_nuc30_cells3d, 64)
plt.plot(bins, img_cdf, 'r')

plt.subplot(224)
plt.hist(img_nuc30_eqhist.flatten(), bins=64, density = True)
img_cdf, bins = exposure.cumulative_distribution(img_nuc30_eqhist, 64)
plt.plot(bins, img_cdf, 'r')


# We can also enhance contrast by a process called image clipping, where we get rid of the outlier pixels
lower_clip, upper_clip = np.percentile(img_nuc30_cells3d, q=(5, 95))
img_nuc30_rescaled = exposure.rescale_intensity(img_nuc30_cells3d, in_range=(lower_clip, upper_clip), out_range=np.float32)

plt.figure(figsize=(16,16))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d, 'inferno', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_nuc30_rescaled, 'inferno', vmin = 0, vmax = 1)
plt.gca().set_title('Clipped image')

plt.subplot(223)
plt.hist(img_nuc30_cells3d.flatten(), bins=64, density = True)
img_cdf, bins = exposure.cumulative_distribution(img_nuc30_cells3d, 64)
plt.plot(bins, img_cdf, 'r')

plt.subplot(224)
plt.hist(img_nuc30_rescaled.flatten(), bins=64, density = True)
img_cdf, bins = exposure.cumulative_distribution(img_nuc30_rescaled, 64)
plt.plot(bins, img_cdf, 'r')

"""
_____NOTE:_____
Operations like equalization and clipping make the output image look brighter and sharper for better display.
However, this is pixel value manipulation, and must NEVER be used in your quantitative analysis pipeline!
"""
