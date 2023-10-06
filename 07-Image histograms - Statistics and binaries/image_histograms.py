## --------- Import all packages required ---------


import matplotlib.pyplot as plt
# %matplotlib inlinee
from skimage import color, data, util


## --------- Image histograms : Statistics and binaries ---------

# Let's work with images having different kinds of histograms
img_cell = util.img_as_float(data.cell())
img_mitosis = util.img_as_float(data.human_mitosis())
img_retina = util.img_as_float(color.rgb2gray(data.retina()))
img_nuc30_cells3d = util.img_as_float(data.cells3d()[29, 1, :, :])


# To create histograms, we first have to flatten the 2D pixel array into a 1D array of numbers
# Then we can plot the histograms using Matplotlib
plt.figure(figsize = (18, 12))

plt.subplot(231)
plt.gca().set_title('QPI of a single cell')
plt.imshow(img_cell, cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(234)
plt.hist(img_cell.flatten(), bins = 64, density = True)

plt.subplot(232)
plt.gca().set_title('Field of nuclei')
plt.imshow(img_mitosis, cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(235)
plt.hist(img_mitosis.flatten(), bins = 64, density = True)

plt.subplot(233)
plt.gca().set_title('Retinal fundus with nerves')
plt.imshow(img_retina, cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(236)
plt.hist(img_retina.flatten(), bins = 64, density = True)


# Sometimes we can't see the realy low bumps properly, so we can set the Y-axis to a log scale
plt.figure(figsize = (18, 12))
plt.subplot(231)
plt.gca().set_title('QPI of a single cell')
plt.imshow(img_cell, cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(234)
plt.hist(img_cell.flatten(), bins = 64, density = True, log = True)

plt.subplot(232)
plt.gca().set_title('Field of nuclei')
plt.imshow(img_mitosis, cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(235)
plt.hist(img_mitosis.flatten(), bins = 64, density = True, log = True)

plt.subplot(233)
plt.gca().set_title('Retinal fundus with nerves')
plt.imshow(img_retina, cmap = 'gray', vmin = 0, vmax = 1)
plt.subplot(236)
plt.hist(img_retina.flatten(), bins = 64, density = True, log = True)


# We can adjust the histograms to alter brightness and contrast in the images
# For eg, let's square all the pixels - this will shift the distribution to the darker side, and make brighter spots stand out even more!
img_mitosis_square = img_mitosis**2

plt.figure(figsize = (12, 12))

plt.subplot(221)
plt.gca().set_title('Original image')
plt.imshow(img_mitosis, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(222)
plt.gca().set_title('Squared image')
plt.imshow(img_mitosis_square, cmap = 'nipy_spectral', vmin = 0, vmax = 1)

plt.subplot(223)
plt.hist(img_mitosis.flatten(), bins=64, density = True)

plt.subplot(224)
plt.hist(img_mitosis_square.flatten(), bins=64, density = True)
