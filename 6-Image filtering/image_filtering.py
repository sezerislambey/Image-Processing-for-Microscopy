## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import color, data, filters, morphology, util



## --------- Image filtering ---------

# In this section, we will use four images to demonstrate various effects

# First, let's reload the image of the mitotic cells, as a Uint8 image to work with
img_mitosis = util.img_as_ubyte(data.human_mitosis())

# Second, we will use the image of zoomed-in nuclei from the middle of the image stack
img_nuc30_cells3d = util.img_as_float(data.cells3d()[29, 1, :, :])

# Third, we will pick an astronomical image - the Hubble Deep Field
img_hubble = util.img_as_float(data.hubble_deep_field())

# Fourth, we will pick an image with a very different topology of thin lines - the retinal fundus
img_retina = util.img_as_float(color.rgb2gray(data.retina()))


# Let's pick one of these images and look at the effect of adding random noise with different strengths

rng = np.random.default_rng(1)

img_nuc30_cells3d_noise5 = img_nuc30_cells3d + (rng.normal(size=img_nuc30_cells3d.shape) * 0.05)
img_nuc30_cells3d_noise10 = img_nuc30_cells3d + (rng.normal(size=img_nuc30_cells3d.shape) * 0.10)
img_nuc30_cells3d_noise20 = img_nuc30_cells3d + (rng.normal(size=img_nuc30_cells3d.shape) * 0.20)

plt.figure(figsize = (12,12))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_nuc30_cells3d_noise5, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Noisy image (stdev = 5%)')

# Can you now display a noisier image with stdev 10?
plt.subplot(223)
plt.imshow(img_nuc30_cells3d_noise10, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Noisy image (stdev = 10%)')

# What about noise with stdev 20?
plt.subplot(224)
plt.imshow(img_nuc30_cells3d_noise20, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Noisy image (stdev = 20%)')


# First, let's write out what a 3 x 3 square mean filter looks like
# You could write - kernel_mean = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]] OR more elegantly...
kernel_mean = (1/9)*np.ones((3,3))


# Another way to write out a kernel is to simply provide its shape ; scikit-image then fills in the rest
# The morphology module provides patterns for these, called "structural elements"
# For a 3 x 3 square kernel, we would write the strel as -
kernel_sq3 = morphology.square(3)


# To apply the mean filter to an image, we use the rank_filters package in scikit-image
# First, we must write out the size and shape of the filter kernel - in this example, we choose a 3 x 3 square
img_mitosis_meanfilt = filters.rank.mean(img_mitosis, kernel_sq3)

plt.figure(figsize = (18,18))

plt.subplot(221)
plt.imshow(img_mitosis, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_mitosis_meanfilt, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('3 x 3 Mean filtered image')

# Can you now write a filter using a 7 x 7 square shaped kernel?
plt.subplot(223)
plt.imshow(filters.rank.mean(img_mitosis, morphology.square(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 square - Mean filtered image')

# What about a filter using a 7 x 7 disk shaped kernel?
plt.subplot(224)
plt.imshow(filters.rank.mean(img_mitosis, morphology.disk(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 disk - Mean filtered image')

# Median
img_mitosis_medianfilt = filters.median(img_mitosis, kernel_sq3)

plt.figure(figsize = (18,18))

plt.subplot(221)
plt.imshow(img_mitosis, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_mitosis_medianfilt, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('3 x 3 Median filtered image')

# Can you now write a filter using a 7 x 7 square shaped kernel?
plt.subplot(223)
plt.imshow(filters.median(img_mitosis, morphology.square(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 square - Median filtered image')

# What about a filter using a 7 x 7 disk shaped kernel?
plt.subplot(224)
plt.imshow(filters.median(img_mitosis, morphology.disk(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 disk - Median filtered image')


# Let's try this with another image
img_nuc30_cells3d_meanfilt = filters.rank.mean(img_nuc30_cells3d, kernel_sq3)

plt.figure(figsize = (12,12))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_nuc30_cells3d_meanfilt, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('3 x 3 Mean filtered image')

# Can you now write a filter using a 7 x 7 square shaped kernel?
plt.subplot(223)
plt.imshow(filters.rank.mean(img_nuc30_cells3d, morphology.square(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 square - Mean filtered image')

# What about a filter using a 7 x 7 disk shaped kernel?
plt.subplot(224)
plt.imshow(filters.rank.mean(img_nuc30_cells3d, morphology.disk(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 disk - Mean filtered image')


# Let's try this with another image
img_nuc30_cells3d_noise5_meanfilt = filters.rank.mean(img_nuc30_cells3d_noise5, kernel_sq3)

plt.figure(figsize = (12,12))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d_noise5, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Noisy image')

plt.subplot(222)
plt.imshow(img_nuc30_cells3d_noise5_meanfilt, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('3 x 3 Mean filtered image')

# Can you now write a filter using a 7 x 7 square shaped kernel?
plt.subplot(223)
plt.imshow(filters.rank.mean(img_nuc30_cells3d_noise5, morphology.square(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 square - Mean filtered image')

# What about a filter using a 7 x 7 disk shaped kernel?
plt.subplot(224)
plt.imshow(filters.rank.mean(img_nuc30_cells3d_noise5, morphology.disk(7)), 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('7 x 7 disk - Mean filtered image')


# What about min and max filters?
img_mitosis_min = filters.rank.minimum(img_mitosis, morphology.square(4))
img_mitosis_max = filters.rank.maximum(img_mitosis, morphology.square(4))

plt.figure(figsize = (16, 16))

plt.subplot(221)
plt.imshow(img_mitosis[300:500,200:400], 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Original image')

plt.subplot(223)
plt.imshow(img_mitosis_min[300:500,200:400], 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Minimum filtered image')

plt.subplot(224)
plt.imshow(img_mitosis_max[300:500,200:400], 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Maximum filtered image')

plt.subplot(222)
plt.imshow(img_mitosis_max[300:500,200:400] - img_mitosis_min[300:500,200:400], 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Maximum-filt - Minimum-filt')


# Let's try with our other noisy image...
img_nuc30_cells3d_noise5_min = filters.rank.minimum(img_nuc30_cells3d_noise5, morphology.square(2))
img_nuc30_cells3d_noise5_max = filters.rank.maximum(img_nuc30_cells3d_noise5, morphology.square(2))

plt.figure(figsize = (12, 12))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d_noise5, 'gray', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(223)
plt.imshow(img_nuc30_cells3d_noise5_min, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Minimum filtered image')

plt.subplot(224)
plt.imshow(img_nuc30_cells3d_noise5_max, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Maximum filtered image')

plt.subplot(222)
plt.imshow(img_nuc30_cells3d_noise5_max - img_nuc30_cells3d_noise5_min, 'gray', vmin = 0, vmax = 255)
plt.gca().set_title('Maximum-filt - Minimum-filt')


# It's easy to now convert this to a non-linear rank filter - let's try the median filter!
# Let's do this on a more zoomed-in image of nuclei - img_nuc30_cells3d - to see the true effects of the filter
img_nuc30_medianfilt = filters.median(img_nuc30_cells3d, morphology.square(3))

plt.figure(figsize = (24,24))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_nuc30_medianfilt, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('3 x 3 square median filtered image')

plt.subplot(223)
plt.imshow(filters.median(img_nuc30_cells3d, morphology.square(7)), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('7 x 7 square median filtered image')

plt.subplot(224)
plt.imshow(filters.median(img_nuc30_cells3d, morphology.disk(7)), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('7-radius disk median filtered image')


# It's easy to now convert this to a non-linear rank filter - let's try the median filter!
# Let's do this on a more zoomed-in image of nuclei - img_nuc30_cells3d - to see the true effects of the filter
img_nuc30_noise5_medianfilt = filters.median(img_nuc30_cells3d_noise5, morphology.square(3))

plt.figure(figsize = (24,24))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d_noise5, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_nuc30_noise5_medianfilt, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('3 x 3 square median filtered image')

plt.subplot(223)
plt.imshow(filters.median(img_nuc30_cells3d_noise5, morphology.square(7)), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('7 x 7 square median filtered image')

plt.subplot(224)
plt.imshow(filters.median(img_nuc30_cells3d_noise5, morphology.disk(7)), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('7-radius disk median filtered image')


# We already saw one way of edge detection - using a max-min filter subtraction...
# Let's look at edge detection using gradient filters!
img_nuc30_sobelmag = filters.sobel(img_nuc30_cells3d)
img_nuc30_sobelh = filters.sobel_h(img_nuc30_cells3d)
img_nuc30_sobelv = filters.sobel_v(img_nuc30_cells3d)

plt.figure(figsize = (24,16))

plt.subplot(231)
plt.imshow(img_nuc30_sobelmag, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Sobel magnitude filtered image')

plt.subplot(232)
plt.imshow(img_nuc30_sobelh, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Sobel horizontal filtered image')

plt.subplot(233)
plt.imshow(img_nuc30_sobelv, 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Sobel vertical filtered image')

# Similarly, you can try out the Roberts, Prewitt, and Laplace filters here
plt.subplot(234)
plt.imshow(filters.roberts(img_nuc30_cells3d), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Roberts filtered image')

plt.subplot(235)
plt.imshow(filters.prewitt(img_nuc30_cells3d), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Prewitt filtered image')

plt.subplot(236)
plt.imshow(filters.laplace(img_nuc30_cells3d), 'gray', vmin = 0, vmax = np.max(img_nuc30_cells3d))
plt.gca().set_title('Laplace filtered image')


# The most important class of filters we will now see are called GAUSSIAN FILTERS (these are also linear filters)
# Let's first try the effects on the mitosis image
img_mitosis_gauss2 = filters.gaussian(img_mitosis, sigma = 2)
img_mitosis_gauss1 = filters.gaussian(img_mitosis, sigma = 1)
img_mitosis_gauss7 = filters.gaussian(img_mitosis, sigma = 7)

plt.figure(figsize = (16, 16))

plt.subplot(221)
plt.imshow(img_mitosis[300:500,200:400], 'gray')
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_mitosis_gauss2[300:500,200:400], 'gray')
plt.gca().set_title('Gaussian (sigma = 2) filtered image')

plt.subplot(223)
plt.imshow(img_mitosis_gauss1[300:500,200:400], 'gray')
plt.gca().set_title('Gaussian (sigma = 1) filtered image')

plt.subplot(224)
plt.imshow(img_mitosis_gauss7[300:500,200:400], 'gray')
plt.gca().set_title('Gaussian (sigma = 7) filtered image')


# Let's try the effects on the more zoomed-in nuclei image
img_nuc30_gauss2 = filters.gaussian(img_nuc30_cells3d, sigma = 2)
img_nuc30_gauss1 = filters.gaussian(img_nuc30_cells3d, sigma = 1)
img_nuc30_gauss7 = filters.gaussian(img_nuc30_cells3d, sigma = 7)

plt.figure(figsize = (24, 24))

plt.subplot(221)
plt.imshow(img_nuc30_cells3d, 'gray')
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_nuc30_gauss2, 'gray')
plt.gca().set_title('Gaussian (sigma = 2) filtered image')

plt.subplot(223)
plt.imshow(img_nuc30_gauss1, 'gray')
plt.gca().set_title('Gaussian (sigma = 1) filtered image')

plt.subplot(224)
plt.imshow(img_nuc30_gauss7, 'gray')
plt.gca().set_title('Gaussian (sigma = 7) filtered image')


# We can also use Gaussan filtering to establish the spatial scales of an image with different-sized objects, and filter out different sized objects
img_hubble_gauss2 = filters.gaussian(img_hubble, sigma = 2)
img_hubble_gauss1 = filters.gaussian(img_hubble, sigma = 1)
img_hubble_gauss7 = filters.gaussian(img_hubble, sigma = 7)

plt.figure(figsize = (16, 16))

plt.subplot(221)
plt.imshow(img_hubble[0:200, 300:500], 'gray')
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_hubble_gauss2[0:200, 300:500], 'gray')
plt.gca().set_title('Gaussian (sigma = 2) filtered image')

plt.subplot(223)
plt.imshow(img_hubble_gauss1[0:200, 300:500], 'gray')
plt.gca().set_title('Gaussian (sigma = 1) filtered image')

plt.subplot(224)
plt.imshow(img_hubble_gauss7[0:200, 300:500], 'gray')
plt.gca().set_title('Gaussian (sigma = 7) filtered image')


# It's also very useful to apply a Difference of Gaussians filter to isolate a specific spatial scale
img_hubble_dog2to7 = filters.difference_of_gaussians(img_hubble, low_sigma = 1, high_sigma = 2)

plt.figure(figsize = (24, 16))

plt.subplot(231)
plt.imshow(img_hubble[0:200, 300:500], 'gray')
plt.gca().set_title('Original image')

plt.subplot(232)
plt.imshow(img_hubble_gauss1[0:200, 300:500] - img_hubble_gauss2[0:200, 300:500], 'gray')
plt.gca().set_title('Manually subtracted Difference of Gaussians (sigma = 1 to 2) filtered image')

plt.subplot(233)
plt.imshow(img_hubble_dog2to7[0:200, 300:500], 'gray')
plt.gca().set_title('Difference of Gaussians (sigma = 1 to 2) filtered image')

plt.subplot(234)
plt.imshow(img_hubble_gauss2[0:200, 300:500] - img_hubble_gauss1[0:200, 300:500], 'gray')
plt.gca().set_title('Difference of Gaussians (sigma = 2 to 1) - not useful!!')

plt.subplot(235)
plt.imshow(img_hubble_gauss1[0:200, 300:500] - img_hubble_gauss7[0:200, 300:500], 'gray')
plt.gca().set_title('Difference of Gaussians (sigma = 1 to 7) filtered image')

plt.subplot(236)
plt.imshow(img_hubble_gauss2[0:200, 300:500] - img_hubble_gauss7[0:200, 300:500], 'gray')
plt.gca().set_title('Difference of Gaussians (sigma = 2 to 7) filtered image')


# The last important filter we will explore here is the Laplacian of Gaussian filter, which is a RIDGE detector!
# For this, we will use the image of the retina, to try and pull out the vessels in the background
# This is called a composite filter, since we're applying the second filter to the output of the first
img_retina_gauss = filters.gaussian(img_retina, sigma = 2)
img_retina_laplace = filters.laplace(img_retina)
img_retina_log = filters.laplace(filters.gaussian(img_retina, sigma = 2))

plt.figure(figsize = (16, 16))

plt.subplot(221)
plt.imshow(img_retina, 'gray')
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_retina_log, 'gray')
plt.gca().set_title('LoG-filtered image')

plt.subplot(223)
plt.imshow(img_retina_gauss, 'gray')
plt.gca().set_title('Gaussian-filtered image')

plt.subplot(224)
plt.imshow(img_retina_laplace, 'gray')
plt.gca().set_title('Laplace-filtered image')


# There are many other filters you can use to identify ridge-like structures
# For example, let's try a couple of others - Sato, Frangi, Meijering
img_retina_sato = filters.sato(img_retina)
img_retina_frangi = filters.frangi(img_retina, sigmas=range(4, 7, 1))
img_retina_meijering = filters.meijering(img_retina)

plt.figure(figsize = (24,12))

plt.subplot(231)
plt.imshow(img_retina, 'inferno')
plt.gca().set_title('Original image')

plt.subplot(232)
plt.imshow(img_retina_log, 'inferno')
plt.gca().set_title('Log-filtered image')

plt.subplot(233)
plt.imshow(filters.sobel(img_retina), 'inferno')
plt.gca().set_title('Sobel edge-filtered image')

plt.subplot(234)
plt.imshow(img_retina_sato, 'inferno')
plt.gca().set_title('Sato tubeness-filtered image')

plt.subplot(235)
plt.imshow(img_retina_frangi, 'inferno')
plt.gca().set_title('Frangi vessel-like-filtered image')

plt.subplot(236)
plt.imshow(img_retina_meijering, 'inferno')
plt.gca().set_title('Meijering neurite-like-filtered image')


# Finally, we will look at how filters can smooth out the spatial background of an image

# We will look at the image of dividing nuclei, which has uneven background illumination
img_mitosis = util.img_as_ubyte(data.human_mitosis())

img_mitosis_bgtophat = morphology.white_tophat(img_mitosis, morphology.disk(7))

plt.figure(figsize = (16,8))

plt.subplot(121)
plt.imshow(img_mitosis, 'nipy_spectral', vmin = 0, vmax = 255)
plt.gca().set_title('Original image')

plt.subplot(122)
plt.imshow(img_mitosis_bgtophat, 'nipy_spectral', vmin = 0, vmax = 255)
plt.gca().set_title('Spatially smoothed image')


# So far, most of the filters we've seen (especially Gaussian filters) smooth out images...
# Is there anything at all we can do to sharpen them? Enter unsharp masking....

# ACTUAL EQUATION : img_usm = img + 0.7*(img - filters.gaussian(img, sigma = 1))

# However, a built-in function also exists in scikit-image
img_mitosis_usm = filters.unsharp_mask(img_mitosis)
img_nuc30_usm = filters.unsharp_mask(img_nuc30_cells3d)

plt.figure(figsize = (16,16))

plt.subplot(221)
plt.imshow(img_mitosis[300:500,200:400], 'inferno')
plt.gca().set_title('Original image')

plt.subplot(222)
plt.imshow(img_mitosis_usm[300:500,200:400], 'inferno', vmin = 0, vmax = 1)
plt.gca().set_title('Unsharp-masked image')

plt.subplot(223)
plt.imshow(img_nuc30_cells3d, 'inferno', vmin = 0, vmax = 1)
plt.gca().set_title('Original image')

plt.subplot(224)
plt.imshow(img_nuc30_usm, 'inferno', vmin = 0, vmax = 1)
plt.gca().set_title('Unsharp-masked image')


"""
_____NOTE_____:
Unsharp masking is close to dubious manipulation of an image. 
Hence, while it's great for display purposes, NEVER do this in your actual quantitative analysis pipeline!
"""