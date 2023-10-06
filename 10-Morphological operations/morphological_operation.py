## --------- Import all packages required ---------


import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import color, data, filters, morphology, util



## --------- Morphological operations ---------

# Let's choose the Otsu-threshold from above and create a mask for our nuclei
img_nuc30_cells3d = util.img_as_float(data.cells3d()[29, 1, :, :])
img_nuc30_mask = (img_nuc30_cells3d > filters.threshold_otsu(filters.gaussian(img_nuc30_cells3d, sigma = 1)))

plt.figure(figsize=(18,6))

plt.subplot(131)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(132)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(133)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask, 'nipy_spectral')
plt.gca().set_title('Masked image')


# How can we fix the holes in the mask? Enter morphological operations...

# Define a structural element to close holes in the image
strel = morphology.square(5) # Can you find an appropriate size and shape of morphological element to fill the holes without creating artefacts?

img_nuc30_mask_closed = morphology.closing(img_nuc30_mask, strel)

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(img_nuc30_mask_closed, 'gray')
plt.gca().set_title('Closed mask')

plt.subplot(144)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask_closed, 'nipy_spectral')
plt.gca().set_title('New masked image')


# Clearly, this is challenging and there's no perfect answer!

# A better solution is area closing
area_thr = 300 # Can you find an appropriate threshold size to fill the holes?

img_nuc30_mask_areaclosed = morphology.area_closing(img_nuc30_mask, area_threshold = area_thr, connectivity = 2) # What about connectivity?

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(img_nuc30_mask_areaclosed, 'gray')
plt.gca().set_title('Area-closed mask')

plt.subplot(144)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask_areaclosed, 'nipy_spectral')
plt.gca().set_title('New masked image')


# We can also try to remove small holes

area_thr = 300 # Can you find an appropriate threshold size to fill the holes?
img_nuc30_mask_rsh = morphology.remove_small_holes(img_nuc30_mask, area_threshold = area_thr, connectivity = 2) # What about connectivity?

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(img_nuc30_mask_rsh, 'gray')
plt.gca().set_title('Mask with small holes removed')

plt.subplot(144)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask_rsh, 'nipy_spectral')
plt.gca().set_title('New masked image')


# Area closing / removing small holes works, but the edges of the mask look a little "bitten" and rough

# We can fix this by either closing again, or by using a filter! For eg, a median filter, applied twice!
area_thr = 300

img_nuc30_mask_filt = filters.median(morphology.area_closing(img_nuc30_mask, area_threshold = area_thr, connectivity = 1), morphology.square(4))

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(img_nuc30_mask_filt, 'gray')
plt.gca().set_title('Smoothed mask')

plt.subplot(144)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask_filt, 'nipy_spectral')
plt.gca().set_title('New masked image')


# On the left edge, we still have some small bright spots that we can get rid of

# We can fix this by area opening!
area_thr = 300

img_nuc30_mask_final = filters.median(morphology.area_opening(img_nuc30_mask_filt, area_threshold = area_thr, connectivity = 1), morphology.square(3))

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(img_nuc30_mask_final, 'gray')
plt.gca().set_title('Cleaned-up mask')

plt.subplot(144)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask_final, 'nipy_spectral')
plt.gca().set_title('New masked image')


# Another operation that works to get a good mask is finding the convex hull of each shape, which draws the outer outline
img_nuc30_mask_convex = morphology.convex_hull_object(img_nuc30_mask, connectivity = 2) # What about connectivity?

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_nuc30_cells3d, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_nuc30_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(img_nuc30_mask_convex, 'gray')
plt.gca().set_title('Mask with convex hull')

plt.subplot(144)
plt.imshow(img_nuc30_cells3d * img_nuc30_mask_convex, 'nipy_spectral')
plt.gca().set_title('New masked image')


# As a quick aside, the opposite of making a convex hull is skeletonization (or thinning, which is equivalent)

# Let's try it out on a mask of the retinal vessels
img_retina = color.rgb2gray(util.img_as_float(data.retina()))
img_retina_meijering = filters.meijering(img_retina)
img_retina_mask = (img_retina_meijering > filters.threshold_yen(img_retina_meijering))
img_retina_skeleton = morphology.skeletonize(img_retina_mask)

plt.figure(figsize=(24,6))

plt.subplot(141)
plt.imshow(img_retina, 'nipy_spectral')
plt.gca().set_title('Original image')

plt.subplot(142)
plt.imshow(img_retina_mask, 'gray')
plt.gca().set_title('Thresholded mask')

plt.subplot(143)
plt.imshow(morphology.dilation(img_retina_skeleton, morphology.square(3)), 'gray')
plt.gca().set_title('Skeletonized mask to show vessel shapes')

plt.subplot(144)
plt.imshow(img_retina_mask * img_retina, 'nipy_spectral')
plt.gca().set_title('Masked image')