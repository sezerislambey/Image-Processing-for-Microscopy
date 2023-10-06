## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import data, util


## --------- Point operations using image matrices ---------

# Let's work with the middle image of the cell membrane and nuclei stack
img_cyt30_cells3d= util.img_as_float(data.cells3d()[29, 0, :, :])
img_nuc30_cells3d= util.img_as_float(data.cells3d()[29, 1, :, :])

plt.figure(figsize=(16,8))

plt.subplot(121)
plt.gca().set_title('Cytoplasm')
plt.imshow(img_cyt30_cells3d, cmap = 'inferno')

plt.subplot(122)
plt.gca().set_title('Nucleus')
plt.imshow(img_nuc30_cells3d, cmap = 'inferno')


# MATRIX ADDITION - We can add these matrices together to superimpose them!
img_composite30_cells3d = 0.5*img_nuc30_cells3d + 0.5*img_cyt30_cells3d
plt.figure(figsize = (8,8))
plt.imshow(img_composite30_cells3d, cmap = 'inferno')


# PROFILING - taking an intensity profile along a line
# For eg, let's take a profile of the composite above at the horizontal 100-line
profile = img_composite30_cells3d[100,:]

plt.figure(figsize = (16,8))

plt.subplot(121)
plt.imshow(img_composite30_cells3d, cmap = 'inferno')
plt.axhline(100, color = 'red')

plt.subplot(122)
plt.plot(profile)



# MATRIX MULTIPLICATION BY SCALARS -
# Let's now use an RGB image and create colour filters
img_ihc = data.immunohistochemistry()

# Say we want to enhance the blue channel - that is, weight the blue channel twice as much as the other two channels
img_ihc_blue_enh = np.zeros_like(img_ihc)
img_ihc_blue_enh[:,:,0] = 0.5*img_ihc[:,:,0]
img_ihc_blue_enh[:,:,1] = 0.5*img_ihc[:,:,1]
img_ihc_blue_enh[:,:,2] = img_ihc[:,:,2]

# Similarly, say we want to apply a blue filter - that is, remove the blue channel and keep the other two
img_ihc_blue_filt = np.zeros_like(img_ihc)
img_ihc_blue_filt[:,:,0] = img_ihc[:,:,0]
img_ihc_blue_filt[:,:,1] = img_ihc[:,:,1]
img_ihc_blue_filt[:,:,2] = 0*img_ihc[:,:,2]

plt.figure(figsize = (16,8))

plt.subplot(131)
plt.gca().set_title('Original image')
plt.imshow(img_ihc)

plt.subplot(132)
plt.gca().set_title('Blue-enhanced image')
plt.imshow(img_ihc_blue_enh)

plt.subplot(133)
plt.gca().set_title('Blue-filtered image')
plt.imshow(img_ihc_blue_filt)
