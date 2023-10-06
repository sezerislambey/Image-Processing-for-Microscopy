## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import color, data



## --------- RGB images ---------

# Let's load a nice color image and view the three colours!
img_lilystem = data.lily()
plt.figure(figsize = (8, 8))
plt.imshow(img_lilystem)


# Let's split this image into each of the three channels, and view them separately!
plt.figure(figsize = (18, 12))

plt.subplot(231)
plt.gca().set_title('Original image')
plt.imshow(img_lilystem)

plt.subplot(234)
plt.gca().set_title('Red channel')
plt.imshow(img_lilystem[0:500,0:500,0], cmap = 'Reds')

plt.subplot(235)
plt.gca().set_title('Green channel')
plt.imshow(img_lilystem[:,:,1], cmap = 'Greens')

plt.subplot(236)
plt.gca().set_title('Blue channel')
plt.imshow(img_lilystem[:,:,2], cmap = 'Blues')


# To look at histopath images, let's load an RGB colour image, showing a histological section of villi in the colon
# A particular protein is highlighted with DAB (brown) immunostaining, while the cells are highlighted with hematoxylin (blue) counter-staining
img_ihc = data.immunohistochemistry()
plt.figure(figsize = (8, 8))
plt.imshow(img_ihc)


# Let's split this image into each of the three channels, and view them separately!
plt.figure(figsize = (24, 8))
plt.subplot(131)
plt.gca().set_title('Red channel')
plt.imshow(img_ihc[:,:,0], cmap = 'Reds')

plt.subplot(132)
plt.gca().set_title('Green channel')
plt.imshow(img_ihc[:,:,1], cmap = 'Greens')

plt.subplot(133)
plt.gca().set_title('Blue channel')
plt.imshow(img_ihc[:,:,2], cmap = 'Blues')


# Note that it's possible to analyze this using a built-in skimage function to separate DAB and H&E staining!
img_ihc_hed = color.rgb2hed(img_ihc)
# Display the channels
plt.figure(figsize = (24, 8))

plt.subplot(131)
plt.gca().set_title('Original RGB image')
plt.imshow(img_ihc)

plt.subplot(132)
plt.gca().set_title('DAB staining isolated')
plt.imshow(img_ihc_hed[:,:,2])

plt.subplot(133)
plt.gca().set_title('Hematoxylin staining isolated')
plt.imshow(img_ihc_hed[:,:,0])


# The above singe-channel images are displayed in the default viridis colormap, because they're considered grayscale!
# If we want to recapture the colours, we need to reconvert them to rgb

# First, we need to create a dummy single-channel image of the same size, to represent the empty channels
null = np.zeros_like(img_ihc_hed[:, :, 0])
# Then, we reconstruct the RGB images appropriately, by putting each single-channel image in the right position and filling the other two positions with the dummy values
img_ihc_h = color.hed2rgb(np.stack((img_ihc_hed[:, :, 0], null, null), axis=-1))
img_ihc_d = color.hed2rgb(np.stack((null, null, img_ihc_hed[:, :, 2]), axis=-1))

# Now we can display the three channels!
plt.figure(figsize = (24, 8))

plt.subplot(131)
plt.gca().set_title('Original RGB image')
plt.imshow(img_ihc)

plt.subplot(132)
plt.gca().set_title('DAB staining isolated')
plt.imshow(img_ihc_d)

plt.subplot(133)
plt.gca().set_title('Hematoxylin staining isolated')
plt.imshow(img_ihc_h)


# Finally, let's convert our RGB image to grayscale
img_ihc_gray = color.rgb2gray(img_ihc)
plt.figure(figsize = (8,8))
plt.imshow(img_ihc_gray, cmap = 'gray')
# NOTE - the image is not "gray" because of the colormap.... It's single-channel, which means grayscale by default
# But it's nice to also make it look gray, hence the colormap!


# Can we convert the grayscale image back to colour?
img_ihc_gray2rgb = color.gray2rgb(img_ihc_gray)
plt.figure(figsize = (16, 8))

plt.subplot(121)
plt.gca().set_title('Original RGB image')
plt.imshow(img_ihc)

plt.subplot(122)
plt.gca().set_title('Reconverted from grayscale')
plt.imshow(img_ihc_gray2rgb)
plt.gcf().suptitle('\n LOST COLOUR DATA CANNOT BE RECONSTRUCTED!', color='red', size=24)
