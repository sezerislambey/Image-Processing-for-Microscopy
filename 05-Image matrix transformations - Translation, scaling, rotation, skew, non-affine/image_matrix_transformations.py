## --------- Import all packages required ---------


import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import data, transform, util
import matplotlib.patches as pat


## --------- Image matrix transformations : Translation, scaling, rotation, skew, non-affine ---------

# First, we'll use a method to draw rectangles on our image (import matplotlib.patches as pat)
# Let's work with the composite middle image of the cell membrane and nuclei - img_composite30_cells3d
img_nuc30_cells3d = util.img_as_float(data.cells3d()[29, 0, :, :])
img_cyt30_cells3d = util.img_as_float(data.cells3d()[29, 1, :, :])
img_composite30_cells3d = 0.5*(img_nuc30_cells3d + img_cyt30_cells3d)

# First, let's demo translation of a part of this image, by taking a piece and moving it around
# Let's choose a central square of size 100 px, starting at 33px, and move it down and right by 100 steps
x0 = 33
step = 100

img_movepiece = img_nuc30_cells3d + img_cyt30_cells3d
img_movepiece[x0+step:x0+100+step, x0+step:x0+100+step] = img_movepiece[x0:x0+100, x0:x0+100]

plt.figure(figsize=(24,8))

plt.subplot(131)
plt.gca().set_title('Original image')
plt.imshow(img_composite30_cells3d, cmap = 'inferno')

plt.subplot(132)
plt.gca().set_title('Original image with areas marked up')
plt.imshow(img_composite30_cells3d, cmap = 'inferno')
plt.gca().add_patch(pat.Rectangle((x0, x0), 100, 100, color = 'red', alpha = 0.2))
plt.gca().add_patch(pat.Rectangle((x0+step, x0+step), 100, 100, color = 'green', alpha = 0.3))

plt.subplot(133)
plt.gca().set_title('Image with red piece translated to green area')
plt.imshow(img_movepiece, cmap = 'inferno')
plt.gca().add_patch(pat.Rectangle((x0, x0), 100, 100, facecolor='white', edgecolor = 'red', linewidth = 5, alpha = 0.2))
plt.gca().add_patch(pat.Rectangle((x0+step, x0+step), 100, 100, facecolor='white', edgecolor = 'green', linewidth = 5, alpha = 0.2))


# Let's now rotate the image by some angle
img_rotated = transform.rotate(img_composite30_cells3d, 27)

plt.figure(figsize=(16,8))

plt.subplot(121)
plt.gca().set_title('Original image')
plt.imshow(img_composite30_cells3d, cmap = 'inferno')

plt.subplot(122)
plt.gca().set_title('Rotated image')
plt.imshow(img_rotated, cmap = 'inferno')


# Lastly, let's try out a non-affine swirl transform
img_swirled = transform.swirl(img_composite30_cells3d, radius = 150, strength = 5)

plt.figure(figsize=(16,8))

plt.subplot(121)
plt.gca().set_title('Original image')
plt.imshow(img_composite30_cells3d, cmap = 'inferno')

plt.subplot(122)
plt.gca().set_title('Swirled image')
plt.imshow(img_swirled, cmap = 'inferno')
