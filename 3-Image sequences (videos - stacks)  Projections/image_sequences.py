## --------- Import all packages required ---------


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import data, util
import plotly
import plotly.express as px



## --------- Image sequences (videos / stacks) : Projections ---------

# Let's load an image sequence to work with
# We will use a 3-D volumetric Z-stack with 2 channels showing nucleus and cytoplasm

img_stack = util.img_as_float(data.cells3d())


# Plotly is a package that helps us animate graphical displays in Python
# Here, we will use Plotly to create a slider that allows us to run through the various images in the Z-stack

fig = px.imshow(img_stack, facet_col=1, animation_frame=0, binary_string=True, binary_format='jpg')
fig.layout.annotations[0]['text'] = 'Cell membranes'
fig.layout.annotations[1]['text'] = 'Nuclei'
plotly.io.show(fig)


# First, we will display 2 different slices to show the differences between them in each channel
# Let's choose to display the 15th and 30th slices of the stack - 1/4th and 1/2-way along the thickness of the section
plt.figure(figsize = (16, 16))

plt.subplot(221)
plt.gca().set_title('Slice 15 : Cytoplasm')
plt.imshow(img_stack[14, 0], cmap = 'inferno', vmin = 0, vmax = 1)

plt.subplot(222)
plt.gca().set_title('Slice 15 : Nucleus')
plt.imshow(img_stack[14, 1], cmap = 'inferno', vmin = 0, vmax = 1)

plt.subplot(223)
plt.gca().set_title('Slice 30 : Cytoplasm')
plt.imshow(img_stack[29, 0], cmap = 'inferno', vmin = 0, vmax = 1)

plt.subplot(224)
plt.gca().set_title('Slice 30 : Nucleus')
plt.imshow(img_stack[29, 1], cmap = 'inferno', vmin = 0, vmax = 1)


# First, let's take a MAXIMUM INTENSITY PROJECTION of all images in the stack
# Numpy provides us a maximum function for any array, given an axis
# In this case, our array is the 3D stack, and we're projecting along the Z-axis for slices!
img_stack_maxp_cyto= np.max(img_stack[:,0], axis=0)
img_stack_maxp_nuc= np.max(img_stack[:,1], axis=0)

plt.figure(figsize = (16, 8))
plt.gcf().suptitle('Maximum intensity projections', size = 24)

plt.subplot(121)
plt.imshow(img_stack_maxp_cyto, cmap = 'inferno', vmin = 0, vmax = 1)

plt.subplot(122)
plt.imshow(img_stack_maxp_nuc, cmap = 'inferno', vmin = 0, vmax = 1)


# Similarly, let's take a MINIMUM INTENSITY PROJECTION of all images in the stack
img_stack_minp_cyto= np.min(img_stack[:,0], axis=0)
img_stack_minp_nuc= np.min(img_stack[:,1], axis=0)

plt.figure(figsize = (16, 8))
plt.gcf().suptitle('Minimum intensity projections', size = 24)

plt.subplot(121)
plt.imshow(img_stack_minp_cyto, cmap = 'inferno', vmin = 0, vmax = 1)

plt.subplot(122)
plt.imshow(img_stack_minp_nuc, cmap = 'inferno', vmin = 0, vmax = 1)


# Let's take a MEAN INTENSITY PROJECTION of all images in the stack
img_stack_meanp_cyto= np.average(img_stack[:,0], axis=0)
img_stack_meanp_nuc= np.average(img_stack[:,1], axis=0)

plt.figure(figsize = (16, 8))
plt.gcf().suptitle('Mean intensity projections', size = 24)

plt.subplot(121)
plt.imshow(img_stack_meanp_cyto, cmap = 'inferno', vmin = 0, vmax = 1)

plt.subplot(122)
plt.imshow(img_stack_meanp_nuc, cmap = 'inferno', vmin = 0, vmax = 1)