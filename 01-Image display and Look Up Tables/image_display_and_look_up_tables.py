## --------- Import all packages required ---------


import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import data, util



# Let's test this, by displaying a built-in sample image of a bush from the "data" module
image_of_a_bush = data.lfw_subset()
image_of_a_bush = image_of_a_bush[0,:,:]

# We will print the number of dimensions, the shape, and the pixel values of this image
print("The number of dimensions of the image is ", image_of_a_bush.ndim)
print("The size of the image is ", image_of_a_bush.shape[0], " by ", image_of_a_bush.shape[1], " pixels.")

# Let's go ahead and also print what exactly the image of our bush looks like to the computer
print(image_of_a_bush)
plt.clf()
plt.imshow(image_of_a_bush, cmap = 'gray')


## --------- Image display and Look Up Tables ---------

# img_stored = skimage.io.imread("./AbsoluteFilePath/SavedPhoto.fmt")

# Let's load a nice interesting image from our built-in dataset, of stained human cell nuclei undergoing mitosis
img_mitosis = data.human_mitosis()

# Let's first take a look at what kind of metadata properties skimage stores for this image, and their current values
for attr in dir(img_mitosis):
    print("obj.%s = %r" % (attr, getattr(img_mitosis, attr)))

# Let's now take a look at the image itself!
# First, as in the example above, we'll take a look at the actual pixel values that the computer sees
print(img_mitosis)

# Next, we will look at the human-viewable display that makes sense to us as an image
# We will use the default colormap for the time being...
plt.figure(figsize = (8, 8))
plt.imshow(img_mitosis)
plt.colorbar()

# Now, let's change the colormaps and try out a few examples....
# First, we'll go with grayscale - the simplest!
plt.figure(figsize = (8, 8))
plt.imshow(img_mitosis, cmap = 'gray')
plt.colorbar()


# Next, we'll invert this image in two ways....
plt.figure(figsize = (24, 8))
plt.gcf().suptitle('The two methods of inversion are different!!\n On the left, we reversed the colormap only. Middle & right, we reversed the actual pixel values!!', fontsize = 24, color='red')

# First, using the same colormap except reversed!
plt.subplot(131)
plt.imshow(img_mitosis, cmap = 'gray_r')
plt.colorbar()

# Next, using the built-in function from skimage - NOTE WHICH COLORMAP TO USE!!
plt.subplot(132)
plt.imshow(util.invert(img_mitosis), cmap = 'gray_r')
plt.colorbar()

# Clearly you can't use the same colormap to get the same display in both cases! Try the other option!
plt.subplot(133)
plt.imshow(util.invert(img_mitosis), cmap = 'gray')
plt.colorbar()

# Here is some documentation for the different colormaps (to look up spelling, etc) - https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.figure(figsize = (24, 16))

plt.subplot(231)
# Display the image with the perceptually uniform "inferno" colormap ; include a colorbar of course!
plt.imshow(img_mitosis, cmap = 'inferno')
plt.colorbar()
plt.gca().set_title('Colormap : Inferno (Perceptually uniform)')

plt.subplot(232)
# Let's now try the sequential "Wistia" colormap
plt.imshow(img_mitosis, cmap = 'Wistia')
plt.colorbar()
plt.gca().set_title('Colormap : Wistia (Sequential)')

plt.subplot(233)
# Now let's try out the diverging "seismic" scale
plt.imshow(img_mitosis, cmap = 'seismic')
plt.colorbar()
plt.gca().set_title('Colormap : Seismic (Diverging)')

plt.subplot(234)
# Next, let's try out the cyclic "hsv" scale
plt.imshow(img_mitosis, cmap = 'hsv')
plt.colorbar()
plt.gca().set_title('Colormap : HSV (Cyclic)')

plt.subplot(235)
# Out of curiosity, let's go here with the qualitative "Accent" scale
plt.imshow(img_mitosis, cmap = 'Accent')
plt.colorbar()
plt.gca().set_title('Colormap : Accent (Qualitative)')

plt.subplot(236)
# Finally, we'll go with my favourite - the "nipy_spectral" scale!
plt.imshow(img_mitosis, cmap = 'nipy_spectral')
plt.colorbar()
plt.gca().set_title('Colormap : Nipy-spectral (My fave for microscopy!)')


# Lastly, we can also display a contour map of the image
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.imshow(img_mitosis, cmap = 'gray')
plt.gca().set_title('Raw image')

plt.subplot(122)
plt.gca().contour(img_mitosis, origin='image')
plt.gca().set_title('Contour plot of the same raw image')
plt.show()


# Now let's get the bit depth of the image
print("Data type = ", img_mitosis.dtype)

# Let's confirm its range by printing the minimum and maximum values in the image
print("Minimum = ", img_mitosis.min(), " & Maximum = ", img_mitosis.max())


# The Uint8 type is on a scale of 0 to 255
# We will now convert this image to float, which is on a completely different scale of 0 to 1
img_mitosis_float = util.img_as_float(img_mitosis)

# Let's confirm the new range by printing the minimum and maximum values in the FLOAT image
print("Data type = ", img_mitosis_float.dtype)
print("Minimum = ", img_mitosis_float.min(), " & Maximum = ", img_mitosis_float.max())
# IMPORTANT NOTE - WE SAVED THE NEW BIT DEPTH AS A COPY OF THE ORIGINAL, WHILE PRESERVING THE ORIGINAL!


# We will now convert this image to 16-bit signed int, which is yet again on a completely different scale
img_mitosis_16sint = util.img_as_int(img_mitosis)

# Let's confirm the new range by printing the minimum and maximum values in the FLOAT image
print("Data type = ", img_mitosis_16sint.dtype)
print("Minimum = ", img_mitosis_16sint.min(), " & Maximum = ", img_mitosis_16sint.max())


# Now let's plot the 3 types of images side-by-side, using the same colormap
plt.figure(figsize = (24, 8))
plt.gcf().suptitle('These images LOOK identical - but they are ABSOLUTELY NOT the same!!', fontsize = 28, color='red')

plt.subplot(131)
plt.gca().set_title('Original 8-bit unsigned image') # NOTE - the gca() function stands for "get current axes" and can be used to chaneg attributes for each subplot
plt.imshow(img_mitosis, cmap = 'inferno')
plt.colorbar()

plt.subplot(132)
plt.gca().set_title('Float image')
plt.imshow(img_mitosis_float, cmap = 'inferno')
plt.colorbar()

plt.subplot(133)
plt.gca().set_title('16-bit signed image')
plt.imshow(img_mitosis_16sint, cmap = 'inferno')
plt.colorbar()
