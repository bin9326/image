# %%
import numpy as np
from skimage import data
from skimage.transform import rescale
import matplotlib.pyplot as plt

# Load an example image
image = data.moon()



# Downsample the image
image_downsampled = rescale(image, 0.5)

# Upsample the image
image_upsampled = rescale(image_downsampled, 2)

# Compute the FFT of the image
fft = np.fft.fft2(image)

# Plot the original and downsampled image
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(image_downsampled)
plt.title('Downsampled')
plt.subplot(1, 3, 3)
plt.imshow(image_upsampled)
plt.title('Upsampled')

# Plot the FFT of the image
plt.figure()
plt.imshow(np.abs(fft))
plt.title('FFT')
plt.show()


# %%
import numpy as np
from skimage import data
from skimage import color
from skimage.feature import canny
from skimage.filters import sobel
import matplotlib.pyplot as plt

# Load an example image
image = data.astronaut()

# Convert the image to grayscale
gray_image = color.rgb2gray(image[...,0:3])

# Perform Sobel edge detection
sobel_edges = sobel(gray_image)

# Perform Canny edge detection
canny_edges = canny(gray_image, sigma=5)

# Plot the original and edge detected images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel')
plt.figure()
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny')
plt.show()


# %%
import numpy as np
from skimage import data
from skimage import color
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.morphology import opening
from skimage.morphology import closing
from skimage.morphology import white_tophat
from skimage.morphology import black_tophat
import matplotlib.pyplot as plt

# Load an example image
image = data.astronaut()

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Define the structuring element
kernel = np.ones((5,5),np.uint8)

# Perform erosion
eroded_image = erosion(gray_image, kernel)

# Perform dilation
dilated_image = dilation(gray_image, kernel)

# Perform opening
opened_image = opening(gray_image, kernel)

# Perform closing
closed_image = closing(gray_image, kernel)

# Perform white top hat
white_tophat_image = white_tophat(gray_image, kernel)

# Perform black top hat
black_tophat_image = black_tophat(gray_image, kernel)

# Plot the original and processed images
plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(3, 2, 2)
plt.imshow(eroded_image, cmap='gray')
plt.title('Erosion')
plt.subplot(3, 2, 3)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilation')
plt.subplot(3, 2, 4)
plt.imshow(opened_image, cmap='gray')
plt.title('Opening')
plt.subplot(3, 2, 5)
plt.imshow(closed_image, cmap='gray')
plt.title('Closing')
plt.subplot(3, 2, 6)
plt.imshow(white_tophat_image, cmap='gray')
plt.title('White Top Hat')
plt.figure()
plt.imshow(black_tophat_image, cmap='gray')
plt.title('Black Top Hat')
plt.show()


# %%
import numpy as np
from PIL import Image
from skimage import data
from skimage.feature import corner_harris, corner_peaks, blob_log, blob_dog
from skimage.feature import hog, haar_like_feature
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Load an example image
image = data.astronaut()

# Convert the image to grayscale
gray_image = rgb2gray(np.array(image))

# Extract Harris corners
corners = corner_peaks(corner_harris(gray_image), min_distance=2)

# Extract Blobs using Laplacian of Gaussian (LoG)
blobs_log = blob_log(gray_image, max_sigma=30, num_sigma=10, threshold=.1)

# Extract Blobs using Difference of Gaussian (DoG)
blobs_dog = blob_dog(gray_image, max_sigma=30, threshold=.1)

# Extract Histogram of Oriented Gradients (HOG) features
hog_features, hog_image = hog(gray_image, visualize=True)

# Extract Haar-like features
haars = haar_like_feature(gray_image, 0, 0, 20, 20, 'type-2-x')

# Plot the original and feature extracted images
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.scatter(corners[:, 1], corners[:, 0], color='cyan', marker='+', s=50)
plt.title('Corners')
plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.scatter(blobs_log[:, 1], blobs_log[:, 0], color='cyan', marker='+', s=50)
plt.title('Blobs using LoG')
plt.subplot(2, 2, 3)
plt.imshow(gray_image, cmap='gray')
plt.title


# %%
import numpy as np
from skimage import data
from skimage import color
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt

# Load an example image
image = data.astronaut()

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Define a convolution kernel
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Perform convolution on the image
convolved_image = signal.convolve2d(gray_image, kernel)

# Define a template to match
template = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# Perform template matching on the image
matched_image = signal.correlate2d(gray_image, template)

# Plot the original and convolved image
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Convolved')

# Plot the matched image
plt.figure()
plt.imshow(matched_image, cmap='gray')
plt.title('Matched')
plt.show()


# %%
import numpy as np
from skimage import data
from skimage import color
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
from skimage.filters import laplace
from skimage.filters import sobel
from scipy import signal
import matplotlib.pyplot as plt

# Load an example image
image = data.astronaut()

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Perform smoothing using a Gaussian filter
gaussian_kernel = np.outer(signal.gaussian(5, 1), signal.gaussian(5, 1))
smooth_image = signal.convolve(gray_image, gaussian_kernel)

# Perform sharpening using a Laplacian filter
sharp_image = gray_image - laplace(gray_image)

# Perform unsharp masking
unsharp_image = unsharp_mask(gray_image, radius=2, amount=1)

# Perform Sobel
sobel_image = sobel(gray_image)

# Plot the original and enhanced images
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(smooth_image, cmap='gray')
plt.title('Smooth')
plt.subplot(2, 2, 3)
plt.imshow(sharp_image, cmap='gray')
plt.title('Sharp')
plt.subplot(2, 2, 4)
plt.imshow(unsharp_image, cmap='gray')
plt.title('Unsharp Mask')
plt.figure()
plt.imshow(sobel_image, cmap='gray')
plt.title('Sobel')
plt.show()


# %%
import numpy as np
from skimage import data
from scipy.ndimage import convolve
from skimage.filters import median
import matplotlib.pyplot as plt

# Load an example image
image = data.checkerboard()

# Add random noise to the image
noisy_image = image + 0.1 * np.random.randn(*image.shape)

# Perform linear smoothing using a Gaussian filter
gaussian_kernel = np.outer(signal.gaussian(5, 1), signal.gaussian(5, 1))
smooth_image_gaussian = convolve(noisy_image, gaussian_kernel)

# Perform nonlinear smoothing using a median filter
smooth_image_median = median(noisy_image, np.ones((5,5)))

# Plot the original, noisy and smoothed images
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(noisy_image)
plt.title('Noisy')
plt.subplot(1, 3, 3)
plt.imshow(smooth_image_gaussian)
plt.title('Linear Smoothing')
plt.figure()
plt.imshow(smooth_image_median)
plt.title('Nonlinear Smoothing')
plt.show()


# %%
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from scipy import ndimage
import matplotlib.pyplot as plt

# Load an example image
image = data.astronaut()

# Convert the image to grayscale
gray_image = rgb2gray(image)

# Compute the gradient of the image in the x and y direction
gradient_x = ndimage.sobel(gray_image, axis=0)
gradient_y = ndimage.sobel(gray_image, axis=1)

# Compute the magnitude of the gradient
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Compute the Laplacian of the image
laplacian = ndimage.laplace(gray_image)

# Plot the original and enhanced images
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient')
plt.subplot(2, 2, 3)
plt.imshow(gradient_x, cmap='gray')
plt.title('Gradient X')
plt.subplot(2, 2, 4)
plt.imshow(gradient_y, cmap='gray')
plt.title('Gradient Y')
plt.figure()
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.show()


# %%
import numpy as np
from skimage import data
from skimage import exposure
from skimage import img_as_ubyte
import skimage.transform
import matplotlib.pyplot as plt

# Load an example image
image = data.moon()

# Perform log transformation on the image
log_transformed = exposure.adjust_log(image, 1)

# Perform power-law transformation on the image
gamma_transformed = exposure.adjust_gamma(image, 2)

# Perform contrast stretching on the image
contrast_stretched = exposure.rescale_intensity(image)

# Perform histogram equalization on the image
hist_eq = exposure.equalize_hist(image)

# Perform thresholding on the image
thresholded = image > 128

# Perform halftoning on the image
# halftoned = img_as_ubyte(skimage.transform(image))

# Plot the original and transformed images
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.subplot(2, 3, 2)
plt.imshow(log_transformed, cmap='gray')
plt.title('Log')
plt.subplot(2, 3, 3)
plt.imshow(gamma_transformed, cmap='gray')
plt.title('Power-law')
plt.subplot(2, 3, 4)
plt.imshow(contrast_stretched, cmap='gray')
plt.title('Contrast')
plt.subplot(2, 3, 5)
plt.imshow(hist_eq, cmap='gray')
plt.title('Histogram equalization')
# plt.subplot(2, 3, 6)
# plt.imshow(halftoned, cmap='gray')
# plt.title('Halftoning')

# Plot the thresholded image
plt.figure()
plt.imshow(thresholded, cmap='gray')
plt.title('Thresholding')
plt.show()


# %%



