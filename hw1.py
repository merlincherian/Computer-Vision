
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')

    # Create a size by size ndarray of type float32
    A = 1 / (2.0 * np.pi * sigma ** 2)
    x, y = np.mgrid[(-size + 1) / 2:(size + 1) / 2, (-size + 1) / 2:(size + 1) / 2]

    # Populate the values of the kernel. Note that the middle `pixel` should be x = 0 and y = 0.
    kernel = A * np.exp(-((x ** 2 / (2*sigma ** 2)) + (y ** 2 / (2*sigma ** 2))))

    # Normalize the values such that the sum of the kernel = 1
    sum = kernel.sum()
    rv = np.divide(kernel, sum, dtype=np.float32)

    return rv


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))


    # Check if the kernel stretches beyond the border of the image.
    if i == 0 or i >= img.shape[0]-1 or j ==0 or j >= img.shape[1]-1:
        # if so, return the input pixel at that location.
        return img[i, j]
    else:
        # perform the convolution.
        kernel = np.rot90(kernel, 2)
        klist = kernel.ravel()
        counter = 0
        sum = 0
        for u in range(i - 1, i - 1 + kernel.shape[0]):
            for v in range(j - 1, j - 1 + kernel.shape[1]):
                if u < 0 or u > img.shape[0] - 1 or v > img.shape[1] - 1 or v < 0:
                    sum += 0 * klist[counter]
                else:
                    sum += img[u, v] * klist[counter]
                counter += 1
    return sum


def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    # Make a copy of the input image to save results
    result = img
    # Populate each pixel in the input by calling convolve_pixel and return results.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i][j] = convolve_pixel(img, kernel, i, j)
    return result


def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    # Implement me
    (r, g, b) = np.dsplit(img, img.shape[-1])
    red, green, blue = [], [], []
    for i in range(len(r)):
        red.append(np.concatenate(r[i]))
        green.append(np.concatenate(g[i]))
        blue.append(np.concatenate(b[i]))
    red = np.asarray(red)
    green = np.asarray(green)
    blue = np.asarray(blue)

    return red, green, blue

def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    # Implement me
    rgb = (r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis])
    return np.concatenate(rgb, axis=-1)



"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Split it into three channels
    logging.info('Splitting it into 3 channels')
    (r, g, b) = split(inputImage)

    # compute the gaussian kernel
    logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                 (args.k, args.sigma))
    kernel = create_gaussian_kernel(args.k, args.sigma)

    # convolve it with each input channel
    logging.info('Convolving the first channel')
    r = convolve(r, kernel)
    logging.info('Convolving the second channel')
    g = convolve(g, kernel)
    logging.info('Convolving the third channel')
    b = convolve(b, kernel)

    # merge the channels back
    logging.info('Merging results')
    resultImage = merge(r, g, b)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
