from __future__ import division
import os
import cv2
import h5py
import random
import numbers
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import rescale
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import rotate
from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import gaussian_filter
from skimage.util import img_as_float, img_as_uint
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import uniform_filter,gaussian_filter

NR_OF_GREY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm
	
def get_all_paths(root):

    paths = []
    dirs = os.listdir(root)
    dirs.sort()
    for folder in dirs:
        if not folder.startswith('.'):  # skip hidden folders
            path = root + '/' + folder
            paths.append(path)
    return paths

# compute the barycenter of mask volume.
# mask: binary indicator
def find_centers(mask):

    x, y, z = mask.nonzero()

    midx = np.mean(x)
    midy = np.mean(y)
    midz = np.mean(z)

    return midx, midy, midz

def interpolate_data_z(ImageIn,factor):
	
	# this function interpolates the 3D data in the z direction depending on the factor
	Nx,Ny,Nz = ImageIn.shape
	x,y,z = np.linspace(1,Nx,Nx),np.linspace(1,Ny,Ny),np.linspace(1,Nz,Nz)
	interp_func = RegularGridInterpolator((x,y,z),ImageIn,method="linear")
	[xi,yi,zi] = np.meshgrid(x,y,np.linspace(1,Nz,factor*Nz),indexing='ij')
	ImageIn = interp_func( np.stack([xi,yi,zi],axis=3) )
	
	return(ImageIn)

def smooth3D_interpolate(data,threshold=20,factor=2):
	
	# this function interpolates the MRI and smoothes it in 3D
	data[data>=1] = 1
	data[data!=1] = 0
	data[data==1] = 50
	data = interpolate_data_z(data,factor)
	data = uniform_filter(data,5)
	data[data <  threshold] = 0
	data[data >= threshold] = 50
	data = data//50
	
	return(data)

###############################################################################################################################################################
### 3D Augmentation Functions
###############################################################################################################################################################
# bounding box size
bbox_size = (160, 240, 96)

def elastic_deformation_3d(volume, mask, alpha, sigma):

    shape = volume.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

    return map_coordinates(volume, indices, order=2).reshape(shape),map_coordinates(mask, indices, order=1).reshape(shape)

def random_rotation_3d(volume, mask, max_angles):

    volume1 = volume
    mask1 = mask
	
    # rotate along z-axis
    angle = random.uniform(-max_angles[2], max_angles[2])
    volume2 = rotate(volume1, angle, order=2, mode='nearest', axes=(0, 1), reshape=False)
    mask2 = rotate(mask1, angle, order=1, mode='nearest', axes=(0, 1), reshape=False)

    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    volume3 = rotate(volume2, angle, order=2, mode='nearest', axes=(0, 2), reshape=False)
    mask3 = rotate(mask2, angle, order=1, mode='nearest', axes=(0, 2), reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angles[0], max_angles[0])
    volume_rot = rotate(volume3, angle, order=2, mode='nearest', axes=(1, 2), reshape=False)
    mask_rot = rotate(mask3, angle, order=1, mode='nearest', axes=(1, 2), reshape=False)

    return volume_rot, mask_rot

def random_scale_3d(volume, mask, max_scale_deltas):

    scalex = random.uniform(1 - max_scale_deltas[0], 1 + max_scale_deltas[0])
    scaley = random.uniform(1 - max_scale_deltas[1], 1 + max_scale_deltas[1])
    scalez = random.uniform(1 - max_scale_deltas[2], 1 + max_scale_deltas[2])

    volume_zoom = zoom(volume, (scalex, scaley, scalez), order=2)
    mask_zoom = zoom(mask, (scalex, scaley, scalez), order=1)

    if volume_zoom.shape[2] < bbox_size[2]:
        top = (bbox_size[2] - volume_zoom.shape[2]) // 2
        bot = (bbox_size[2] - volume_zoom.shape[2]) - top
        volume_zoom = np.pad(volume_zoom, ((0, 0), (0, 0), (bot, top)), 'constant')
        mask_zoom = np.pad(mask_zoom, ((0, 0), (0, 0), (bot, top)), 'constant')

    elif volume_zoom.shape[2] > bbox_size[2]:
        mid = volume_zoom.shape[2] // 2
        bot = mid - bbox_size[2] // 2
        top = bot + bbox_size[2]
        volume_zoom = volume_zoom[:, :, bot:top]
        mask_zoom = mask_zoom[:, :, bot:top]
		
    volume_out = np.zeros([volume.shape[0],volume.shape[1],bbox_size[2]])
    mask_out = np.zeros([volume.shape[0],volume.shape[1],bbox_size[2]])
	
    for i in range(4,volume_zoom.shape[2]-4):
        volume_out[:,:,i] = cv2.resize(volume_zoom[:,:,i],(volume.shape[1],volume.shape[0]))
        mask_out[:,:,i] = cv2.resize(mask_zoom[:,:,i],(volume.shape[1],volume.shape[0]))
	
    return volume_out, mask_out

def random_flip_3d(volume, mask):

    if random.choice([True, False]):
        volume = volume[::-1, :, :].copy()  # here must use copy(), otherwise error occurs
        mask = mask[::-1, :, :].copy()
    if random.choice([True, False]):
        volume = volume[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()
    if random.choice([True, False]):
        volume = volume[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

    return volume, mask



###############################################################################################################################################################
### 2D Augmentation Functions
###############################################################################################################################################################

def elastic_deformation_2d(volume, mask, alpha, sigma):

    shape = volume.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
	
    return map_coordinates(volume, indices, order=2).reshape(shape),map_coordinates(mask, indices, order=1).reshape(shape)

def random_rotation_2d(volume, mask, max_angles):

    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    volume3 = rotate(volume, angle, order=2, mode='nearest', reshape=False)
    mask3 = rotate(mask, angle, order=1, mode='nearest', reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angles[0], max_angles[0])
    volume_rot = rotate(volume3, angle, order=2, mode='nearest', reshape=False)
    mask_rot = rotate(mask3, angle, order=1, mode='nearest', reshape=False)

    return volume_rot, mask_rot
	
def random_scale_2d(volume, mask, max_scale_deltas):

    scalex = random.uniform(1 - max_scale_deltas[0], 1 + max_scale_deltas[0])
    scaley = random.uniform(1 - max_scale_deltas[1], 1 + max_scale_deltas[1])

    volume_zoom = zoom(volume, (scalex, scaley), order=2)
    mask_zoom = zoom(mask, (scalex, scaley), order=1)

    volume_out = cv2.resize(volume_zoom,(volume.shape[1],volume.shape[0]))
    mask_out = cv2.resize(mask_zoom,(volume.shape[1],volume.shape[0]))
	
    return volume_out, mask_out

def random_flip_2d(volume, mask):

    if random.choice([True, False]):
        volume = volume[::-1, :].copy()  # here must use copy(), otherwise error occurs
        mask = mask[::-1, :].copy()
    else:
        volume = volume[:, ::-1].copy()
        mask = mask[:, ::-1].copy()

    return volume, mask



###############################################################################################################################################################
### Mutli-Label 2D Augmentation Functions
###############################################################################################################################################################

def multilabel_elastic_deformation_2d(volume, mask, alpha, sigma):

    shape = volume.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
	
    for i in range(0,mask.shape[2]):
        mask[:,:,i] = map_coordinates(mask[:,:,i], indices, order=1).reshape(shape)
	
    return map_coordinates(volume, indices, order=2).reshape(shape),mask

def multilabel_random_rotation_2d(volume, mask, max_angles):

    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    volume3 = rotate(volume, angle, order=2, mode='nearest', reshape=False)
	
    for i in range(0,mask.shape[2]):
        mask[:,:,i] = rotate(mask[:,:,i], angle, order=1, mode='nearest', reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angles[0], max_angles[0])
    volume_rot = rotate(volume3, angle, order=2, mode='nearest', reshape=False)
	
    for i in range(0,mask.shape[2]):
        mask[:,:,i] = rotate(mask[:,:,i], angle, order=1, mode='nearest', reshape=False)
		
    return volume_rot, mask
	
def multilabel_random_scale_2d(volume, mask, max_scale_deltas):

    scalex = random.uniform(1 - max_scale_deltas[0], 1 + max_scale_deltas[0])
    scaley = random.uniform(1 - max_scale_deltas[1], 1 + max_scale_deltas[1])

    volume_zoom = zoom(volume, (scalex, scaley), order=2)
    volume_out  = cv2.resize(volume_zoom,(volume.shape[1],volume.shape[0]))

    for i in range(0,mask.shape[2]):
        mask_zoom   = zoom(mask[:,:,i], (scalex, scaley), order=1)
        mask[:,:,i] = cv2.resize(mask_zoom,(volume.shape[1],volume.shape[0]))

    return volume_out, mask

def multilabel_random_flip_2d(volume, mask):

    if random.choice([True, False]):
        volume = volume[::-1,:].copy()  # here must use copy(), otherwise error occurs
        mask   = mask[::-1,:,:].copy()
    else:
        volume = volume[:,::-1].copy()
        mask   = mask[:,::-1,:].copy()

    return volume, mask

###############################################################################################################################################################
### CLAHE
###############################################################################################################################################################

def equalize_adapthist_3d(image, kernel_size=None,
                          clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.
    Parameters
    ----------
    image : (N1, ...,NN[, C]) ndarray
        Input image.
    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1, ...,NN[, C]) ndarray
        Equalized image.
    See Also
    --------
    equalize_hist, rescale_intensity
    Notes
    -----
    * For color images, the following steps are performed:
       - The image is converted to HSV color space
       - The CLAHE algorithm is run on the V (Value) channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.
    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))

    if kernel_size is None:
        kernel_size = tuple([image.shape[dim] // 8 for dim in range(image.ndim)])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        ValueError('Incorrect value of `kernel_size`: {}'.format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit * nbins, nbins)
    image = img_as_float(image)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins=128):
    """Contrast Limited Adaptive Histogram Equalization.
    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size: int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.
    The number of "effective" greylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    ns = [int(np.ceil(image.shape[dim] / kernel_size[dim])) for dim in range(image.ndim)]

    steps = [int(np.floor(image.shape[dim] / ns[dim])) for dim in range(image.ndim)]

    bin_size = 1 + NR_OF_GREY // nbins
    lut = np.arange(NR_OF_GREY)
    lut //= bin_size

    map_array = np.zeros(tuple(ns) + (nbins,), dtype=int)

    # Calculate greylevel mappings for each contextual region

    for inds in np.ndindex(*ns):

        region = tuple([slice(inds[dim] * steps[dim], (inds[dim] + 1) * steps[dim]) for dim in range(image.ndim)])
        sub_img = image[region]

        if clip_limit > 0.0:  # Calculate actual cliplimit
            clim = int(clip_limit * sub_img.size / nbins)
            if clim < 1:
                clim = 1
        else:
            clim = NR_OF_GREY  # Large value, do not clip (AHE)

        hist = lut[sub_img.ravel()] #lut[sub_img.ravel().astype(np.int32)]
        hist = np.bincount(hist)
        hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
        hist = clip_histogram(hist, clim)
        hist = map_histogram(hist, 0, NR_OF_GREY - 1, sub_img.size)
        map_array[inds] = hist

    # Interpolate greylevel mappings to get CLAHE image

    offsets = [0] * image.ndim
    lowers = [0] * image.ndim
    uppers = [0] * image.ndim
    starts = [0] * image.ndim
    prev_inds = [0] * image.ndim

    for inds in np.ndindex(*[ns[dim] + 1 for dim in range(image.ndim)]):

        for dim in range(image.ndim):
            if inds[dim] != prev_inds[dim]:
                starts[dim] += offsets[dim]

        for dim in range(image.ndim):
            if dim < image.ndim - 1:
                if inds[dim] != prev_inds[dim]:
                    starts[dim + 1] = 0

        prev_inds = inds[:]

        # modify edges to handle special cases
        for dim in range(image.ndim):
            if inds[dim] == 0:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = 0
                uppers[dim] = 0
            elif inds[dim] == ns[dim]:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = ns[dim] - 1
                uppers[dim] = ns[dim] - 1
            else:
                offsets[dim] = steps[dim]
                lowers[dim] = inds[dim] - 1
                uppers[dim] = inds[dim]

        maps = []
        for edge in np.ndindex(*([2] * image.ndim)):
            maps.append(map_array[tuple([[lowers, uppers][edge[dim]][dim] for dim in range(image.ndim)])])

        slices = [np.arange(starts[dim], starts[dim] + offsets[dim]) for dim in range(image.ndim)]

        interpolate(image, slices[::-1], maps, lut)

    return image


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.
    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).
    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.
    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    prev_n_excess = n_excess

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            under_mask = hist < 0
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under_mask[indices] = True
            under_mask = (under_mask) & (hist < clip_limit)
            hist[under_mask] += 1
            n_excess -= under_mask.sum()
            index += 1
        # bail if we have not distributed any excess
        if prev_n_excess == n_excess:
            break
        prev_n_excess = n_excess

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).
    It does so by cumulating the input histogram.
    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.
    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist).astype(float)
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val
    return out.astype(int)


def interpolate(image, slices, maps, lut):
    """Find the new grayscale level for a region using bilinear interpolation.
    Parameters
    ----------
    image : ndarray
        Full image.
    slices : list of array-like
       Indices of the region.
    maps : list of ndarray
        Mappings of greylevels from histograms.
    lut : ndarray
        Maps grayscale levels in image to histogram levels.
    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.
    Notes
    -----
    This function calculates the new greylevel assignments of pixels within
    a submatrix of the image. This is done by linear interpolation between
    2**image.ndim different adjacent mappings in order to eliminate boundary artifacts.
    """

    norm = np.product([slices[dim].size for dim in range(image.ndim)])  # Normalization factor

    # interpolation weight matrices
    coeffs = np.meshgrid(*tuple([np.arange(slices[dim].size) for dim in range(image.ndim)]), indexing='ij')
    coeffs = [coeff.transpose() for coeff in coeffs]

    inv_coeffs = [np.flip(coeffs[dim], axis=image.ndim - dim - 1) + 1 for dim in range(image.ndim)]

    region = tuple([slice(int(slices[dim][0]), int(slices[dim][-1] + 1)) for dim in range(image.ndim)][::-1])
    view = image[region]

    im_slice = lut[view.astype(np.int32)]

    new = np.zeros_like(view, dtype=int)
    for iedge, edge in enumerate(np.ndindex(*([2] * image.ndim))):
        edge = edge[::-1]
        new += np.product([[inv_coeffs, coeffs][edge[dim]][dim] for dim in range(image.ndim)], 0) * maps[iedge][
            im_slice]

    new = (new / norm).astype(view.dtype)
    view[::] = new
    return image
