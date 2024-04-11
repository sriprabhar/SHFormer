import torch
import numpy as np

from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
from numpy.lib.stride_tricks import as_strided



def npComplexToTorch(kspace_np):

    # Converts a numpy complex to torch 
    kspace_real_torch=torch.from_numpy(kspace_np.real)
    kspace_imag_torch=torch.from_numpy(kspace_np.imag)
    kspace_torch = torch.stack([kspace_real_torch,kspace_imag_torch],dim=2)
    
    return kspace_torch


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

#def cartesian_mask(shape, acc, sample_n=10, centred=False):
def cartesian_mask(shape, acc):
    sample_n=10
    centred=False
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask

def gaussian2d(pattern_shape, factor, center=None, cov=None):
    """
    Description: creates a 2D gaussian sampling pattern of a 2D image
    :param factor: sampling factor in the desired direction
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern image. It is a boolean image
    """

    N = pattern_shape[0] * pattern_shape[1]  # Image length

    factor = int(N * factor)

    if center is None:
        center = np.array([1.0 * pattern_shape[0] / 2 - 0.5, \
                           1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[0] / 5.5) ** 2, 0], \
                        [0, (1.0 * pattern_shape[1] / 5.5) ** 2]])
#         print(cov)

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor

    while (samples.shape[0] < factor):

        samples = np.random.multivariate_normal(center, cov, m * factor)
        samples = np.rint(samples).astype(int)
        indexesx = np.logical_and(samples[:, 0] >= 0, samples[:, 0] < pattern_shape[0])
        indexesy = np.logical_and(samples[:, 1] >= 0, samples[:, 1] < pattern_shape[1])
        indexes = np.logical_and(indexesx, indexesy)
        samples = samples[indexes]
        # samples[:,0] = np.clip(samples[:,0],0,input_shape[0]-1)
        # samples[:,1] = np.clip(samples[:,1],0,input_shape[1]-1)
        samples = np.unique(samples[:, 0] + 1j * samples[:, 1])
        samples = np.column_stack((samples.real, samples.imag)).astype(int)
        if samples.shape[0] < factor:
            m *= 2
            continue

    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]

    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[samples[:, 0], samples[:, 1]] = True
    return under_pattern

def gaussian_pattern(pattern_shape,factor, dim="1D",direction = "column",center=None, cov=None):
    """
    Description: creates a Gaussian distributed sampling pattern.
    :param pattern_shape: shape of the desired sampling pattern.
    :param factor: sampling factor.
    :param dim:  '1D' or '2D' sampling pattern
    :param direction: sampling direction, 'row' or 'column'. Only valid for 1D sampling
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern. It is a boolean image
    """

    if dim == "1D":
        return gaussian1d(pattern_shape,factor,direction,center,cov)
    elif dim == "2D":
        return gaussian2d(pattern_shape,factor,center,cov)
    else:
        raise("Invalid option")

def centered_circle(image_shape,radius):
    """
    Description: creates a boolean centered circle image with a pre-defined radius
    :param image_shape: shape of the desired image
    :param radius: radius of the desired circle
    :return: circle image. It is a boolean image
    """

    center_x = image_shape[0] // 2
    center_y = image_shape[1] // 2
    
    X,Y = np.indices(image_shape)
    circle_image = ((X-center_x)**2+(Y-center_y)**2) < radius**2  # type: bool

    return circle_image        
                                                                                  
def gaussian_mask(size,acc):
    us_mask1_1 = gaussian_pattern((size[0],size[1]),(1/acc),"2D")
    us_mask1_2 = centered_circle((size[0],size[1]),5)
    us_mask1_3 = np.logical_or(us_mask1_1,us_mask1_2)
    return np.fft.fftshift(us_mask1_3).astype(float)     

def CreateZeroFilledImage(fsimage, us_factor):
    fs_kspace = np.fft.fft2(fsimage, norm='ortho')
    h,w = fsimage.shape
    mask = cartesian_mask((h,w), us_factor)
    us_kspace = fs_kspace * mask
    usimg = np.abs(np.fft.ifft2(us_kspace,norm='ortho'))
    return usimg, mask, us_kspace


def CreateZeroFilledImageFn(fsimage, us_factor,mask_type):
    mask_fn= mask_type+'_mask'
    fs_kspace = np.fft.fft2(fsimage, norm='ortho')
    h,w = fsimage.shape
    mask = eval(mask_fn)((h,w), us_factor)
    us_kspace = fs_kspace * mask
    usimg = np.abs(np.fft.ifft2(us_kspace,norm='ortho'))
    return usimg, mask, us_kspace

