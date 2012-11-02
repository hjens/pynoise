import numpy as np
from scipy import fftpack
from helper_functions import *


def power_spectrum_nd(input_array, box_dims):
	''' Calculate the n-dimensional power spectrum 

	The input array does not need to be cubical, but the 
	individual cells must have the same size along all axes.

	Parameters:
		* input_array (numpy array): the array to calculate PS from
		* box_dims  (list-like) tuple with the size of the box in 
		comoving Mpc along each axis

	Returns:
		The power spectrum as a numpy array of the same dimensions
		as the input array.
	'''


	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2

	# scale
	boxvol = np.product(map(float,box_dims))
	pixelsize = boxvol/(np.product(map(float,input_array.shape)))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum


def radial_average(input_array, box_dims, bins=10, weights=None):
	''' 
	Radially average data.
	Parameters:
		* input_array --- array containing the data to be averaged
		* box_dims  --- tuple with the size of the box in comoving Mpc along each axis
	kwargs:
		* bins = 10 --- the k bins. Can be an integer specifying the number of bins,
		or an array with the bin edges 
	Returns:
		Tuple containing the binned data and the bin edges
	'''
	
	if weights != None:
		input_array *= weights

	#Make an array containing distances to the center
	dim = len(input_array.shape)
	if dim == 2:
		x,y = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
		kx = (x-center[0])/box_dims[0]
		ky = (y-center[1])/box_dims[1]
	elif dim == 3:
		x,y,z = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0, (z.max()-z.min())/2.0])
		kx = (x-center[0])/box_dims[0]
		ky = (y-center[1])/box_dims[1]
		kz = (z-center[2])/box_dims[2]

	else:
		raise Exception('Check your dimensions!')

	#Calculate k values
	if dim == 3:
		k = np.sqrt(kx**2 + ky**2 + kz**2 ) * 2.*np.pi
	else:
		k = np.sqrt(kx**2 + ky**2 ) * 2.*np.pi

	#If bins is an integer, make linearly spaced bins
	if isinstance(bins,int):
		kmin = 2.*np.pi/min(box_dims)
		bins = np.linspace(kmin, k.max(), bins+1)
	
	#Bin the data
	nbins = len(bins)-1
	dk = (bins[1:]-bins[:-1])/2.
	outdata = np.zeros(nbins)
	for ki in range(nbins):
		kmin = bins[ki]
		kmax = bins[ki+1]
		idx = (k >= kmin) * (k < kmax)
		outdata[ki] = np.mean(input_array[idx])

		if weights != None:
			outdata[ki] /= weights[idx].mean()

	return outdata, bins[:-1]+dk


def power_spectrum_sphav(input_array_nd, box_size, bins=100, dimensionless=False, weights = None):
	''' 
	Calculate the spherically averaged power spectrum 

	Parameters:
		* input_array_nd  (numpy array): the data array
		* box_size  (float or list-like): size of the box in comoving Mpc. 
		Can be a single number or a tuple giving the size along each axis
	Kwargs:
		* bins (int or list-like): can be an array of k bin edges or a number of bins.
		* dimensionless (bool) if true, the dimensionless powerspectrum, k^3/(2pi^2)P(k),
			is returned
		* weights (numpy array): if given, these are the weights applied to the points
			when calculating the power spectrum. Can be calculated in the 
			parameter structure.

	Returns
		Tuple with ps, k

		ps is the power spectrum, P(k) or Delta^2(k) and k is the mid points
		of the k bins in Mpc^-1

	Example (generate noise, calculate and plot power spectrum):
		>>> par = pn.params_from_file('myparams.bin')
		>>> image = par.get_image_cube()
		>>> image_w = par.get_physical_size()
		>>> bins = 10**np.linspace(-2,1,15)
		>>> ps,k = power_spectrum_sphav(image, image_w, bins=bins)
		>>> pl.loglog(k,ps)
	'''

	if hasattr(box_size, "__iter__"):
		assert(len(box_size) == len(input_array_nd.shape))
		box_dims = box_size
	else:
		box_dims = (box_size,box_size,box_size)

	input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)	

	ps,k = radial_average(input_array, box_dims=box_dims, bins=bins, weights=weights)

	if dimensionless:
		v = k*2.**(-1./3.)*np.pi**(-2./3.)
		return ps*v**len(input_array_nd.shape), k
	return ps, k



