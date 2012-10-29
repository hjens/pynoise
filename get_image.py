import numpy as np
from scipy import fftpack

def get_image(vis, fov):
	''' 
	Convert visibility to image
	Parameters:
		* vis -- the complex visibility. array with dimensions NxN
		* fov -- the field of view in degrees
	kwargs:

	Returns 
		* image --- real array of dimensions NxN, in mK
	'''


	#Change image coordinates
	vis_shift = fftpack.ifftshift(vis)

	#Backward FT
	#image = fftpack.ifftn(vis_shift)
	image = fftpack.ifft2(vis_shift)

	#ifft2 divides output by 1/n, where n i the number of Fourier modes
	n = np.product(image.shape)
	image *= n

	#Calculate pixel size
	u_min = 180./(fov*np.pi)
	pixelsize_uv = u_min**2

	#Scale image
	image *= pixelsize_uv

	#Take the real part and shift back
	image = np.real(image)
	image = fftpack.fftshift(image)


	return image
