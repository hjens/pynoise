import parameters as par
import pylab as pl
import numpy as np

def plot_visibility_slice(parameters, visibility_slice = None, **kwargs):
	'''
	Make an image plot of a visibility slice.

	Only the real part is plotted

	Parameters:
		* parameters (Parameters structure) --- the structure holding the 
			parameters
	Kwargs:
		* visibility_slice = None --- the slice to plot. If none is given, a
			slice is generated from the Parameters structure.

	Additional kwargs are passed to imshow
	'''

	#Fix the visibility slice to plot
	if visibility_slice == None:
		visibility_slice = parameters.get_visibility_slice()

	visibility_slice = np.real(visibility_slice)

	#Get the uv range
	uvrange = parameters.get_uv_range()
	extent = [-uvrange/2., uvrange/2., -uvrange/2., uvrange/2.]

	#Generate title string
	title = 'Visibility noise for %d hrs observation @ %.1f MHz' % (parameters.get_t(), 
			parameters.get_nu_c())

	#Plot it
	pl.imshow(visibility_slice, extent=extent, **kwargs)
	pl.xlabel('$u/\lambda$')
	pl.ylabel('$v/\lambda$')
	cbar = pl.colorbar()
	cbar.set_label('mK')
	pl.title(title)


def plot_image_slice(parameters, image_slice = None, **kwargs):
	'''
	Make an image plot of a image slice.

	Parameters:
		* parameters (Parameters structure) --- the structure holding the 
			parameters
	Kwargs:
		* image_slice = None --- the slice to plot. If none is given, a
			slice is generated from the Parameters structure.

	Additional kwargs are passed to imshow
	'''

	#Fix the image slice to plot
	if image_slice == None:
		image_slice = parameters.get_image_slice()

	#Get the extent
	fov = parameters.get_fov()
	extent = [-fov/2., fov/2., -fov/2., fov/2.]

	#Generate title string
	title = 'Image noise for %d hrs observation @ %.1f MHz' % (parameters.get_t(), 
			parameters.get_nu_c())

	#Plot it
	pl.imshow(image_slice, extent=extent, **kwargs)
	pl.xlabel('Degrees')
	pl.ylabel('Degrees')
	cbar = pl.colorbar()
	cbar.set_label('mK')
	pl.title(title)
