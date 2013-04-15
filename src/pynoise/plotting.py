import parameters as par
import pylab as pl
import numpy as np

def plot_visibility_slice(parameters, visibility_slice = None, **kwargs):
	'''
	Make an image plot of a visibility slice.

	Only the real part is plotted

	Parameters:
		* parameters (Parameters structure) : the structure holding 
		the parameters
	Kwargs:
		* visibility_slice (numpy array) : the slice to plot. If none is given, 
		a slice is generated from the Parameters structure.

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
		* parameters (Parameters structure) : the structure holding 
		the parameters
	Kwargs:
		* image_slice (numpy array) : the slice to plot. If none is given, 
		a slice is generated from the Parameters structure.

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


def plot_uv_coverage(parameters, **kwargs):
	'''
	Make an image plot of the uv coverage of a Parameters structure

	Parameters:
		* parameters (Parameters structure) --- the structure holding the 
			parameters
	'''

	#Get the grid to plot
	uv_grid = parameters.get_uv_grid()

	#Get the uv range
	uvrange = parameters.get_uv_range()
	extent = [-uvrange/2., uvrange/2., -uvrange/2., uvrange/2.]

	#Plot it
	pl.imshow(uv_grid, extent=extent, **kwargs)
	pl.xlabel('$u/\lambda$')
	pl.ylabel('$v/\lambda$')
	pl.title('uv coverage')


def plot_uv_coverage_radial(parameters, bins = 50, **kwargs):
	'''
	Plot the radially averaged uv coverage of a Parameters structure

	Parameters:
		* parameters (Parameters structure) --- the structure holding the 
			parameters

	Kwargs:
		* bins (integer or sequence) --- if integer, make
			linearly spaced bins, otherwise treat as bin edges
	'''

	#Get the uv_grid
	uv_grid = parameters.get_uv_grid()

	#Radially average
	x,y = np.indices(uv_grid.shape)
	center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
	r = np.hypot(x-center[0],y-center[0])
	pixel = float(parameters.get_uv_range())/float(uv_grid.shape[0])
	u = r*pixel

	if not hasattr(bins, '__iter__'):
		bins = np.linspace(0, u.max(), bins)

	#Bin the data
	nbins = len(bins)-1
	du = (bins[1:]-bins[:-1])/2.
	outdata = np.zeros(nbins)
	for ri in range(nbins):
		rmin = bins[ri]
		rmax = bins[ri+1]
		idx = (u >= rmin) * (u < rmax)
		outdata[ri] = np.mean(uv_grid[idx])

	outdata[outdata != outdata] = 0.

	#Plot it
	pl.plot(bins[:-1], outdata, **kwargs)
	pl.xlabel('$|\mathbf{u}|/\lambda$')
	pl.ylabel('$uv$ density')


def plot_psf(parameters, **kwargs):
	'''
	Make an image plot of the point spread function

	Parameters:
		* parameters (Parameters structure) --- the structure holding the 
			parameters
	'''

	#Get the grid to plot
	psf = parameters.get_psf()

	#Get the extent
	fov = parameters.get_fov()
	extent = [-fov/2., fov/2., -fov/2., fov/2.]

	#Plot it
	pl.imshow(psf, extent=extent, **kwargs)
	pl.xlabel('Degrees')
	pl.ylabel('Degrees')
	pl.title('Point spread function')
