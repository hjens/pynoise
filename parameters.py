import numpy as np
import uvgrid 
import uvnoise 
import get_image
import helper_functions as hf
import cPickle as pickle

class Parameters:
	''' This class acts as a storage unit for various instrument and experiment parameters '''

	#-----------  Handle parameters ------------

	def __init__(self):
		''' Set all parameters to some default values '''
		self._epsilon = 1.
		self._Tsys = 350. #K
		self._dnu = 1. #MHz
		self._Aphys = 500. #m^2
		self._Aeff = 500. #m^2
		self._nu_c = 150. #MHz
		self._uv_grid = None
		self._uv_range = 2000.
		self._nu_range = [0.,0.] #MHz. Used when calculating cubes
		self._T = 400. #hours
		self._num_tel = 10
		self._num_pol = 1

	def print_params(self):
		''' Print the current values of all parameters '''
		print 'Current parameter values:'
		print '\t * Antenna efficiency (epsilon):', self._epsilon
		print '\t * System temperature (K):', self._Tsys
		print '\t * Channel width (MHz):', self._dnu
		print '\t * Effective area (m^2):', self.get_aeff()
		print '\t * Physical area (m^2):', self._Aphys
		print '\t * Field of view (deg):', self.get_fov()
		print '\t * Central frequency (MHz):', self._nu_c
		print '\t * nu_max, nu_min (MHz):', self._nu_range
		print '\t * Integration time (hours): ', self._T
		print '\t * Number of telescopes: ', self._num_tel
		print '\t * uv range (u_max-u_min in wavelengths):', self._uv_range
		print '\t * Number of polarizations: ', self._num_pol

	def write_params_to_file(self,f):
		if type(f) == str:
			f = open(f, 'w')
		f.write('Antenna efficiency (epsilon): %f\n'% self._epsilon)
		f.write('System temperature (K): %f\n'% self._Tsys)
		f.write('Bandwidth (MHz): %f\n'% self._dnu)
		f.write('Effective area (m^2): %f\n' % self._Aeff)
		f.write('Physical area (m^2): %f\n' % self._Aphys)
		f.write('Central frequency (MHz): %f\n'% self._nu_c)
		f.write('nu_max, nu_min (MHz): %d, %d\n' % ( self._nu_range) )
		f.write('Integration time (hours): %f \n'% self._T)
		f.write('Number of telescopes: %f\n'% self._num_tel)
		f.write('uv range (u_max-u_min in wavelengths): %f\n' % self._uv_range)
		f.write('Number of polarizations: %d'% self._num_pol)
		f.close()


	#--------------Save and load Parameters object --------------------
	def save_to_file(self,f):
		''' Save (pickle) parameters object to file f
		Parameters:
			* f --- binary file or filename
		'''
		if type(f) == str:
			fi = open(f,'wb')
		else:
			fi = f
		pickle.dump(self,fi)
		fi.close()

	#def  __getstate__(self):
	#	''' This method ensures that _last_vis_noise and _last_vis_noise_cube
	#	are not pickled '''
	#	odict = self.__dict__.copy() 
	#	del odict['_last_vis_noise']
	#	del odict['_last_vis_noise_cube'] 
	#	return odict


	#-------------- Methods for calculating the uv grid	 --------------
	def set_uv_grid_from_telescopes(self, tel_positions, ha_range, decl = 90, ha_step = 50, mirror_points=False):
		'''
		Calculate the uv coverage of a telescope array and set the uv_array parameter.

		Parameters:
			* tel_positions --- array with telescope x,y,z positions in m, cartesian geocentric coordinates
			Must have the shape (N,3) where N is the number of telescopes. Can also be a string specifying a text file with the telescope positions
			* ha_range --- tuple with start and stop hour angle (in hours)

		kwargs:
			* ha_step = 50 --- time resolution in uv calculation in seconds
			* decl = 90 --- declination of the source, in degrees
			* mirror_points = False --- whether to include (-u,-v) points

		'''

		#Check if we should first read the telescope positions
		if type(tel_positions) == str:
			pos = np.loadtxt(tel_positions)
		else:
			pos = tel_positions

		#Calculate the grid
		grid = uvgrid.get_uv_grid_from_telescopes(pos, self.get_fov(), self.get_nu_c(), ha_range, decl, ha_step, self.get_uv_range(), mirror_points)

		self._uv_grid = grid


	def set_uv_grid_from_function(self, rho_uv):
		'''
		Calculate the uv coverage based on a radial function and set the uv_array parameter.

		Parameters:
			* rho_uv --- callable taking one parameter. The function should give the baseline
			density as a function of baseline length.
		'''

		grid = uvgrid.get_uv_grid_from_function(rho_uv, self.get_fov(), self.get_uv_range())
		self._uv_grid = grid

	def set_uv_grid_from_antenna_distribution(self, rho_ant, num_points_phi=301, num_points_r=2001):
		'''
		Calculate uv coverage grid from a function of r giving the 
		antenna density as a function of distance from the array center.
		Still somewhat experimental. Please check the results for numerical
		problems.

		Parameters:
			* rho_ant --- callable taking one parameter. This function should give the 
			density of antennae in the array as a function of distance (in meters) 
			from the array center
		kwargs:
			* num_points_phi = 301 --- number of sample points for phi when integrating
			* num_points_r = 10001 --- number of sample points for r when integrating

		Returns:
			* uv grid, normalized so that the integral is 1
		'''

		func = uvgrid.get_rho_uv_from_antenna_distribution(rho_ant, self.get_fov(), self.get_uv_range(), 
				self.get_wavel(), num_points_phi, num_points_r)
		self.set_uv_grid_from_function(func)

	def get_uv_weights(self, los_axis=0):
		''' Get weights for use with the powerspectrum routines
		Parameters:
		kwargs:
		Returns:
			weights
		'''

		#TODO: take care of non-cubes

		#Weights
		n = self._uv_grid.shape[0]
		weights = np.zeros((n,n,n))
		if los_axis == 0:
			weights += self._uv_grid[np.newaxis,:,:]
		elif los_axis == 1:
			weights += self._uv_grid[:,np.newaxis,:]
		elif los_axis == 2:
			weights += self._uv_grid[:,:,np.newaxis]
		else:
			raise Exception('Invalid los axis')
		weights *= weights

		return weights

	def apply_uv_taper(self, taper_func):
		''' 
		Apply a tapering function to the uv grid. 
		If the uv grid is recalculated, the tapering function
		has to be applied again.

		Parameters:
		* taper_func --- callable. A function of one variable - the 
			baseline length in wavelengths - which will be multiplied
			with the uv grid

		'''

		if self.get_uv_grid() == None:
			print 'No uv grid specified'
			return

		self._uv_grid = uvgrid.get_tapered_uv_grid(self._uv_grid, self._uv_range, taper_func)
	

	#----------- Calculate noise in image and vis space-----
	def get_visibility_slice(self, seed=None):
		''' Calculate a noise realization in visibility space. Also save the noise internally for image calculation later.
		Parameters:
		kwargs:
			* seed = None --- the random seed. If None, the Python default is used

		Returns:
			* noise --- complex array of same dimensions as uv grid, containing real and imaginary noise in mK'''
		if uvgrid == None:
			raise Exception('No uv grid specified')
		noise = uvnoise.get_visibility_noise(self.get_tsys(), self.get_epsilon(), self.get_aeff(), self.get_dnu(), self.get_nu_c(), self.get_t(), self.get_num_tel(), self.get_uv_grid(), self.get_num_pol(), seed)

		return noise

	def get_image_slice(self, visibility_slice=None):
		'''
		Calculate noise in image space, from the last visibility noise calculated. If no visibility 
		noise has been supplied, a slice will be calculated, but not returned.
		kwargs:
			* visibility_slice = None --- the visibility slice to use as input. If none, a new
			slice will be calculated

		Returns:
			* image --- real array with same dimensions as uv grid, in mK
		'''
		if visibility_slice == None:
			visibility_slice = self.get_visibility_slice()
		image = get_image.get_image(visibility_slice, self.get_fov())

		return image

	#----------- Make noise and image cubes ------------------
	def get_visibility_cube(self, seed=None):
		'''
		Calculate a noise cube in visibility space. The extent along the frequency
		axis is determined by d_nu and nu_range. To make a cube, first run
		set_nu_range_cubic()
		TODO: allow for uv-coverage recalculation for each frequency

		Parameters:
		kwargs:
			* seed = None --- the random seed. If None, the Python default is used
		Returns:
			* Noise cube in visibility space with frequency as the first index.
			
		'''

		if self._nu_range[1] >= self._nu_c or self._nu_range[0] <= self._nu_c:
			raise Exception('Invalid frequency range when calculating noise cube')

		if seed != None:
			np.random.seed(seed)

		#TODO: verify that the dnu and frequency range are consistent
		gridn = self.get_uv_grid().shape[0]
		noise_cube = np.zeros((gridn,gridn,gridn)) + np.zeros((gridn,gridn,gridn))*1.j
		for i in range(gridn):
			noise_cube[i,:,:] = self.get_visibility_slice()

		return noise_cube

	def get_image_cube(self, visibility_cube = None, seed=None):
		'''
		Calculate a noise cube in image space. The calculation is based on the 
		visibility_noise cube supplied. If this is None, it is calculated.
		Parameters:
		kwargs:
			* seed = None --- the random seed. If None, the Python default is used
			* visibility_cube = None --- the visibility cube to use as input. If None
				a temporary cube will be calculated.
		Returns:
			* Noise cube in image space with frequency as the first index.
			
		'''
		if visibility_cube == None:
			visibility_cube = self.get_visibility_cube(seed=seed)

		image_cube = np.zeros(visibility_cube.shape)
		for i in range(image_cube.shape[0]):
			image_cube[i,:,:] = self.get_image_slice(visibility_slice=visibility_cube[i,:,:])

		return image_cube


	#----------- Calculate various useful quantities ----------

	def get_wavel(self):
		''' Get wavelength in m '''
		return 300./self.get_nu_c()

	def get_z(self):
		''' Get redshift '''
		return hf.nu_to_z(self.get_nu_c())

	def get_fov(self):
		''' Calculate FoV in degrees'''
		d = np.sqrt(4.*self._Aphys/np.pi)
		fov = self.get_wavel()/d
		return fov*180./np.pi

	def set_nu_range_cubic(self):
		''' Set the parameters nu_range and dnu so that the
		result of get_noise_cube and get_image_cube have the same
		comoving extent along the frequency axis as along the sides
		The uv grid must be set prior to running this method.
		'''
		#Determine the size of the box
		redsh = hf.nu_to_z(self.get_nu_c())
		cdist = hf.cdist(redsh)
		boxl = cdist*self.get_fov()*np.pi/180.

		#Determine frequency range
		z_lower = hf.cdist_to_z(cdist-boxl/2.)
		z_upper = hf.cdist_to_z(cdist+boxl/2.)
		nu_upper = hf.z_to_nu(z_upper)
		nu_lower = hf.z_to_nu(z_lower)
		self.set_nu_range((nu_lower,nu_upper))

		#Determine frequency step
		if self.get_uv_grid() == None or self.get_uv_grid().shape[0] == 0:
			raise Exception('uv grid must be set before running set_nu_range_cubic')
		gridn = self.get_uv_grid().shape[0]
		self.set_dnu( (nu_lower-nu_upper)/float(gridn) )

	def get_psf(self):
		''' Calculate the point spread function based on the
		current uv grid. The psf is normalized so that the sum
		is 1.

		Returns:
			* psf --- array with the same grid dimensions as uv_grid, self.get_fov() across '''

		#Make a uv grid mask
		uv_mask = self.get_uv_grid().copy()
		uv_mask[self.get_uv_grid() < 1.e-15] = 0.
		uv_mask[self.get_uv_grid() > 1.e-15] = 1.
		psf = get_image.get_image(uv_mask, self.get_fov())
		psf /= psf.sum()
		return psf

	def set_fov(self, fov):
		'''
		Set the physical area to give the desired field of view
		Parameters:
			* fov --- the field of view in degrees
		'''

		fov *= np.pi/180.
		a = (self.get_wavel()/(2.*fov))**2*np.pi
		self.set_aphys(a)

	def get_physical_size(self):
		''' Calculate the physical size of the box
		Returns:
			box size in comoving Mpc '''
		redsh = hf.nu_to_z(self.get_nu_c())
		cdist = hf.cdist(redsh)
		l = cdist * self.get_fov() * np.pi/180.
		return l


	#-----------  Getter/setter methods ---------------

	def set_epsilon(self, epsilon):
		self._epsilon = epsilon
	def get_epsilon(self):
		return self._epsilon

	def set_tsys(self, Tsys):
		self._Tsys = Tsys
	def get_tsys(self):
		return self._Tsys

	def set_dnu(self, dnu):
		self._dnu = dnu
	def get_dnu(self):
		return self._dnu

	def set_aphys(self, Aphys):
		self._Aphys = Aphys
	def get_aphys(self):
		return self._Aphys

	def set_aeff(self, Aeff):
		self._Aeff = Aeff
	def get_aeff(self):
		return self._Aeff

	def set_nu_c(self, nu_c):
		self._nu_c = nu_c
	def get_nu_c(self):
		return self._nu_c

	def set_t(self, t):
		self._T = t
	def get_t(self):
		return self._T

	def get_uv_grid(self): 
		return self._uv_grid
	def set_uv_grid(self, uv_grid): #See above for more setter methods
		self._uv_grid = uv_grid

	def set_num_tel(self, num_tel):
		self._num_tel = num_tel
	def get_num_tel(self):
		return self._num_tel

	def set_uv_range(self, uv_range):
		self._uv_range = uv_range
	def get_uv_range(self):
		return self._uv_range

	def get_nu_range(self):
		return self._nu_range
	def set_nu_range(self, nu_range): #See also set_nu_range_cubic
		self._nu_range = nu_range

	def get_last_visibility_noise(self):
		return self._last_vis_noise

	def get_last_visibility_cube(self):
		return self._last_vis_cube
	
	def get_num_pol(self):
		return self._num_pol
	def set_num_pol(self, num_pol):
		if num_pol == 1 or num_pol == 2:
			self._num_pol = num_pol
		else:
			print 'WARNING: invalid number of polarizations: %d' % num_pol