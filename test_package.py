import numpy as np
import pylab as pl
import radio_noise as rn

params = rn.Parameters()

telpos = np.loadtxt('lofar_hba_station_coordinate_Brentjens_8new.dat')
uv = rn.get_uv_grid_from_telescopes(telpos, params, [-2.,2.], uv_range = 2*850., mirror_points=False)

#uv = rn.get_uv_grid_from_function(lambda r: 1./r)
#params.set_num_tel(15)

#uv = np.ones((150,150))
#params.set_num_tel(15)

params.set_uv_grid(uv)

params.print_params()

noise = rn.get_visibility_noise(params)
image  = rn.get_image(noise)

print 'mean noise:', np.mean(np.abs(np.real(noise)))

pl.imshow(np.real(noise))
pl.colorbar()

pl.figure()
pl.imshow(image)
pl.colorbar()
pl.show()
