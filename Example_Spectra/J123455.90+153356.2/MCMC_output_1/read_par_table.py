from astropy.io import fits

hdulist = fits.open('par_table.fits')

tbdata = hdulist[1].data

pars = []
print('\n Best-fit Parameters:')
for i in range(0,len(tbdata),1):
	print('\n %s = %0.4f, %0.4f, %0.4f' % (tbdata[i][0],tbdata[i][1],tbdata[i][2],tbdata[i][3]))
	pars.append(tbdata[i][1])

print('\n Initial Conditions for Next Run:')

print pars

