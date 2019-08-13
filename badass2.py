#####################################################

# Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS2,version 4.1)
# by Remington Oliver Sexton
# contact (email): rsext001@ucr.edu
# Disclaimer: still in development; use at own risk.

#####################################################
import numpy as np
from numpy.polynomial import legendre, hermite
from numpy import linspace, meshgrid
import scipy.optimize as op
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy import optimize, linalg, special, fftpack
from scipy.interpolate import griddata, interp1d
from scipy.stats import kde, norm
from scipy.integrate import simps
from astropy.io import fits
import glob
import time
from os import path
import os
import shutil
import sys
import psutil
import emcee
from astropy.stats import mad_std,sigma_clip
import ntpath
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM
import re
import natsort
plt.style.use('dark_background') # For cool tron-style dark plots
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 100000

#####################################################


#### Find Nearest Function #######################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

##################################################################################


#### Convert Seconds to Minutes ##################################################

# Python Program to Convert seconds 
# into hours, minutes and seconds 
  
def time_convert(seconds): 
    seconds = seconds % (24. * 3600.) 
    hour = seconds // 3600.
    seconds %= 3600.
    minutes = seconds // 60.
    seconds %= 60.
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)

##################################################################################


#### Setup Directory Structure ###################################################

def setup_dirs(work_dir):

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split('(\d+)', text) ]
    
    # Get list of folders in work_dir:
    folders = glob.glob(work_dir+'MCMC_output_*')
    folders.sort(key=natural_keys)
    if (len(folders)==0):
        print(' Folder has not been created.  Creating MCMC_output folder...')
        # Create the first MCMC_output file starting with index 1
        os.mkdir(work_dir+'MCMC_output_1')
        run_dir = work_dir+'MCMC_output_1/' # running directory
        prev_dir = None
        if os.path.exists(run_dir+'/templates/')==False:
            os.mkdir(run_dir+'/templates/')
    else: 
        # Get last folder name
        s = folders[-1]
        result = re.search('MCMC_output_(.*)', s)
        # The next folder is named with this number
        fnum = str(int(result.group(1))+1)
        prev_num = str(int(result.group(1)))
        # Create the first MCMC_output file starting with index 1
        new_fold = work_dir+'MCMC_output_'+fnum+'/'
        prev_fold = work_dir+'MCMC_output_'+prev_num+'/'
        # print new_fold
        os.mkdir(new_fold)
        run_dir = new_fold
        if os.path.exists(prev_fold+'MCMC_chain.csv')==True:
            prev_dir = prev_fold
        else:
            prev_dir = prev_fold
        if os.path.exists(run_dir+'/templates/')==False:
            os.mkdir(run_dir+'/templates/')
        print(' Storing MCMC_output in %s' % run_dir)
        # break

    return run_dir,prev_dir

##################################################################################


#### Determine fitting region ####################################################

def determine_upper_bound(first_good,last_good):
	# Set some rules for the upper spectrum limit
	# Indo-US Library of Stellar Templates has a upper limit of 9464
	if ((last_good>=7000.) & (last_good<=9464.)) and (last_good-first_good>=500.): # cap at 7000 A
		auto_upp = last_good #7000.
	elif ((last_good>=6750.) & (last_good<=7000.)) and (last_good-first_good>=500.): # include Ha/[NII]/[SII] region
		auto_upp = last_good
	elif ((last_good>=6400.) & (last_good<=6750.)) and (last_good-first_good>=500.): # omit H-alpha/[NII] region if we can't fit all lines in region
		auto_upp = 6400.
	elif ((last_good>=5050.) & (last_good<=6400.)) and (last_good-first_good>=500.): # Full MgIb/FeII region
		auto_upp = last_good
	elif ((last_good>=4750.) & (last_good<=5025.)) and (last_good-first_good>=500.): # omit H-beta/[OIII] region if we can't fit all lines in region
		auto_upp = 4750.
	elif ((last_good>=4400.) & (last_good<=4750.)) and (last_good-first_good>=500.):
		auto_upp = last_good
	elif ((last_good>=4300.) & (last_good<=4400.)) and (last_good-first_good>=500.): # omit H-gamma region if we can't fit all lines in region
		auto_upp = 4300.
	elif ((last_good>=3500.) & (last_good<=4300.)) and (last_good-first_good>=500.): # omit H-gamma region if we can't fit all lines in region
		auto_upp = last_good
	elif (last_good-first_good>=500.):
		print('\n Not enough spectrum to fit! ')
		auto_upp = None 
	return auto_upp


def determine_fit_reg(file,good_thresh,run_dir,fit_reg='auto'):
	# Open spectrum file
	hdu = fits.open(file)
	specobj = hdu[2].data
	z = specobj['z'][0]
	t = hdu['COADD'].data
	lam_gal = (10**(t['loglam']))/(1+z)

	#mask = ((lam_gal > 4230.) & (lam_gal < 5150.))

	#lam_gal = lam_gal#[mask]
	gal  = t['flux']#[mask]
	ivar = t['ivar']#[mask]
	and_mask = t['and_mask']#[mask]
	# Edges of wavelength vector
	first_good = lam_gal[0]
	last_good  = lam_gal[-1]
	# print first_good,last_good

	if ((fit_reg=='auto') or (fit_reg is None) or (fit_reg=='full')):
		# The lower limit of the spectrum must be the lower limit of our stellar templates
		auto_low = np.max([3500.,first_good]) # Indo-US Library of Stellar Templates has a lower limit of 3460
		auto_upp = determine_upper_bound(first_good,last_good)
		if (auto_upp is not None):
			new_fit_reg = (int(auto_low),int(auto_upp))	
		elif (auto_upp is None):
			new_fit_reg = None
			return None, None
	elif ((isinstance(fit_reg,tuple)==True) or (isinstance(fit_reg,list)==True) ):
		# Check to see if tuple/list makes sense
		if ((fit_reg[0]>fit_reg[1]) or (fit_reg[1]<fit_reg[0])): # if boundaries overlap
			print('\n Fitting boundary error. \n')
			new_fit_reg = None
			return None, None
		elif ((fit_reg[1]-fit_reg[0])<500.0): # if fitting region is < 500 A
			print('\n Your fitting region is suspiciously small... \n')
			new_fit_reg = None
			return None, None
		else:
			man_low = np.max([3500.,first_good,fit_reg[0]])
			# print man_low
			man_upper_bound  = determine_upper_bound(fit_reg[0],fit_reg[1])
			man_upp = np.min([man_upper_bound,fit_reg[1],last_good])
			new_fit_reg = (int(man_low),int(man_upp))

	# Determine number of good pixels in new fitting region
	mask = ((lam_gal >= new_fit_reg[0]) & (lam_gal <= new_fit_reg[1]))
	igood = np.where((gal[mask]>0) & (ivar[mask]>0) & (and_mask[mask]==0))[0]
	ibad  = np.where(and_mask[mask]!=0)[0]
	good_frac = (len(igood)*1.0)/len(gal[mask])

	if 0:
		##################################################################################
		fig = plt.figure(figsize=(14,6))
		ax1 = fig.add_subplot(1,1,1)

		ax1.plot(lam_gal,gal,linewidth=0.5)
		ax1.axvline(new_fit_reg[0],linestyle='--',color='xkcd:yellow')
		ax1.axvline(new_fit_reg[1],linestyle='--',color='xkcd:yellow')

		ax1.scatter(lam_gal[mask][ibad],gal[mask][ibad],color='red')
		ax1.set_ylabel(r'$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
		ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)')

		plt.tight_layout()
		plt.savefig(run_dir+'good_pixels.pdf',fmt='pdf',dpi=150)
	##################################################################################
	# Close the fits file 
	hdu.close()
	##################################################################################

	return new_fit_reg,good_frac

##################################################################################


#### Galactic Extinction Correction ##############################################

def ccm_unred(wave, flux, ebv, r_v=""):
    """ccm_unred(wave, flux, ebv, r_v="")
    Deredden a flux vector using the CCM 1989 parameterization 
    Returns an array of the unreddened flux
  
    INPUTS:
    wave - array of wavelengths (in Angstroms)
    dec - calibrated flux array, same number of elements as wave
    ebv - colour excess E(B-V) float. If a negative ebv is supplied
          fluxes will be reddened rather than dereddened     
  
    OPTIONAL INPUT:
    r_v - float specifying the ratio of total selective
          extinction R(V) = A(V)/E(B-V). If not specified,
          then r_v = 3.1
            
    OUTPUTS:
    funred - unreddened calibrated flux array, same number of 
             elements as wave
             
    NOTES:
    1. This function was converted from the IDL Astrolib procedure
       last updated in April 1998. All notes from that function
       (provided below) are relevant to this function 
       
    2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
       ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction 
       can be represented with a CCM curve, if the proper value of 
       R(V) is supplied.
    4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989, ApJ, 339,474)
    5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the 
       original flux vector.
    6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1).    But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.
 
    7. For the optical/NIR transformation, the coefficients from 
       O'Donnell (1994) are used
  
    >>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 ) 
    array([9.7976e+012, 1.12064e+07, 32287.1])
    """
    wave = np.array(wave, float)
    flux = np.array(flux, float)
    
    if wave.size != flux.size: raise TypeError, 'ERROR - wave and flux vectors must be the same size'
    
    if not bool(r_v): r_v = 3.1 

    x = 10000.0/wave
    npts = wave.size
    a = np.zeros(npts, float)
    b = np.zeros(npts, float)
    
    ###############################
    #Infrared
    
    good = np.where( (x > 0.3) & (x < 1.1) )
    a[good] = 0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)
    
    ###############################
    # Optical & Near IR

    good = np.where( (x  >= 1.1) & (x < 3.3) )
    y = x[good] - 1.82
    
    c1 = np.array([ 1.0 , 0.104,   -0.609,    0.701,  1.137, \
                  -1.718,   -0.827,    1.647, -0.505 ])
    c2 = np.array([ 0.0,  1.952,    2.908,   -3.989, -7.985, \
                  11.102,    5.491,  -10.805,  3.347 ] )

    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    ###############################
    # Mid-UV
    
    good = np.where( (x >= 3.3) & (x < 8) )   
    y = x[good]
    F_a = np.zeros(np.size(good),float)
    F_b = np.zeros(np.size(good),float)
    good1 = np.where( y > 5.9 )    
    
    if np.size(good1) > 0:
        y1 = y[good1] - 5.9
        F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
        F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3

    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
    
    ###############################
    # Far-UV
    
    good = np.where( (x >= 8) & (x <= 11) )   
    y = x[good] - 8.0
    c1 = [ -1.073, -0.628,  0.137, -0.070 ]
    c2 = [ 13.670,  4.257, -0.420,  0.374 ]
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    # Applying Extinction Correction
    
    a_v = r_v * ebv
    a_lambda = a_v * (a + b/r_v)
    
    funred = flux * 10.0**(0.4*a_lambda)   

    return funred #,a_lambda



#### Prepare SDSS spectrum for pPXF ################################################

def sdss_prepare(file,fit_reg,temp_dir,run_dir,plot=False):
	"""
	Prepare an SDSS spectrum for pPXF, returning all necessary 
	parameters. 
	
	file: fully-specified path of the spectrum 
	z: the redshift; we use the SDSS-measured redshift
	fit_reg: (min,max); tuple specifying the minimum and maximum 
	        wavelength bounds of the region to be fit. 
	
	"""
	# Load the data
	hdu = fits.open(file)

	specobj = hdu[2].data
	z = specobj['z'][0]
	ra = specobj['PLUG_RA'][0]
	dec = specobj['PLUG_DEC'][0]

	t = hdu['COADD'].data
	
	# Only use the wavelength range in common between galaxy and stellar library.
	# Determine limits of spectrum vs templates
	# mask = ( (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409)) )
	fit_min,fit_max = float(fit_reg[0]),float(fit_reg[1])
	mask = ( (t['loglam'] > np.log10(fit_min*(1+z))) & (t['loglam'] < np.log10(fit_max*(1+z))) )
	
	# Unpack the spectra
	flux = t['flux'][mask]
	galaxy = flux  
	# SDSS spectra are already log10-rebinned
	loglam_gal = t['loglam'][mask] # This is the observed SDSS wavelength range, NOT the rest wavelength range of the galaxy
	lam_gal = 10**loglam_gal
	ivar = t['ivar'][mask]# inverse variance
	noise = np.sqrt(1.0/ivar)

	c = 299792.458                  # speed of light in km/s
	frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
	dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom
	# print('\n Size of every pixel: %s (A)' % dlam_gal)
	wdisp = t['wdisp'][mask]        # Intrinsic dispersion of every pixel, in pixels units
	fwhm_gal = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in Angstroms
	velscale = np.log(frac)*c       # Constant velocity scale in km/s per pixel
	
	# If the galaxy is at significant redshift, one should bring the galaxy
	# spectrum roughly to the rest-frame wavelength, before calling pPXF
	# (See Sec2.4 of Cappellari 2017). In practice there is no
	# need to modify the spectrum in any way, given that a red shift
	# corresponds to a linear shift of the log-rebinned spectrum.
	# One just needs to compute the wavelength range in the rest-frame
	# and adjust the instrumental resolution of the galaxy observations.
	# This is done with the following three commented lines:
	#
	lam_gal = lam_gal/(1+z)  # Compute approximate restframe wavelength
	fwhm_gal = fwhm_gal/(1+z)   # Adjust resolution in Angstrom

	# Read the list of filenames from the Single Stellar Population library
	# by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
	# of the library is included for this example with permission
	# num_temp = 10 # number of templates
	# temp_list = glob.glob(temp_dir + '/Mun1.30Z*.fits')#[:num_temp]
	temp_list = glob.glob(temp_dir + '/*.fits')#[:num_temp]

	temp_list = natsort.natsorted(temp_list) # Sort them in the order they appear in the directory
	# fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
	fwhm_tem = 1.35 # Indo-US Template Library FWHM

	# Extract the wavelength range and logarithmically rebin one spectrum
	# to the same velocity scale of the SDSS galaxy spectrum, to determine
	# the size needed for the array which will contain the template spectra.
	#
	hdu = fits.open(temp_list[0])
	ssp = hdu[0].data
	h2 = hdu[0].header
	lam_temp = np.array(h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1']))
	# By cropping the templates we save some fitting time
	mask_temp = ( (lam_temp > (fit_min-200.)) & (lam_temp < (fit_max+200.)) )
	ssp = ssp[mask_temp]
	lam_temp = lam_temp[mask_temp]

	lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
	sspNew = log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
	templates = np.empty((sspNew.size, len(temp_list)))
	
	# Interpolates the galaxy spectral resolution at the location of every pixel
	# of the templates. Outside the range of the galaxy spectrum the resolution
	# will be extrapolated, but this is irrelevant as those pixels cannot be
	# used in the fit anyway.
	fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

	# Convolve the whole Vazdekis library of spectral templates
	# with the quadratic difference between the SDSS and the
	# Vazdekis instrumental resolution. Logarithmically rebin
	# and store each template as a column in the array TEMPLATES.
	
	# Quadratic sigma difference in pixels Vazdekis --> SDSS
	# The formula below is rigorously valid if the shapes of the
	# instrumental spectral profiles are well approximated by Gaussians.
	#
	# In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
	# In principle it should never happen and a higher resolution template should be used.
	#
	fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
	sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

	for j, fname in enumerate(temp_list):
	    hdu = fits.open(fname)
	    ssp = hdu[0].data
	    ssp = ssp[mask_temp]
	    ssp = gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
	    sspNew,loglam_temp,velscale_temp = log_rebin(lamRange_temp, ssp, velscale=velscale)#[0]
	    templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
	
	# The galaxy and the template spectra do not have the same starting wavelength.
	# For this reason an extra velocity shift DV has to be applied to the template
	# to fit the galaxy spectrum. We remove this artificial shift by using the
	# keyword VSYST in the call to PPXF below, so that all velocities are
	# measured with respect to DV. This assume the redshift is negligible.
	# In the case of a high-redshift galaxy one should de-redshift its
	# wavelength to the rest frame before using the line below (see above).
	#
	dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s
	vsyst = dv

	# Here the actual fit starts. The best fit is plotted on the screen.
	# Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
	#
	vel = 0.0#c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
	start = [vel, 200.,0.0,0.0]  # (km/s), starting guess for [V, sigma]

	#################### Correct for galactic extinction ##################

	co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
	try: 
	    table = IrsaDust.get_query_table(co,section='ebv')
	    ebv = table['ext SandF mean'][0]
	except: 
	    ebv = 0.04
	galaxy = ccm_unred(lam_gal,galaxy,ebv)

	#######################################################################

	npix = galaxy.shape[0] # number of output pixels
	ntemp = np.shape(templates)[1]# number of templates
	
	# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
	temp_fft,npad = template_rfft(templates) # we will use this throughout the code
	
	################################################################################   
	if plot:
		# Plot the galaxy+ templates
		fig = plt.figure(figsize=(14,6))
		ax1 = fig.add_subplot(2,1,1)
		ax2 = fig.add_subplot(2,1,2)
		ax1.step(lam_gal,galaxy,label='Galaxy',linewidth=0.5)
		# ax1.fill_between(lam_gal,galaxy-noise,galaxy+noise,color='gray',alpha=0.5,linewidth=0.5)
		ax1.step(lam_gal,noise,label='Error Spectrum',linewidth=0.5,color='gray')
		ax1.axhline(0.0,color='white',linewidth=0.5,linestyle='--')
		ax2.plot(np.exp(loglam_temp),templates[:,:],alpha=0.5,label='Template',linewidth=0.5)
		ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)',fontsize=12)
		ax2.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)',fontsize=12)
		ax1.set_ylabel(r'$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
		ax2.set_ylabel(r'Normalized Flux',fontsize=12)
		ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
		ax2.set_xlim(np.min(lam_gal),np.max(lam_gal))
		ax1.legend(loc='best')
		plt.tight_layout()
		# plt.savefig(run_dir+'sdss_prepare.png',dpi=300,fmt='png')
		plt.savefig(run_dir+'sdss_prepare.pdf',dpi=150,fmt='pdf')
	################################################################################
	# Close the fits file
	hdu.close()
	################################################################################
	
	
	return lam_gal,galaxy,templates,noise,velscale,vsyst,temp_list,z,ebv,npix,ntemp,temp_fft,npad

##################################################################################




#### Initialize Parameters #######################################################


def initialize_mcmc(lam_gal,galaxy,fit_reg,fit_type='init',fit_feii=True,fit_losvd=True,fit_host=True,fit_power=True,fit_broad=True,fit_narrow=True,fit_outflows=True,tie_narrow=True):
	# Issue warnings for dumb options
	if ((fit_narrow==False) & (fit_outflows==True)): # why would you fit outflow without narrow lines?
		raise ValueError('\n Why would you fit outflows without narrow lines? Turn on narrow line component! \n')

	################################################################################
	# Initial conditions for some parameters
	max_flux = np.max(galaxy)
	total_flux_init = np.median(galaxy)#np.median(galaxy[(lam_gal>5025.) & (lam_gal<5800.)])
	# cont_flux_init = 0.01*(np.median(galaxy))
	feii_flux_init= (0.1*np.median(galaxy))

	if (((fit_reg[0]+25) < 6085. < (fit_reg[1]-25))==True):
		fevii6085_amp_init= (np.max(galaxy[(lam_gal>6085.-25.) & (lam_gal<6085.+25.)]))
	if (((fit_reg[0]+25) < 5722. < (fit_reg[1]-25))==True):
		fevii5722_amp_init= (np.max(galaxy[(lam_gal>5722.-25.) & (lam_gal<5722.+25.)]))
	if (((fit_reg[0]+25) < 6302. < (fit_reg[1]-25))==True):
		oi_amp_init= (np.max(galaxy[(lam_gal>6302.-25.) & (lam_gal<6302.+25.)]))
	if (((fit_reg[0]+25) < 3727. < (fit_reg[1]-25))==True):
		oii_amp_init= (np.max(galaxy[(lam_gal>3727.-25.) & (lam_gal<3727.+25.)]))
	if (((fit_reg[0]+25) < 3870. < (fit_reg[1]-25))==True):
		neiii_amp_init= (np.max(galaxy[(lam_gal>3870.-25.) & (lam_gal<3870.+25.)]))
	if (((fit_reg[0]+25) < 4102. < (fit_reg[1]-25))==True):
		hd_amp_init= (np.max(galaxy[(lam_gal>4102.-25.) & (lam_gal<4102.+25.)]))
	if (((fit_reg[0]+25) < 4341. < (fit_reg[1]-25))==True):
		hg_amp_init= (np.max(galaxy[(lam_gal>4341.-25.) & (lam_gal<4341.+25.)]))
	if (((fit_reg[0]+25) < 4862. < (fit_reg[1]-25))==True):
		hb_amp_init= (np.max(galaxy[(lam_gal>4862.-25.) & (lam_gal<4862.+25.)]))
	if (((fit_reg[0]+25) < 5007. < (fit_reg[1]-25))==True):
		oiii5007_amp_init = (np.max(galaxy[(lam_gal>5007.-25.) & (lam_gal<5007.+25.)]))
	if (((fit_reg[0]+25) < 6564. < (fit_reg[1]-25))==True):
		ha_amp_init = (np.max(galaxy[(lam_gal>6564.-25.) & (lam_gal<6564.+25.)]))
	if (((fit_reg[0]+25) < 6725. < (fit_reg[1]-25))==True):
		sii_amp_init = (np.max(galaxy[(lam_gal>6725.-15.) & (lam_gal<6725.+15.)]))
	################################################################################
    
	mcmc_input = {} # dictionary of parameter dictionaries

	#### Host Galaxy ###############################################################
	# Galaxy template amplitude
	if ((fit_type=='init') or ((fit_type=='final') and (fit_losvd==False))) and (fit_host==True):
		print(' Fitting a host-galaxy template.')

		mcmc_input['gal_temp_amp'] = ({'name':'gal_temp_amp',
		                			   'label':'$A_\mathrm{gal}$',
		                			   'init':0.5*total_flux_init,
		                			   'plim':(0.0,max_flux),
		                			   'pcolor':'blue',
		                			   })
	# Stellar velocity
	if ((fit_type=='final') and (fit_losvd==True)):
		print(' Fitting the stellar LOSVD.')
		mcmc_input['stel_vel'] = ({'name':'stel_vel',
		                   		   'label':'$V_*$',
		                   		   'init':100. ,
		                   		   'plim':(-500.,500.),
		                   		   'pcolor':'blue',
		                   		   })
		# Stellar velocity dispersion
		mcmc_input['stel_disp'] = ({'name':'stel_disp',
		                   			'label':'$\sigma_*$',
		                   			'init':100.0,
		                   			'plim':(30.0,400.),
		                   			'pcolor':'dodgerblue',
		                   			})
	##############################################################################

	#### AGN Power-Law ###########################################################
	if (fit_power==True):
		print(' Fitting AGN power-law continuum.')
		# AGN simple power-law amplitude
		mcmc_input['power_amp'] = ({'name':'power_amp',
		                   		   'label':'$A_\mathrm{power}$',
		                   		   'init':(0.5*total_flux_init),
		                   		   'plim':(0.0,max_flux),
		                   		   'pcolor':'orangered',
		                   		   })
		# AGN simple power-law slope
		mcmc_input['power_slope'] = ({'name':'power_slope',
		                   			 'label':'$m_\mathrm{power}$',
		                   			 'init':-1.0  ,
		                   			 'plim':(-4.0,2.0),
		                   			 'pcolor':'salmon',
		                   			 })
		
	##############################################################################

	#### FeII Templates ##########################################################
	if (fit_feii==True):
		print(' Fitting narrow and broad FeII.')
		# Narrow FeII amplitude
		mcmc_input['na_feii_amp'] = ({'name':'na_feii_amp',
		                   			  'label':'$A_\mathrm{Na\;FeII}$',
		                   			  'init':feii_flux_init,
		                   			  'plim':(0.0,total_flux_init),
		                   			  'pcolor':'sandybrown',
		                   			  })
		# Broad FeII amplitude
		mcmc_input['br_feii_amp'] = ({'name':'br_feii_amp',
		                   			  'label':'$A_\mathrm{Br\;FeII}$',
		                   			  'init':feii_flux_init,
		                   			  'plim':(0.0,total_flux_init),
		                   			  'pcolor':'darkorange',
		                   			  })
	##############################################################################

	#### Emission Lines ##########################################################

	#### Jenna's Lines ##############################################
	if 0:#((fit_narrow==True) and ((fit_reg[1]-25.) > 5722.) ):
		print(" Fitting Jenna's Lines: ")
		if (((fit_reg[0]+25.) < 6302.046 < (fit_reg[1]-25.))==True):
			print('          Fitting narrow [OI]6300 emission.')
			# Na. [OI]6300 Core Amplitude
			mcmc_input['na_oi_6300_amp'] = ({'name':'na_oi_6300_amp',
			                   				    'label':'$A_{\mathrm{[OI]6300}}$',
			                   				    'init':(oi_amp_init-total_flux_init),
			                   				    'plim':(0.0,max_flux),
			                   				    'pcolor':'green',
			                   				    })
			# Na. [OI]6300 Core FWHM
			mcmc_input['na_oi_6300_fwhm'] = ({'name':'na_oi_6300_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[OI]6300}}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
			# Na. [OI]6300 Core VOFF
			mcmc_input['na_oi_6300_voff'] = ({'name':'na_oi_6300_voff',
			                   					 'label':'$\mathrm{VOFF}_{\mathrm{[OI]6300}}$',
			                   					 'init':0.,
			                   					 'plim':(-1000.,1000.),
			                   					 'pcolor':'palegreen',
			                   					 })
		if (((fit_reg[0]+25.) < 5722. < (fit_reg[1]-25.))==True):
			print('          Fitting narrow [FeVII]5722 emission.')
			# Na. [OI]6300 Core Amplitude
			mcmc_input['na_fevii_5722_amp'] = ({'name':'na_fevii_5722_amp',
			                   				    'label':'$A_{\mathrm{[FeVII]5722}}$',
			                   				    'init':(fevii5722_amp_init-total_flux_init),
			                   				    'plim':(0.0,max_flux),
			                   				    'pcolor':'green',
			                   				    })
			# Na. [OI]6300 Core FWHM
			mcmc_input['na_fevii_5722_fwhm'] = ({'name':'na_fevii_5722_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[FeVII]5722}}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
			# Na. [OI]6300 Core VOFF
			mcmc_input['na_fevii_5722_voff'] = ({'name':'na_fevii_5722_voff',
			                   					 'label':'$\mathrm{VOFF}_{\mathrm{[FeVII]5722}}$',
			                   					 'init':0.,
			                   					 'plim':(-1000.,1000.),
			                   					 'pcolor':'palegreen',
			                   					 })
		if (((fit_reg[0]+25.) < 6085. < (fit_reg[1]-25.))==True):
			print('          Fitting narrow [FeVII]6085 emission.')
			# Na. [OI]6300 Core Amplitude
			mcmc_input['na_fevii_6085_amp'] = ({'name':'na_fevii_6085_amp',
			                   				    'label':'$A_{\mathrm{[FeVII]6085}}$',
			                   				    'init':(fevii6085_amp_init-total_flux_init),
			                   				    'plim':(0.0,max_flux),
			                   				    'pcolor':'green',
			                   				    })
			# Na. [OI]6300 Core FWHM
			mcmc_input['na_fevii_6085_fwhm'] = ({'name':'na_fevii_6085_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[FeVII]6085}}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
			# Na. [OI]6300 Core VOFF
			mcmc_input['na_fevii_6085_voff'] = ({'name':'na_fevii_6085_voff',
			                   					 'label':'$\mathrm{VOFF}_{\mathrm{[FeVII]6085}}$',
			                   					 'init':0.,
			                   					 'plim':(-1000.,1000.),
			                   					 'pcolor':'palegreen',
			                   					 })

	###################################################################

	#### Narrow [OII] Doublet ##############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 3727.092 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow H-delta emission.')
		# Na. [OII]3727 Core Amplitude
		mcmc_input['na_oii3727_core_amp'] = ({'name':'na_oii3727_core_amp',
		                   				    'label':'$A_{\mathrm{[OII]3727}}$',
		                   				    'init':(oii_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. [OII]3727 Core FWHM
			mcmc_input['na_oii3727_core_fwhm'] = ({'name':'na_oii3727_core_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[OII]3727}}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
		# Na. [OII]3727 Core VOFF
		mcmc_input['na_oii3727_core_voff'] = ({'name':'na_oii3727_core_voff',
		                   					 'label':'$\mathrm{VOFF}_{\mathrm{[OII]3727}}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })
		# Na. [OII]3729 Core Amplitude
		mcmc_input['na_oii3729_core_amp'] = ({'name':'na_oii3729_core_amp',
		                   				    'label':'$A_{\mathrm{[OII]3729}}$',
		                   				    'init':(oii_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })

	###################################################################
	#### Narrow [NeIII]3870 ##############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 3869.81 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow [NIII]3870 emission.')
		# Na. [NIII]3870 Core Amplitude
		mcmc_input['na_neiii_core_amp'] = ({'name':'na_neiii_core_amp',
		                   				    'label':'$A_{\mathrm{[NeIII]}}$',
		                   				    'init':(neiii_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. [NIII]3870 Core FWHM
			mcmc_input['na_neiii_core_fwhm'] = ({'name':'na_neiii_core_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[NeIII]}}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
		# Na. [NIII]3870 Core VOFF
		mcmc_input['na_neiii_core_voff'] = ({'name':'na_neiii_core_voff',
		                   					 'label':'$\mathrm{VOFF}_{\mathrm{[NeIII]}}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })

	###################################################################

	#### Narrow H-delta ###############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 4102.89 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow H-delta emission.')
		# Na. H-delta Core Amplitude
		mcmc_input['na_Hd_amp'] = ({'name':'na_Hd_amp',
		                   				    'label':'$A_{\mathrm{H}\delta}$',
		                   				    'init':(hd_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. H-delta Core FWHM
			mcmc_input['na_Hd_fwhm'] = ({'name':'na_Hd_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\delta}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
		# Na. H-delta Core VOFF
		mcmc_input['na_Hd_voff'] = ({'name':'na_Hd_voff',
		                   					 'label':'$\mathrm{VOFF}_{\mathrm{H}\delta}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })

	##############################################################################

	#### Broad Line H-delta ######################################################
	# if ((fit_broad==True) and ((((fit_reg[0]+25.) < 4102.89 < (fit_reg[1]-25.))==True))):
	# 	print(' Fitting broadline H-delta.')
	# 	# Br. H-delta amplitude
	# 	mcmc_input['br_Hd_amp'] = ({'name':'br_Hd_amp',
	# 	                   			'label':'$A_{\mathrm{Br.\;Hd}}$' ,
	# 	                   			'init':(hd_amp_init-total_flux_init)/2.0  ,
	# 	                   			'plim':(0.0,max_flux),
	# 	                   			'pcolor':'steelblue',
	# 	                   			})
	# 	# Br. H-delta FWHM
	# 	mcmc_input['br_Hd_fwhm'] = ({'name':'br_Hd_fwhm',
	# 	               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hd}}$',
	# 	               	   			 'init':2500.,
	# 	               	   			 'plim':(0.0,10000.),
	# 	               	   			 'pcolor':'royalblue',
	# 	               	   			 })
	# 	# Br. H-delta VOFF
	# 	mcmc_input['br_Hd_voff'] = ({'name':'br_Hd_voff',
	# 	               	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Hd}}$',
	# 	               	   		 	 'init':0.,
	# 	               	   		 	 'plim':(-1000.,1000.),
	# 	               	   		 	 'pcolor':'turquoise',
	# 	               	   		 	 })
	##############################################################################

	#### Narrow H-gamma/[OIII]4363 ###############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 4341.68 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow H-gamma/[OIII]4363 emission.')
		# Na. H-gamma Core Amplitude
		mcmc_input['na_Hg_amp'] = ({'name':'na_Hg_amp',
		                   				    'label':'$A_{\mathrm{H}\gamma}$',
		                   				    'init':(hg_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. H-gamma Core FWHM
			mcmc_input['na_Hg_fwhm'] = ({'name':'na_Hg_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\gamma}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
		# Na. H-gamma Core VOFF
		mcmc_input['na_Hg_voff'] = ({'name':'na_Hg_voff',
		                   					 'label':'$\mathrm{VOFF}_{\mathrm{H}\gamma}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })
		# Na. [OIII]4363 Core Amplitude
		mcmc_input['oiii4363_core_amp'] = ({'name':'oiii4363_core_amp',
		                   				    'label':'$A_\mathrm{[OIII]4363\;Core}$',
		                   				    'init':(hg_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. [OIII]4363 Core FWHM
			mcmc_input['oiii4363_core_fwhm'] = ({'name':'oiii4363_core_fwhm',
			                   					 'label':'$\mathrm{FWHM}_\mathrm{[OIII]4363\;Core}$',
			                   					 'init':250.,
			                   					 'plim':(0.0,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
		# Na. [OIII]4363 Core VOFF
		mcmc_input['oiii4363_core_voff'] = ({'name':'oiii4363_core_voff',
		                   					 'label':'$\mathrm{VOFF}_\mathrm{[OIII]4363\;Core}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })

	##############################################################################

	#### Broad Line H-gamma ######################################################
	if ((fit_broad==True) and ((((fit_reg[0]+25.) < 4341.68 < (fit_reg[1]-25.))==True))):
		print(' Fitting broadline H-gamma.')
		# Br. H-beta amplitude
		mcmc_input['br_Hg_amp'] = ({'name':'br_Hg_amp',
		                   			'label':'$A_{\mathrm{Br.\;Hg}}$' ,
		                   			'init':(hg_amp_init-total_flux_init)/2.0  ,
		                   			'plim':(0.0,max_flux),
		                   			'pcolor':'steelblue',
		                   			})
		# Br. H-beta FWHM
		mcmc_input['br_Hg_fwhm'] = ({'name':'br_Hg_fwhm',
		               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hg}}$',
		               	   			 'init':2500.,
		               	   			 'plim':(0.0,10000.),
		               	   			 'pcolor':'royalblue',
		               	   			 })
		# Br. H-beta VOFF
		mcmc_input['br_Hg_voff'] = ({'name':'br_Hg_voff',
		               	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Hg}}$',
		               	   		 	 'init':0.,
		               	   		 	 'plim':(-1000.,1000.),
		               	   		 	 'pcolor':'turquoise',
		               	   		 	 })
	##############################################################################



	#### Narrow Hb/[OIII] Core ###########################################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow H-beta/[OIII]4959,5007 emission.')
		# Na. [OIII]5007 Core Amplitude
		mcmc_input['oiii5007_core_amp'] = ({'name':'oiii5007_core_amp',
		                   				    'label':'$A_\mathrm{[OIII]5007\;Core}$',
		                   				    'init':(oiii5007_amp_init-total_flux_init),
		                   				    'plim':(0.0,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		# If tie_narrow=True, then all line widths are tied to [OIII]5007, as well as their outflows (currently H-alpha/[NII]/[SII] outflows only)
		# Na. [OIII]5007 Core FWHM
		mcmc_input['oiii5007_core_fwhm'] = ({'name':'oiii5007_core_fwhm',
		                   					 'label':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Core}$',
		                   					 'init':250.,
		                   					 'plim':(0.0,1000.),
		                   					 'pcolor':'limegreen',
		                   					 })
		# Na. [OIII]5007 Core VOFF
		mcmc_input['oiii5007_core_voff'] = ({'name':'oiii5007_core_voff',
		                   					 'label':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Core}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })
		# Na. H-beta amplitude
		mcmc_input['na_Hb_amp'] = ({'name':'na_Hb_amp',
		                   		 	'label':'$A_{\mathrm{Na.\;Hb}}$' ,
		                   		 	'init':(hb_amp_init-total_flux_init) ,
		                   		 	'plim':(0.0,max_flux),
		                   		 	'pcolor':'gold',
		                   		 	})
		# Na. H-beta FWHM tied to [OIII]5007 FWHM
		# Na. H-beta VOFF
		mcmc_input['na_Hb_voff'] = ({'name':'na_Hb_voff',
		                   			 'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Hb}}$',
		                   			 'init':0.,
		                   			 'plim':(-1000,1000.),
		                   			 'pcolor':'yellow',
		                   			 })
	##############################################################################

	#### Hb/[OIII] Outflows ######################################################
	if ((fit_narrow==True) and (fit_outflows==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
		print(' Fitting H-beta/[OIII]4959,5007 outflows.')
		# Br. [OIII]5007 Outflow amplitude
		mcmc_input['oiii5007_outflow_amp'] = ({'name':'oiii5007_outflow_amp',
		                   					   'label':'$A_\mathrm{[OIII]5007\;Outflow}$' ,
		                   					   'init':(oiii5007_amp_init-total_flux_init)/2.,
		                   					   'plim':(0.0,max_flux),
		                   					   'pcolor':'mediumpurple',
		                   					   })
		# Br. [OIII]5007 Outflow FWHM
		mcmc_input['oiii5007_outflow_fwhm'] = ({'name':'oiii5007_outflow_fwhm',
		                   						'label':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Outflow}$',
		                   						'init':450.,
		                   						'plim':(0.0,2500.),
		                   						'pcolor':'darkorchid',
		                   						})
		# Br. [OIII]5007 Outflow VOFF
		mcmc_input['oiii5007_outflow_voff'] = ({'name':'oiii5007_outflow_voff',
		                   						'label':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Outflow}$',
		                   						'init':-50.,
		                   						'plim':(-2000.,2000.),
		                   						'pcolor':'orchid',
		                   						})
		# Br. [OIII]4959 Outflow is tied to all components of [OIII]5007 outflow
	##############################################################################

	#### Broad Line H-beta ############################################################
	if ((fit_broad==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
		print(' Fitting broadline H-beta.')
		# Br. H-beta amplitude
		mcmc_input['br_Hb_amp'] = ({'name':'br_Hb_amp',
		                   			'label':'$A_{\mathrm{Br.\;Hb}}$' ,
		                   			'init':(hb_amp_init-total_flux_init)/2.0  ,
		                   			'plim':(0.0,max_flux),
		                   			'pcolor':'steelblue',
		                   			})
		# Br. H-beta FWHM
		mcmc_input['br_Hb_fwhm'] = ({'name':'br_Hb_fwhm',
		               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$',
		               	   			 'init':2500.,
		               	   			 'plim':(0.0,10000.),
		               	   			 'pcolor':'royalblue',
		               	   			 })
		# Br. H-beta VOFF
		mcmc_input['br_Hb_voff'] = ({'name':'br_Hb_voff',
		               	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Hb}}$',
		               	   		 	 'init':0.,
		               	   		 	 'plim':(-1000.,1000.),
		               	   		 	 'pcolor':'turquoise',
		               	   		 	 })
	##############################################################################

	##############################################################################

	#### Narrow Ha/[NII]/[SII] ###################################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 6732.67 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow Ha/[NII]/[SII] emission.')
		# Na. [NII]6585 Amp.
		mcmc_input['nii6585_core_amp'] = ({'name':'nii6585_core_amp',
		                   				   'label':'$A_\mathrm{[NII]6585\;Core}$',
		                   				   'init':(ha_amp_init-total_flux_init)*0.75,
		                   				   'plim':(0.0,max_flux),
		                   				   'pcolor':'green',
		                   				   })
		if (tie_narrow==False):
			# Na. [NII]6585 FWHM
			mcmc_input['nii6585_core_fwhm'] = ({'name':'nii6585_core_fwhm',
			                   					'label':'$\mathrm{FWHM}_\mathrm{[NII]6585\;Core}$',
			                   					'init':250.,
			                   					'plim':(0.0,1000.),
			                   					'pcolor':'limegreen',
			                   					})
	
		# Na. [NII]6585 VOFF
		mcmc_input['nii6585_core_voff'] = ({'name':'nii6585_core_voff',
		                   					'label':'$\mathrm{VOFF}_\mathrm{[NII]6585\;Core}$',
		                   					'init':0.,
		                   					'plim':(-1000.,1000.),
		                   					'pcolor':'palegreen',
		                   					})
	

		# Na. H-alpha amplitude
		mcmc_input['na_Ha_amp'] = ({'name':'na_Ha_amp',
		                   		 	'label':'$A_{\mathrm{Na.\;Ha}}$' ,
		                   		 	'init':(ha_amp_init-total_flux_init) ,
		                   		 	'plim':(0.0,max_flux),
		                   		 	'pcolor':'gold',
		                   		 	})
		# Na. H-alpha FWHM tied to [NII]6585 FWHM
		# Na. H-alpha VOFF
		mcmc_input['na_Ha_voff'] = ({'name':'na_Ha_voff',
		                   			 'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Ha}}$',
		                   			 'init':0.,
		                   			 'plim':(-1000,1000.),
		                   			 'pcolor':'yellow',
		                   			 })

		# Na. [SII]6732 Amp.
		mcmc_input['sii6732_core_amp'] = ({'name':'sii6732_core_amp',
		                   				   'label':'$A_\mathrm{[SII]6732\;Core}$',
		                   				   'init':(sii_amp_init-total_flux_init),
		                   				   'plim':(0.0,max_flux),
		                   				   'pcolor':'green',
		                   				   })
	
		# Na. [SII]6732 VOFF
		mcmc_input['sii6732_core_voff'] = ({'name':'sii6732_core_voff',
		                   					'label':'$\mathrm{VOFF}_\mathrm{[SII]6732\;Core}$',
		                   					'init':0.,
		                   					'plim':(-1000.,1000.),
		                   					'pcolor':'palegreen',
		                   					})
		# Na. [SII]6718 Amp.
		mcmc_input['sii6718_core_amp'] = ({'name':'sii6718_core_amp',
		                   				   'label':'$A_\mathrm{[SII]6718\;Core}$',
		                   				   'init':(sii_amp_init-total_flux_init),
		                   				   'plim':(0.0,max_flux),
		                   				   'pcolor':'green',
		                   				   })
	#### Ha/[NII]/[SII] Outflows ######################################################
	if ((fit_narrow==True) and (fit_outflows==True) and ((((fit_reg[0]+25.) < 6732.67 < (fit_reg[1]-25.))==True))):
		print(' Fitting Ha/[NII]/[SII] outflows.')
		# Br. [OIII]5007 Outflow amplitude
		mcmc_input['nii6585_outflow_amp'] = ({'name':'nii6585_outflow_amp',
		                   					   'label':'$A_\mathrm{[NII]6585\;Outflow}$' ,
		                   					   'init':(ha_amp_init-total_flux_init)*0.25,
		                   					   'plim':(0.0,max_flux),
		                   					   'pcolor':'mediumpurple',
		                   					   })
		if (tie_narrow==False):
			# Br. [OIII]5007 Outflow FWHM
			mcmc_input['nii6585_outflow_fwhm'] = ({'name':'nii6585_outflow_fwhm',
			                   						'label':'$\mathrm{FWHM}_\mathrm{[NII]6585\;Outflow}$',
			                   						'init':450.,
			                   						'plim':(0.0,2500.),
			                   						'pcolor':'darkorchid',
			                   						})
		# Br. [OIII]5007 Outflow VOFF
		mcmc_input['nii6585_outflow_voff'] = ({'name':'nii6585_outflow_voff',
		                   						'label':'$\mathrm{VOFF}_\mathrm{[NII]6585\;Outflow}$',
		                   						'init':-50.,
		                   						'plim':(-2000.,2000.),
		                   						'pcolor':'orchid',
		                   						})
		# All components [NII]6585 of outflow are tied to all outflows of the Ha/[NII]/[SII] region

	##############################################################################

	#### Broad Line H-alpha ###########################################################

	if ((fit_broad==True) and ((((fit_reg[0]+25.) < 6732.67 < (fit_reg[1]-25.))==True))):
		print(' Fitting broadline H-alpha.')
		# Br. H-alpha amplitude
		mcmc_input['br_Ha_amp'] = ({'name':'br_Ha_amp',
		                   			'label':'$A_{\mathrm{Br.\;Ha}}$' ,
		                   			'init':(ha_amp_init-total_flux_init)/2.0  ,
		                   			'plim':(0.0,max_flux),
		                   			'pcolor':'steelblue',
		                   			})
		# Br. H-alpha FWHM
		mcmc_input['br_Ha_fwhm'] = ({'name':'br_Ha_fwhm',
		               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Ha}}$',
		               	   			 'init':2500.,
		               	   			 'plim':(0.0,10000.),
		               	   			 'pcolor':'royalblue',
		               	   			 })
		# Br. H-alpha VOFF
		mcmc_input['br_Ha_voff'] = ({'name':'br_Ha_voff',
		               	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Ha}}$',
		               	   		 	 'init':0.,
		               	   		 	 'plim':(-1000.,1000.),
		               	   		 	 'pcolor':'turquoise',
		               	   		 	 }) 

	

	##############################################################################

	##############################################################################
	
	# return param_names,param_labels,params,param_limits,param_init,param_colors
	return mcmc_input

##################################################################################

#### Outflow Test ################################################################
def outflow_test(lam_gal,galaxy,noise,run_dir,velscale,mcbs_niter):
	fit_reg = (4400,5800)
	param_dict = initialize_mcmc(lam_gal,galaxy,fit_reg=fit_reg,fit_type='init',
	                             fit_feii=True,fit_losvd=True,
	                             fit_power=False,fit_broad=True,
	                             fit_narrow=True,fit_outflows=True)

	# for key in param_dict:
	# 	print key
	# if 1: sys.exit()

	gal_temp = galaxy_template(lam_gal,age=None)
	na_feii_temp,br_feii_temp = initialize_feii(lam_gal,velscale,fit_reg)

	# Create mask to mask out parts of spectrum; should speed things up
	if 1:
		mask = np.where( (lam_gal > fit_reg[0]) & (lam_gal < fit_reg[1]) )
		lam_gal       = lam_gal[mask]
		galaxy        = galaxy[mask]
		noise         = noise[mask]
		gal_temp      = gal_temp[mask]
		na_feii_temp  = na_feii_temp[mask]
		br_feii_temp  = br_feii_temp[mask]

	rdict,sigma = max_likelihood(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
                                        None,None,None,velscale,None,None,run_dir,monte_carlo=True,niter=mcbs_niter)

	
	for key in rdict:
		print("          %s = %0.2f +/- %0.2f" % (key, rdict[key]['med'],rdict[key]['std']) )
	print('          sigma = %0.2f' % sigma)

	# Determine the significance of outflows
	# Outflow criteria:
	#	(1) (FWHM_outflow - dFWHM_outflow) > (FWHM_core + dFWHM_core)
	cond1 = ((rdict['oiii5007_outflow_fwhm']['med']-rdict['oiii5007_outflow_fwhm']['std']) > (rdict['oiii5007_core_fwhm']['med']+rdict['oiii5007_core_fwhm']['std']))
	if (cond1==True):
		print('          Outflow FWHM condition: Pass')
	elif (cond1==False):
		print('          Outflow FWHM condition: Fail')
	#	(2) (VOFF_outflow + dVOFF_outflow) < (VOFF_core - dVOFF_core)
	cond2 = ((rdict['oiii5007_outflow_voff']['med']+rdict['oiii5007_outflow_voff']['std']) < (rdict['oiii5007_core_voff']['med']-rdict['oiii5007_core_voff']['std']))
	if (cond2==True):
		print('          Outflow VOFF condition: Pass')
	elif (cond2==False):
		print('          Outflow VOFF condition: Fail')
	#	(3) (AMP_outflow - dAMP_outflow) > sigma
	cond3 = ((rdict['oiii5007_outflow_amp']['med']-rdict['oiii5007_outflow_amp']['std']) > sigma)
	if (cond3==True):
		print('          Outflow amplitude condition: Pass')
	elif (cond3==False):
		print('          Outflow amplitude condition: Fail')

	if (all([cond1,cond2,cond3])==True):
		return True
	elif (any([cond1,cond2,cond3])==False):
		return False

##################################################################################


#### Maximum Likelihood Fitting ##################################################

def max_likelihood(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
				   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
				   fit_type='init',move_temp=False,output_model=False,
				   monte_carlo=False,niter=25):
	
	# This function performs an initial maximum likelihood
	# estimation to acquire robust initial parameters.
	param_names  = [param_dict[key]['name'] for key in param_dict ]
	params       = [param_dict[key]['init'] for key in param_dict ]
	bounds       = [param_dict[key]['plim'] for key in param_dict ]

	# for i in range(0,len(params),1):
	# 	print param_names[i], params[i], bounds[i]
	# if 1: sys.exit()

	# Constraints for Outflow components
	def oiii_amp_constraint(p):
		# (core_amp >= outflow_amp) OR (core_amp - outflow_amp >= 0)
	    return p[param_names.index('oiii5007_core_amp')]-p[param_names.index('oiii5007_outflow_amp')]

	def oiii_fwhm_constraint(p):
		# (outflow_fwhm >= core_fwhm) OR (outflow_fwhm - core_fwhm >= 0)
		return p[param_names.index('oiii5007_outflow_fwhm')]-p[param_names.index('oiii5007_core_fwhm')]
	def oiii_voff_constraint(p):
		# (core_voff >= outflow_voff) OR (core_voff - outflow_voff >= 0)
	    return p[param_names.index('oiii5007_core_voff')]-p[param_names.index('oiii5007_outflow_voff')]
	def nii_amp_constraint(p):
		# (core_amp >= outflow_amp) OR (core_amp - outflow_amp >= 0)
	    return p[param_names.index('nii6585_core_amp')]-p[param_names.index('nii6585_outflow_amp')]
	def nii_fwhm_constraint(p):
		# (outflow_fwhm >= core_fwhm) OR (outflow_fwhm - core_fwhm >= 0)
		return p[param_names.index('nii6585_outflow_fwhm')]-p[param_names.index('nii6585_core_fwhm')]
	def nii_voff_constraint(p):
		# (core_voff >= outflow_voff) OR (core_voff - outflow_voff >= 0)
	    return p[param_names.index('nii6585_core_voff')]-p[param_names.index('nii6585_outflow_voff')]

	cons1 = [{'type':'ineq','fun': oiii_amp_constraint  },
	         {'type':'ineq','fun': oiii_fwhm_constraint },
	         {'type':'ineq','fun': oiii_voff_constraint }]

	cons2 = [{'type':'ineq','fun': nii_amp_constraint   },
	         {'type':'ineq','fun': nii_fwhm_constraint  },
	         {'type':'ineq','fun': nii_voff_constraint  }]
	
	cons3 = [{'type':'ineq','fun': oiii_amp_constraint  },
	         {'type':'ineq','fun': oiii_fwhm_constraint },
	         {'type':'ineq','fun': oiii_voff_constraint },
	         {'type':'ineq','fun': nii_amp_constraint   },
	         {'type':'ineq','fun': nii_fwhm_constraint  },
	         {'type':'ineq','fun': nii_voff_constraint  }]

	# Perform maximum likelihood estimation for initial guesses of MCMC fit
	# If model includes narrow lines and outflows
	
	if (all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
											 'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==True) and \
	   (all(comp in param_names for comp in ['nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
	   										 'na_Ha_amp','na_Ha_voff',
	   										 'sii6732_core_amp','sii6732_core_voff','sii6718_core_amp',
	   										 'nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==False):
		print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] region...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,move_temp,output_model),\
		     				 method='SLSQP', bounds = bounds, constraints=cons1, options={'ftol':1.0e-10,'maxiter':10000,'disp': True})

	elif (all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
											 'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==False) and \
	     (all(comp in param_names for comp in ['nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
	   										 'na_Ha_amp','na_Ha_voff',
	   										 'sii6732_core_amp','sii6732_core_voff','sii6718_core_amp',
	   										 'nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==True):
		print('\n Performing max. likelihood fitting.  Constraining outflow components for Ha/[NII]/[SII] region...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,move_temp,output_model),\
		     				 method='SLSQP', bounds = bounds, constraints=cons2, options={'ftol':1.0e-10,'maxiter':10000,'disp': True})


	elif (all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
												  'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff',
												  'nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
												  'na_Ha_amp','na_Ha_voff',
												  'sii6732_core_amp','sii6732_core_voff',
												  'sii6718_core_amp',
												  'nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==True):
		print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] and Ha/[NII]/[SII] regions...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,move_temp,output_model),\
		     				 method='SLSQP', bounds = bounds, constraints=cons3, options={'ftol':1.0e-10,'maxiter':10000,'disp': True})




	elif all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff','oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==False:
		print('\n Not fitting outflow components...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,move_temp,output_model),\
		     				 method='SLSQP', bounds = bounds, options={'maxiter':10000,'disp': True})
	elap_time = (time.time() - start_time)


	if 1: # Set to 1 to plot and stop
		# print result['x']
		output_model = True
		pdict = {}
		comp_dict = fit_model(result['x'],param_names,
				              lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
				              temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
				              fit_type,move_temp,output_model)

		# for k in range(0,len(param_names),1):
		# 	print(' %s = %0.2f' % (param_names[k],result['x'][k]) )


		# return result

	# if 1:sys.exit()

	###### Monte Carlo Simulate for Outflows ###############################################################

	if ((monte_carlo==True) ):
		
		par_best     = result['x']
		fit_type     = 'monte_carlo'
		move_temp    = False
		output_model = False

		# for i,name in enumerate(param_names): print name,par_best[i]

		comp_dict = fit_model(par_best,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
							  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
							  fit_type,move_temp,output_model)
	
		# Compute the total standard deviation at each pixel 
		# To take into account the uncertainty of the model, we compute the median absolute deviation of
		# the residuals from the initial fit, and add it in quadrature to the per-pixel standard deviation
		# of the sdss spectra.
		model_std = mad_std(comp_dict['residuals']['comp'])
	
		sigma = np.sqrt(mad_std(noise)**2 + model_std**2)
		# print('sigma = %0.4f' % (sigma) )

		model = comp_dict['model']['comp']
		# data  = comp_dict['data']['comp']

		# Iteratively box-car smooth until
		# fig = plt.figure(figsize=(14,6))
		# ax1 = fig.add_subplot(1,1,1)
		# ax1.plot(lam_gal,data,color='white',linewidth=0.5)

		# from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
		# model = convolve(data, Box1DKernel(3))
		# ax1.plot(lam_gal,convolve(data, Box1DKernel(3)),linewidth=0.5,label=r'kernel = $1\sigma$')

		# ax1.legend()
		# plt.savefig(run_dir+'smooth_test.pdf',fmt='pdf',dpi=300)

		# if 1: sys.exit()

		mcpars = np.empty((len(par_best), niter)) # stores best-fit parameters of each MC iteration
	
		

		if (na_feii_temp is None) or (br_feii_temp is None):
			na_feii_temp = np.full_like(lam_gal,0)
			br_feii_temp = np.full_like(lam_gal,0)


		print( '\n Performing Monte Carlo resampling to determine if outflows are present...')
		print( '\n       Approximate time for %d iterations: %s \n' % (niter,time_convert(elap_time*niter))  )
		for n in range(0,niter,1):
			print('       Completed %d of %d iterations.' % (n+1,niter) )
			# Generate an array of random normally distributed data using sigma
			rand  = np.random.normal(0.0,sigma,len(lam_gal))
	
			mcgal = model + rand
			fit_type     = 'init'
			move_temp    = False
			output_model = False

			if all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff','oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==True:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = params, \
	         				 		 args=(param_names,lam_gal,mcgal,noise,gal_temp,na_feii_temp,br_feii_temp,
	         				 	   		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
	         				 	   		   fit_type,move_temp,output_model),\
	         				 		 method='SLSQP', bounds = bounds, constraints=cons1, options={'maxiter':10000,'disp': False})

			elif all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff','oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==False:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = params, \
	         				 		 args=(param_names,lam_gal,mcgal,noise,gal_temp,na_feii_temp,br_feii_temp,
	         				 	   		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
	         				 	   		   fit_type,move_temp,output_model),\
	         				 		 method='SLSQP', bounds = bounds, options={'maxiter':10000,'disp': False})
			mcpars[:,n] = resultmc['x']
	
			# For testing: plots every max. likelihood iteration
			if 0:
				output_model = True
				fit_model(resultmc['x'],param_names,lam_gal,mcgal,noise,gal_temp,na_feii_temp,br_feii_temp,
			  			  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
			  			  fit_type,move_temp,output_model)
	
		# create dictionary of param names, means, and standard deviations
		mc_med = np.median(mcpars,axis=1)
		mc_std = mad_std(mcpars,axis=1)
		pdict = {}
		for k in range(0,len(param_names),1):
			pdict[param_names[k]] = {'med':mc_med[k],'std':mc_std[k]}
		
		# for key in pdict:
		# 	print('    %s =  %0.2f  +/-  %0.2f ' % (key,pdict[key]['med'],pdict[key]['std']) )

		if 1:
			output_model = True
			comp_dict = fit_model([pdict[key]['med'] for key in pdict],pdict.keys(),
					              lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
					              temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
					              fit_type,move_temp,output_model)
			plt.savefig(run_dir+'outflow_mcbs.pdf',fmt='pdf',dpi=150)
	
		# Outflow determination and LOSVD for final model: 
		# determine if the object has outflows and if stellar velocity dispersion should be fit

		return pdict, sigma

	elif (monte_carlo==False):
		pdict = {}
		for k in range(0,len(param_names),1):
			pdict[param_names[k]] = {'res':result['x'][k]}
		return pdict


#### Likelihood function #########################################################

# Maximum Likelihood (initial fitting), Prior, and log Probability functions
def lnlike(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		   fit_type,move_temp,output_model):

	# Create model
	model = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
					  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
					  fit_type,move_temp,output_model)
	# Calculate log-likelihood
	l = -0.5*(galaxy-model)**2/(noise)**2
	l = np.sum(l,axis=0)
	return l

##################################################################################

#### Priors ######################################################################

def lnprior(params,param_names,bounds):

	lower_lim = []
	upper_lim = []
	for i in range(0,len(bounds),1):
		lower_lim.append(bounds[i][0])
		upper_lim.append(bounds[i][1])

	# print bounds
	# print lower_lim
	# print upper_lim

	pdict = {}
	for k in range(0,len(param_names),1):
			pdict[param_names[k]] = {'p':params[k]}

	# print pdict

	# if 1: sys.exit()

	if (all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
											 'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==True) and \
	   (all(comp in param_names for comp in ['nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
	   										 'na_Ha_amp','na_Ha_voff',
	   										 'sii6732_core_amp','sii6732_core_voff','sii6718_core_amp',
	   										 'nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==False):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
		    (pdict['oiii5007_core_amp']['p'] >= pdict['oiii5007_outflow_amp']['p']) & \
		    (pdict['oiii5007_outflow_fwhm']['p'] >= pdict['oiii5007_core_fwhm']['p']) & \
		    (pdict['oiii5007_core_voff']['p'] >= pdict['oiii5007_outflow_voff']['p']):
		    return 0.0
		else: return -np.inf
	elif (all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
												 'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==False) and \
		   (all(comp in param_names for comp in ['nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
		   										 'na_Ha_amp','na_Ha_voff',
		   										 'sii6732_core_amp','sii6732_core_voff','sii6718_core_amp',
		   										 'nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==True):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['nii6585_core_amp']['p'] >= pdict['nii6585_outflow_amp']['p']) & \
			(pdict['nii6585_outflow_fwhm']['p'] >= pdict['nii6585_core_fwhm']['p']) & \
			(pdict['nii6585_core_voff']['p'] >= pdict['nii6585_outflow_voff']['p']):
			return 0.0
		else: return -np.inf
	elif (all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
													  'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff',
													  'nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
													  'na_Ha_amp','na_Ha_voff',
													  'sii6732_core_amp','sii6732_core_voff',
													  'sii6718_core_amp',
													  'nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==True):
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
		    (pdict['oiii5007_core_amp']['p'] >= pdict['oiii5007_outflow_amp']['p']) & \
		    (pdict['oiii5007_outflow_fwhm']['p'] >= pdict['oiii5007_core_fwhm']['p']) & \
		    (pdict['oiii5007_core_voff']['p'] >= pdict['oiii5007_outflow_voff']['p']) & \
		    (pdict['nii6585_core_amp']['p'] >= pdict['nii6585_outflow_amp']['p']) & \
			(pdict['nii6585_outflow_fwhm']['p'] >= pdict['nii6585_core_fwhm']['p']) & \
			(pdict['nii6585_core_voff']['p'] >= pdict['nii6585_outflow_voff']['p']):
			return 0.0
		else: return -np.inf
	elif all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
											  'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==False:
		if np.all((params >= lower_lim) & (params <= upper_lim)):
			return 0.0
		else: return -np.inf

##################################################################################

##################################################################################

def lnprob(params,param_names,bounds,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir):
	# lnprob (params,args)

	lp = lnprior(params,param_names,bounds)

	if not np.isfinite(lp):
		return -np.inf
	elif (np.isfinite(lp)==True):
		fit_type     = 'final'
		move_temp    = True
		output_model = False
		return lp + lnlike(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		   					temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		   					fit_type,move_temp,output_model)

####################################################################################




#### Model Function ##############################################################

def fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
			  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
			  fit_type,move_temp,output_model):


	"""
	Constructs galaxy model by convolving templates with a LOSVD given by 
	a specified set of velocity parameters. 
	
	Parameters:
	    pars: parameters of Markov-chain
	    lam_gal: wavelength vector used for continuum model
	    temp_fft: the Fourier-transformed templates
	    npad: 
	    velscale: the velocity scale in km/s/pixel
	    npix: number of output pixels; must be same as galaxy
	    vsyst: dv; the systematic velocity fr
	"""

	# Construct dictionary of parameter names and their respective parameter values
	# param_names  = [param_dict[key]['name'] for key in param_dict ]
	# params       = [param_dict[key]['init'] for key in param_dict ]
	keys = param_names
	values = params
	p = dict(zip(keys, values))
	# if 1: sys.exit()
	# print p
	c = 299792.458 # speed of light
	host_model = np.copy(galaxy)
	comp_dict  = {} 

	############################# Power-law Component ######################################################

	if all(comp in param_names for comp in ['power_amp','power_slope'])==True:
		# print p['power_slope'],p['power_amp']
		# Create a template model for the power-law continuum
		power = simple_power_law(lam_gal,p['power_amp'],p['power_slope']) # ind 2 = alpha, ind 3 = amplitude
		host_model = (host_model) - (power) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['power'] = {'comp':power,'pcolor':'xkcd:orange red','linewidth':1.0}

	########################################################################################################

    ############################# Fe II Component ##########################################################

	if all(comp in param_names for comp in ['na_feii_amp','br_feii_amp'])==True:

		# Create template model for narrow and broad FeII emission 
		na_feii_amp  = p['na_feii_amp']
		br_feii_amp  = p['br_feii_amp']
		
		na_feii_template = na_feii_amp*na_feii_temp
		br_feii_template = br_feii_amp*br_feii_temp
		
		# If including FeII templates, initialize them here
		 
		host_model = (host_model) - (na_feii_template) - (br_feii_template)
		comp_dict['na_feii_template'] = {'comp':na_feii_template,'pcolor':'xkcd:deep red','linewidth':1.0}
		comp_dict['br_feii_template'] = {'comp':br_feii_template,'pcolor':'xkcd:bright orange','linewidth':1.0}

    ########################################################################################################

    ############################# Emission Line Components #################################################    


    # Narrow lines 

	#### Jenna's Lines #################################################################################
	if all(comp in param_names for comp in ['na_oi_6300_amp','na_oi_6300_fwhm','na_oi_6300_voff',
											'na_fevii_5722_amp','na_fevii_5722_fwhm','na_fevii_5722_voff',
											'na_fevii_6085_amp','na_fevii_6085_fwhm','na_fevii_6085_voff'])==True:
		# Narrow [OI]6300
		na_oi6300_core_center = 6302.046 # Angstroms
		na_oi6300_core_amp    = p['na_oi_6300_amp'] # flux units
		na_oi6300_core_fwhm   = np.sqrt(p['na_oi_6300_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oi6300_core_voff   = p['na_oi_6300_voff']  # km/s
		na_oi6300_core   = gaussian(lam_gal,na_oi6300_core_center,na_oi6300_core_amp,na_oi6300_core_fwhm,na_oi6300_core_voff,velscale)
		host_model         = host_model - na_oi6300_core
		comp_dict['na_oi6300_core'] = {'comp':na_oi6300_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [FeVII]5722
		na_fevii5722_core_center = 5722.000 # Angstroms
		na_fevii5722_core_amp    = p['na_fevii_5722_amp'] # flux units
		na_fevii5722_core_fwhm   = np.sqrt(p['na_fevii_5722_fwhm']**2+(2.355*velscale)**2) # km/s
		na_fevii5722_core_voff   = p['na_fevii_5722_voff']  # km/s
		na_fevii5722_core   = gaussian(lam_gal,na_fevii5722_core_center,na_fevii5722_core_amp,na_fevii5722_core_fwhm,na_fevii5722_core_voff,velscale)
		host_model         = host_model - na_fevii5722_core
		comp_dict['na_fevii5722_core'] = {'comp':na_fevii5722_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [FeVII]6085
		na_fevii6085_core_center = 6085.000 # Angstroms
		na_fevii6085_core_amp    = p['na_fevii_6085_amp'] # flux units
		na_fevii6085_core_fwhm   = np.sqrt(p['na_fevii_6085_fwhm']**2+(2.355*velscale)**2) # km/s
		na_fevii6085_core_voff   = p['na_fevii_6085_voff']  # km/s
		na_fevii6085_core   = gaussian(lam_gal,na_fevii6085_core_center,na_fevii6085_core_amp,na_fevii6085_core_fwhm,na_fevii6085_core_voff,velscale)
		host_model         = host_model - na_fevii6085_core
		comp_dict['na_fevii6085_core'] = {'comp':na_fevii6085_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}

	####################################################################################################

    #### [OII]3727,3729 #################################################################################
	if all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_fwhm','na_oii3727_core_voff','na_oii3729_core_amp'])==True:
		# Narrow [OII]3727
		na_oii3727_core_center = 3727.092 # Angstroms
		na_oii3727_core_amp    = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm   = np.sqrt(p['na_oii3727_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oii3727_core_voff   = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core   = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model         = host_model - na_oii3727_core
		comp_dict['na_oii3727_core'] = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729
		na_oii3729_core_center = 3729.875 # Angstroms
		na_oii3729_core_amp    = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm   = na_oii3727_core_fwhm # km/s
		na_oii3729_core_voff   = na_oii3727_core_voff  # km/s
		na_oii3729_core   = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model         = host_model - na_oii3729_core
		comp_dict['na_oii3729_core'] = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_voff','na_oii3729_core_amp'])==True:
		# Narrow [OII]3727
		na_oii3727_core_center = 3727.092 # Angstroms
		na_oii3727_core_amp    = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oii3727_core_voff   = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core   = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model         = host_model - na_oii3727_core
		comp_dict['na_oii3727_core'] = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729
		na_oii3729_core_center = 3729.875 # Angstroms
		na_oii3729_core_amp    = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oii3729_core_voff   = na_oii3727_core_voff  # km/s
		na_oii3729_core        = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model             = host_model - na_oii3729_core
		comp_dict['na_oii3729_core'] = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### [NeIII]3870 #################################################################################
	if all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_fwhm','na_neiii_core_voff'])==True:
		# Narrow H-gamma
		na_neiii_core_center = 3869.810 # Angstroms
		na_neiii_core_amp    = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm   = np.sqrt(p['na_neiii_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_neiii_core_voff   = p['na_neiii_core_voff']  # km/s
		na_neiii_core        = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model           = host_model - na_neiii_core
		comp_dict['na_neiii_core'] = {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_voff'])==True:
		# Narrow H-gamma
		na_neiii_core_center = 3869.810 # Angstroms
		na_neiii_core_amp    = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_neiii_core_voff   = p['na_neiii_core_voff']  # km/s
		na_neiii_core        = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model           = host_model - na_neiii_core
		comp_dict['na_neiii_core'] = {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### H-delta #####################################################################################
	if all(comp in param_names for comp in ['na_Hd_amp','na_Hd_fwhm','na_Hd_voff'])==True:
		# Narrow H-gamma
		na_hd_center = 4102.890 # Angstroms
		na_hd_amp    = p['na_Hd_amp'] # flux units
		na_hd_fwhm   = np.sqrt(p['na_Hd_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hd_voff   = p['na_Hd_voff']  # km/s
		na_Hd_core   = gaussian(lam_gal,na_hd_center,na_hd_amp,na_hd_fwhm,na_hd_voff,velscale)
		host_model   = host_model - na_Hd_core
		comp_dict['na_Hd_core'] = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif all(comp in param_names for comp in ['na_Hd_amp','na_Hd_voff'])==True:
		# Narrow H-gamma
		na_hd_center = 4102.890 # Angstroms
		na_hd_amp    = p['na_Hd_amp'] # flux units
		na_hd_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hd_voff   = p['na_Hd_voff']  # km/s
		na_Hd_core   = gaussian(lam_gal,na_hd_center,na_hd_amp,na_hd_fwhm,na_hd_voff,velscale)
		host_model   = host_model - na_Hd_core
		comp_dict['na_Hd_core'] = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### H-gamma/[OIII]4363 ##########################################################################
	if all(comp in param_names for comp in ['na_Hg_amp','na_Hg_fwhm','na_Hg_voff','oiii4363_core_amp','oiii4363_core_fwhm','oiii4363_core_voff'])==True:
		# Narrow H-gamma
		na_hg_center = 4341.680 # Angstroms
		na_hg_amp    = p['na_Hg_amp'] # flux units
		na_hg_fwhm   = np.sqrt(p['na_Hg_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hg_voff   = p['na_Hg_voff']  # km/s
		na_Hg_core   = gaussian(lam_gal,na_hg_center,na_hg_amp,na_hg_fwhm,na_hg_voff,velscale)
		host_model   = host_model - na_Hg_core
		comp_dict['na_Hg_core'] = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_center = 4364.436 # Angstroms
		na_oiii4363_amp    =  p['oiii4363_core_amp'] # flux units
		na_oiii4363_fwhm   =  np.sqrt(p['oiii4363_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii4363_voff   =  p['oiii4363_core_voff'] # km/s
		na_oiii4363_core   = gaussian(lam_gal,na_oiii4363_center,na_oiii4363_amp,na_oiii4363_fwhm,na_oiii4363_voff,velscale)
		host_model         = host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif all(comp in param_names for comp in ['na_Hg_amp','na_Hg_voff','oiii4363_core_amp','oiii4363_core_voff'])==True:
		# Narrow H-gamma
		na_hg_center = 4341.680 # Angstroms
		na_hg_amp    = p['na_Hg_amp'] # flux units
		na_hg_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hg_voff   = p['na_Hg_voff']  # km/s
		na_Hg_core   = gaussian(lam_gal,na_hg_center,na_hg_amp,na_hg_fwhm,na_hg_voff,velscale)
		host_model   = host_model - na_Hg_core
		comp_dict['na_Hg_core'] = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_center = 4364.436 # Angstroms
		na_oiii4363_amp    =  p['oiii4363_core_amp'] # flux units
		na_oiii4363_fwhm   =  np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii4363_voff   =  p['oiii4363_core_voff'] # km/s
		na_oiii4363_core   = gaussian(lam_gal,na_oiii4363_center,na_oiii4363_amp,na_oiii4363_fwhm,na_oiii4363_voff,velscale)
		host_model         = host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### H-beta/[OIII] #########################################################################################
	if all(comp in param_names for comp in ['oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',
											'na_Hb_amp','na_Hb_voff'])==True:
		# Narrow [OIII]5007 Core
		na_oiii5007_center = 5008.240 # Angstroms
		na_oiii5007_amp    = p['oiii5007_core_amp'] # flux units
		na_oiii5007_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii5007_voff   = p['oiii5007_core_voff']  # km/s
		na_oiii5007_core   = gaussian(lam_gal,na_oiii5007_center,na_oiii5007_amp,na_oiii5007_fwhm,na_oiii5007_voff,velscale)
		host_model         = host_model - na_oiii5007_core
		comp_dict['na_oiii5007_core'] = {'comp':na_oiii5007_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    	# Narrow [OIII]4959 Core
		na_oiii4959_center = 4960.295 # Angstroms
		na_oiii4959_amp    = (1.0/3.0)*na_oiii5007_amp # flux units
		na_oiii4959_fwhm   = na_oiii5007_fwhm # km/s
		na_oiii4959_voff   = na_oiii5007_voff  # km/s
		na_oiii4959_core   = gaussian(lam_gal,na_oiii4959_center,na_oiii4959_amp,na_oiii4959_fwhm,na_oiii4959_voff,velscale)
		host_model         = host_model - na_oiii4959_core
		comp_dict['na_oiii4959_core'] = {'comp':na_oiii4959_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow H-beta
		na_hb_center = 4862.680 # Angstroms
		na_hb_amp    = p['na_Hb_amp'] # flux units
		na_hb_fwhm   = np.sqrt(na_oiii5007_fwhm**2+(2.355*velscale)**2) # km/s
		na_hb_voff   = p['na_Hb_voff']  # km/s
		na_Hb_core   = gaussian(lam_gal,na_hb_center,na_hb_amp,na_hb_fwhm,na_hb_voff,velscale)
		host_model   = host_model - na_Hb_core
		comp_dict['na_Hb_core'] = {'comp':na_Hb_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	#### H-alpha/[NII]/[SII] ####################################################################################
	if all(comp in param_names for comp in ['nii6585_core_amp','nii6585_core_fwhm','nii6585_core_voff',
											'na_Ha_amp','na_Ha_voff',
											'sii6732_core_amp','sii6732_core_voff',
											'sii6718_core_amp'])==True:
		# Narrow [NII]6585 Core
		na_nii6585_center = 6585.270 # Angstroms
		na_nii6585_amp    = p['nii6585_core_amp'] # flux units
		na_nii6585_fwhm   = np.sqrt(p['nii6585_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_nii6585_voff   = p['nii6585_core_voff']  # km/s
		na_nii6585_core   = gaussian(lam_gal,na_nii6585_center,na_nii6585_amp,na_nii6585_fwhm,na_nii6585_voff,velscale)
		host_model        = host_model - na_nii6585_core
		comp_dict['na_nii6585_core'] = {'comp':na_nii6585_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    	# Narrow [NII]6549 Core
		na_nii6549_center = 6549.860 # Angstroms
		na_nii6549_amp    = (1.0/2.93)*na_nii6585_amp # flux units
		na_nii6549_fwhm   = na_nii6585_fwhm # km/s
		na_nii6549_voff   = na_nii6585_voff  # km/s
		na_nii6549_core   = gaussian(lam_gal,na_nii6549_center,na_nii6549_amp,na_nii6549_fwhm,na_nii6549_voff,velscale)
		host_model        = host_model - na_nii6549_core
		comp_dict['na_nii6549_core'] = {'comp':na_nii6549_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow H-alpha
		na_ha_center = 6564.610 # Angstroms
		na_ha_amp    = p['na_Ha_amp'] # flux units
		na_ha_fwhm   = na_nii6585_fwhm # km/s
		na_ha_voff   = p['na_Ha_voff']  # km/s
		na_Ha_core   = gaussian(lam_gal,na_ha_center,na_ha_amp,na_ha_fwhm,na_ha_voff,velscale)
		host_model   = host_model - na_Ha_core
		comp_dict['na_Ha_core'] = {'comp':na_Ha_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6732
		na_sii6732_center = 6732.670 # Angstroms
		na_sii6732_amp    = p['sii6732_core_amp'] # flux units
		na_sii6732_fwhm   = na_nii6585_fwhm #np.sqrt(p['sii6732_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_sii6732_voff   = p['sii6732_core_voff']  # km/s
		na_sii6732_core   = gaussian(lam_gal,na_sii6732_center,na_sii6732_amp,na_sii6732_fwhm,na_sii6732_voff,velscale)
		host_model        = host_model - na_sii6732_core
		comp_dict['na_sii6732_core'] = {'comp':na_sii6732_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6718
		na_sii6718_center = 6718.290 # Angstroms
		na_sii6718_amp    = p['sii6718_core_amp'] # flux units
		na_sii6718_fwhm   = na_nii6585_fwhm #na_sii6732_fwhm # km/s
		na_sii6718_voff   = na_sii6732_voff  # km/s
		na_sii6718_core   = gaussian(lam_gal,na_sii6718_center,na_sii6718_amp,na_sii6718_fwhm,na_sii6718_voff,velscale)
		host_model        = host_model - na_sii6718_core
		comp_dict['na_sii6718_core'] = {'comp':na_sii6718_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif all(comp in param_names for comp in ['nii6585_core_amp','nii6585_core_voff',
											  'na_Ha_amp','na_Ha_voff',
											  'sii6732_core_amp','sii6732_core_voff',
											  'sii6718_core_amp'])==True:
		# Narrow [NII]6585 Core
		na_nii6585_center = 6585.270 # Angstroms
		na_nii6585_amp    = p['nii6585_core_amp'] # flux units
		na_nii6585_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_nii6585_voff   = p['nii6585_core_voff']  # km/s
		na_nii6585_core   = gaussian(lam_gal,na_nii6585_center,na_nii6585_amp,na_nii6585_fwhm,na_nii6585_voff,velscale)
		host_model        = host_model - na_nii6585_core
		comp_dict['na_nii6585_core'] = {'comp':na_nii6585_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    	# Narrow [NII]6549 Core
		na_nii6549_center = 6549.860 # Angstroms
		na_nii6549_amp    = (1.0/2.93)*na_nii6585_amp # flux units
		na_nii6549_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_nii6549_voff   = na_nii6585_voff  # km/s
		na_nii6549_core   = gaussian(lam_gal,na_nii6549_center,na_nii6549_amp,na_nii6549_fwhm,na_nii6549_voff,velscale)
		host_model        = host_model - na_nii6549_core
		comp_dict['na_nii6549_core'] = {'comp':na_nii6549_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow H-alpha
		na_ha_center = 6564.610 # Angstroms
		na_ha_amp    = p['na_Ha_amp'] # flux units
		na_ha_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_ha_voff   = p['na_Ha_voff']  # km/s
		na_Ha_core   = gaussian(lam_gal,na_ha_center,na_ha_amp,na_ha_fwhm,na_ha_voff,velscale)
		host_model   = host_model - na_Ha_core
		comp_dict['na_Ha_core'] = {'comp':na_Ha_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6732
		na_sii6732_center = 6732.670 # Angstroms
		na_sii6732_amp    = p['sii6732_core_amp'] # flux units
		na_sii6732_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_sii6732_voff   = p['sii6732_core_voff']  # km/s
		na_sii6732_core   = gaussian(lam_gal,na_sii6732_center,na_sii6732_amp,na_sii6732_fwhm,na_sii6732_voff,velscale)
		host_model        = host_model - na_sii6732_core
		comp_dict['na_sii6732_core'] = {'comp':na_sii6732_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6718
		na_sii6718_center = 6718.290 # Angstroms
		na_sii6718_amp    = p['sii6718_core_amp'] # flux units
		na_sii6718_fwhm   = np.sqrt(p['oiii5007_core_fwhm']**2+(2.355*velscale)**2)  # km/s
		na_sii6718_voff   = na_sii6732_voff  # km/s
		na_sii6718_core   = gaussian(lam_gal,na_sii6718_center,na_sii6718_amp,na_sii6718_fwhm,na_sii6718_voff,velscale)
		host_model        = host_model - na_sii6718_core
		comp_dict['na_sii6718_core'] = {'comp':na_sii6718_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	########################################################################################################

	# Outflow Components
    #### Hb/[OIII] outflows ################################################################################
	if (all(comp in param_names for comp in ['oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff'])==True):
		# Broad [OIII]5007 Outflow;
		br_oiii5007_center  = 5008.240 # Angstroms
		br_oiii5007_amp     = p['oiii5007_outflow_amp'] # flux units
		br_oiii5007_fwhm    = np.sqrt(p['oiii5007_outflow_fwhm']**2+(2.355*velscale)**2) # km/s
		br_oiii5007_voff    = p['oiii5007_outflow_voff']  # km/s
		br_oiii5007_outflow = gaussian(lam_gal,br_oiii5007_center,br_oiii5007_amp,br_oiii5007_fwhm,br_oiii5007_voff,velscale)
		host_model          = host_model - br_oiii5007_outflow
		comp_dict['br_oiii5007_outflow'] = {'comp':br_oiii5007_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
    	# Broad [OIII]4959 Outflow; 
		br_oiii4959_center  = 4960.295 # Angstroms
		br_oiii4959_amp     = br_oiii5007_amp*na_oiii4959_amp/na_oiii5007_amp # flux units
		br_oiii4959_fwhm    = br_oiii5007_fwhm # km/s
		br_oiii4959_voff    = br_oiii5007_voff  # km/s
		if (br_oiii4959_amp!=br_oiii4959_amp/1.0) or (br_oiii4959_amp==np.inf): br_oiii4959_amp=0.0
		br_oiii4959_outflow = gaussian(lam_gal,br_oiii4959_center,br_oiii4959_amp,br_oiii4959_fwhm,br_oiii4959_voff,velscale)
		host_model          = host_model - br_oiii4959_outflow
		comp_dict['br_oiii4959_outflow'] = {'comp':br_oiii4959_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# Broad H-beta Outflow; only a model, no free parameters, tied to [OIII]5007
		br_hb_outflow_amp  =  br_oiii5007_amp*na_hb_amp/na_oiii5007_amp
		br_hb_outflow_fwhm = np.sqrt(br_oiii5007_fwhm**2+(2.355*velscale)**2) # km/s
		br_hb_outflow_voff = na_hb_voff+br_oiii5007_voff
		if (br_hb_outflow_amp!=br_hb_outflow_amp/1.0) or (br_hb_outflow_amp==np.inf): br_hb_outflow_amp=0.0
		br_Hb_outflow      = gaussian(lam_gal,na_hb_center,br_hb_outflow_amp,br_hb_outflow_fwhm,br_hb_outflow_voff,velscale)
		host_model         = host_model - br_Hb_outflow
		comp_dict['br_Hb_outflow'] = {'comp':br_Hb_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	#### Ha/[NII]/[SII] outflows ###########################################################################
	if (all(comp in param_names for comp in ['nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==True):
		# Outflows in H-alpha/[NII] are poorly constrained due to the presence of a broad line, therefore
		# we tie all outflows in this region together with the [SII] outflow (as we do similarly for the Hb/[OIII] region)
		# Broad [NII]6585 Outflow;
		br_nii6585_center  = 6585.270 # Angstroms
		br_nii6585_amp     = p['nii6585_outflow_amp'] # flux units
		br_nii6585_fwhm    = np.sqrt(p['nii6585_outflow_fwhm']**2+(2.355*velscale)**2) # km/s
		br_nii6585_voff    = p['nii6585_outflow_voff']  # km/s
		br_nii6585_outflow = gaussian(lam_gal,br_nii6585_center,br_nii6585_amp,br_nii6585_fwhm,br_nii6585_voff,velscale)
		host_model         = host_model - br_nii6585_outflow
		comp_dict['br_nii6585_outflow'] = {'comp':br_nii6585_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad [NII]6549 Outflow; 
		br_nii6549_center  = 6549.860 # Angstroms
		br_nii6549_amp     = br_nii6585_amp*na_nii6549_amp/na_nii6585_amp # flux units
		br_nii6549_fwhm    = br_nii6585_fwhm # km/s
		br_nii6549_voff    = br_nii6585_voff  # km/s
		if (br_nii6549_amp!=br_nii6549_amp/1.0) or (br_nii6549_amp==np.inf): br_nii6549_amp=0.0
		br_nii6549_outflow = gaussian(lam_gal,br_nii6549_center,br_nii6549_amp,br_nii6549_fwhm,br_nii6549_voff,velscale)
		host_model         = host_model - br_nii6549_outflow
		comp_dict['br_nii6549_outflow'] = {'comp':br_nii6549_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad H-alpha Outflow; 
		br_ha_center  = 6564.610 # Angstroms
		br_ha_amp     = br_nii6585_amp*na_ha_amp/na_nii6585_amp # flux units
		br_ha_fwhm    = br_nii6585_fwhm # km/s
		br_ha_voff    = br_nii6585_voff  # km/s
		if (br_ha_amp!=br_ha_amp/1.0) or (br_ha_amp==np.inf): br_ha_amp=0.0
		br_Ha_outflow = gaussian(lam_gal,br_ha_center,br_ha_amp,br_ha_fwhm,br_ha_voff,velscale)
		host_model    = host_model - br_Ha_outflow
		comp_dict['br_Ha_outflow'] = {'comp':br_Ha_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad [SII]6732 Outflow; 
		br_sii6732_center  = 6732.670 # Angstroms
		br_sii6732_amp     = br_nii6585_amp*na_sii6732_amp/na_nii6585_amp # flux units
		br_sii6732_fwhm    = br_nii6585_fwhm # km/s
		br_sii6732_voff    = br_nii6585_voff  # km/s
		if (br_sii6732_amp!=br_sii6732_amp/1.0) or (br_sii6732_amp==np.inf): br_sii6732_amp=0.0
		br_sii6732_outflow = gaussian(lam_gal,br_sii6732_center,br_sii6732_amp,br_sii6732_fwhm,br_sii6732_voff,velscale)
		host_model         = host_model - br_sii6732_outflow
		comp_dict['br_sii6732_outflow'] = {'comp':br_sii6732_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	elif ((all(comp in param_names for comp in ['nii6585_outflow_amp','nii6585_outflow_fwhm','nii6585_outflow_voff'])==False) & 
		  (all(comp in param_names for comp in ['oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff',
		  										'nii6585_core_amp','nii6585_core_voff',
											  	'na_Ha_amp','na_Ha_voff',
											  	'sii6732_core_amp','sii6732_core_voff',
											  	'sii6718_core_amp'])==True)):
		# Outflows in H-alpha/[NII] are poorly constrained due to the presence of a broad line, therefore
		# we tie all outflows in this region together with the [SII] outflow (as we do similarly for the Hb/[OIII] region)
		# Broad [NII]6585 Outflow;
		br_nii6585_center  = 6585.270 # Angstroms
		br_nii6585_amp     = p['nii6585_outflow_amp'] # flux units
		br_nii6585_fwhm    = np.sqrt(p['oiii5007_outflow_fwhm']**2+(2.355*velscale)**2) # km/s
		br_nii6585_voff    = p['nii6585_outflow_voff']  # km/s
		br_nii6585_outflow = gaussian(lam_gal,br_nii6585_center,br_nii6585_amp,br_nii6585_fwhm,br_nii6585_voff,velscale)
		host_model         = host_model - br_nii6585_outflow
		comp_dict['br_nii6585_outflow'] = {'comp':br_nii6585_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad [NII]6549 Outflow; 
		br_nii6549_center  = 6549.860 # Angstroms
		br_nii6549_amp     = br_nii6585_amp*na_nii6549_amp/na_nii6585_amp # flux units
		br_nii6549_fwhm    = br_nii6585_fwhm # km/s
		br_nii6549_voff    = br_nii6585_voff  # km/s
		if (br_nii6549_amp!=br_nii6549_amp/1.0) or (br_nii6549_amp==np.inf): br_nii6549_amp=0.0
		br_nii6549_outflow = gaussian(lam_gal,br_nii6549_center,br_nii6549_amp,br_nii6549_fwhm,br_nii6549_voff,velscale)
		host_model         = host_model - br_nii6549_outflow
		comp_dict['br_nii6549_outflow'] = {'comp':br_nii6549_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad H-alpha Outflow; 
		br_ha_center  = 6564.610 # Angstroms
		br_ha_amp     = br_nii6585_amp*na_ha_amp/na_nii6585_amp # flux units
		br_ha_fwhm    = br_nii6585_fwhm # km/s
		br_ha_voff    = br_nii6585_voff  # km/s
		if (br_ha_amp!=br_ha_amp/1.0) or (br_ha_amp==np.inf): br_ha_amp=0.0
		br_Ha_outflow = gaussian(lam_gal,br_ha_center,br_ha_amp,br_ha_fwhm,br_ha_voff,velscale)
		host_model    = host_model - br_Ha_outflow
		comp_dict['br_Ha_outflow'] = {'comp':br_Ha_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad [SII]6732 Outflow; 
		br_sii6732_center  = 6732.670 # Angstroms
		br_sii6732_amp     = br_nii6585_amp*na_sii6732_amp/na_nii6585_amp # flux units
		br_sii6732_fwhm    = br_nii6585_fwhm # km/s
		br_sii6732_voff    = br_nii6585_voff  # km/s
		if (br_sii6732_amp!=br_sii6732_amp/1.0) or (br_sii6732_amp==np.inf): br_sii6732_amp=0.0
		br_sii6732_outflow = gaussian(lam_gal,br_sii6732_center,br_sii6732_amp,br_sii6732_fwhm,br_sii6732_voff,velscale)
		host_model         = host_model - br_sii6732_outflow
		comp_dict['br_sii6732_outflow'] = {'comp':br_sii6732_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}

		# Broad [SII]6718 Outflow; 
		br_sii6718_center  = 6718.290 # Angstroms
		br_sii6718_amp     = br_nii6585_amp*na_sii6718_amp/na_nii6585_amp # flux units
		br_sii6718_fwhm    = br_nii6585_fwhm # km/s
		br_sii6718_voff    = br_nii6585_voff  # km/s
		if (br_sii6718_amp!=br_sii6718_amp/1.0) or (br_sii6718_amp==np.inf): br_sii6718_amp=0.0
		br_sii6718_outflow = gaussian(lam_gal,br_sii6718_center,br_sii6718_amp,br_sii6718_fwhm,br_sii6718_voff,velscale)
		host_model         = host_model - br_sii6718_outflow
		comp_dict['br_sii6718_outflow'] = {'comp':br_sii6718_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}


	########################################################################################################

	# Broad Lines
	# #### Br. H-delta #######################################################################################
	# Commented out because Br. H-delta is usually not distinguishable from the noise.
	# if all(comp in param_names for comp in ['br_Hd_amp','br_Hd_fwhm','br_Hd_voff'])==True:
	# 	na_hd_center = 4102.89 # Angstroms
	# 	br_hd_amp    = p['br_Hd_amp'] # flux units
	# 	br_hd_fwhm   = np.sqrt(p['br_Hd_fwhm']**2+(2.355*velscale)**2) # km/s
	# 	br_hd_voff   = p['br_Hd_voff']  # km/s
	# 	br_Hd        = gaussian(lam_gal,na_hd_center,br_hd_amp,br_hd_fwhm,br_hd_voff,velscale)
	# 	host_model         = host_model - br_Hd
	# 	comp_dict['br_Hd'] = {'comp':br_Hd,'pcolor':'xkcd:blue','linewidth':0.5}
	#### Br. H-gamma #######################################################################################
	if all(comp in param_names for comp in ['br_Hg_amp','br_Hg_fwhm','br_Hg_voff'])==True:
		na_hg_center = 4341.680 # Angstroms
		br_hg_amp    = p['br_Hg_amp'] # flux units
		br_hg_fwhm   = np.sqrt(p['br_Hg_fwhm']**2+(2.355*velscale)**2) # km/s
		br_hg_voff   = p['br_Hg_voff']  # km/s
		br_Hg        = gaussian(lam_gal,na_hg_center,br_hg_amp,br_hg_fwhm,br_hg_voff,velscale)
		host_model         = host_model - br_Hg
		comp_dict['br_Hg'] = {'comp':br_Hg,'pcolor':'xkcd:blue','linewidth':1.0}
	#### Br. H-beta ########################################################################################
	if all(comp in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])==True:
		na_hb_center = 4862.68 # Angstroms
		br_hb_amp    = p['br_Hb_amp'] # flux units
		br_hb_fwhm   = np.sqrt(p['br_Hb_fwhm']**2+(2.355*velscale)**2) # km/s
		br_hb_voff   = p['br_Hb_voff']  # km/s
		br_Hb        = gaussian(lam_gal,na_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)
		host_model         = host_model - br_Hb
		comp_dict['br_Hb'] = {'comp':br_Hb,'pcolor':'xkcd:blue','linewidth':1.0}
 
	#### Br. H-alpha #######################################################################################
	if all(comp in param_names for comp in ['br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True:
		na_ha_center = 6564.610 # Angstroms
		br_ha_amp    = p['br_Ha_amp'] # flux units
		br_ha_fwhm   = np.sqrt(p['br_Ha_fwhm']**2+(2.355*velscale)**2) # km/s
		br_ha_voff   = p['br_Ha_voff']  # km/s
		br_Ha        = gaussian(lam_gal,na_ha_center,br_ha_amp,br_ha_fwhm,br_ha_voff,velscale)
		host_model         = host_model - br_Ha
		comp_dict['br_Ha'] = {'comp':br_Ha,'pcolor':'xkcd:blue','linewidth':1.0}

	########################################################################################################

	########################################################################################################

	############################# Host-galaxy Component ######################################################

	if all(comp in param_names for comp in ['gal_temp_amp'])==True:
		gal_temp = p['gal_temp_amp']*(gal_temp)
		host_model = (host_model) - (gal_temp) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['gal_temp'] = {'comp':gal_temp,'pcolor':'xkcd:lime green','linewidth':1.0}

	########################################################################################################   

	############################# LOSVD Component ####################################################

	if all(comp in param_names for comp in ['stel_vel','stel_disp'])==True:
		# Convolve the templates with a LOSVD
		losvd_params = [p['stel_vel'],p['stel_disp']] # ind 0 = velocity*, ind 1 = sigma*
		conv_temp    = convolve_gauss_hermite(temp_fft,npad,float(velscale),\
					   losvd_params,npix,velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
		
		# Fitted weights of all templates using Non-negative Least Squares (NNLS)

		weights     = nnls(conv_temp,host_model)
		# weights		= fnnls(conv_temp,host_model)
		host_galaxy = (np.sum(weights*conv_temp,axis=1)) 
		comp_dict['host_galaxy'] = {'comp':host_galaxy,'pcolor':'xkcd:lime green','linewidth':1.0}
		if move_temp==True:
			# Templates that are used are placed in templates folder
			def path_leaf(path):
				head, tail = ntpath.split(path)
				return tail or ntpath.basename(head)

			a =  np.where(weights>0)[0]
			for i in a:
				s = temp_list[i]
				temp_name = path_leaf(temp_list[i])
				if os.path.exists(run_dir+'templates/'+temp_name)==False:
					# Check if template file was already copied to template folder
					# If not, copy it to templates folder for each object
					# print(' Moving template %s' % temp_name)
					shutil.copyfile(temp_list[i],run_dir+'templates/'+temp_name)


	# if 1: sys.exit()
    ########################################################################################################
    
	# The final model
	gmodel = np.sum((d['comp'] for d in comp_dict.values() if d),axis=0)
	
	########################## Measure Emission Line Fluxes #################################################

	# comp_fluxes = [] # list of all computed fluxes 
	em_line_fluxes = {}
	if (fit_type=='final'):
		for key in comp_dict:
			# compute the integrated flux 
			flux = simps(comp_dict[key]['comp'],lam_gal)
			# add key/value pair to dictionary
			em_line_fluxes[key+'_flux'] = [flux]
		df_fluxes = pd.DataFrame(data=em_line_fluxes,columns=em_line_fluxes.keys())
		# Write to csv
		# Create a folder in the working directory to save plots and data to
		if os.path.exists(run_dir+'em_line_fluxes.csv')==False:
			# If file has not been created, create it 
			# print(' File does not exist!')
			df_fluxes.to_csv(run_dir+'em_line_fluxes.csv',sep=',')
		elif os.path.exists(run_dir+'em_line_fluxes.csv')==True:
			# If file has been created, append df_fluxes to existing file
			# print( ' File exists!')
			with open(run_dir+'em_line_fluxes.csv', 'a') as f:
				df_fluxes.to_csv(f, header=False)
				f.close()

	##################################################################################

	# Add last components to comp_dict for plotting purposes 
	# Add galaxy, sigma, model, and residuals to comp_dict
	comp_dict['data']      = {'comp':galaxy      ,'pcolor':'xkcd:white','linewidth':0.5}
	comp_dict['wave']      = {'comp':lam_gal 	 ,'pcolor':'xkcd:black','linewidth':0.5}
	comp_dict['noise']     = {'comp':noise       ,'pcolor':'xkcd:cyan','linewidth':0.5}
	comp_dict['model']     = {'comp':gmodel      ,'pcolor':'xkcd:red','linewidth':1.0}
	comp_dict['residuals'] = {'comp':galaxy-gmodel,'pcolor':'xkcd:white','linewidth':0.5}

	##################################################################################
	if (output_model==True) and (fit_type=='init'):
		fig = plt.figure(figsize=(14,6))
		ax1 = fig.add_subplot(1,1,1)
		for key in comp_dict:
			if ((key != 'wave') and (key != 'noise')):
				ax1.plot(lam_gal,comp_dict[key]['comp'],color=comp_dict[key]['pcolor'],linewidth=comp_dict[key]['linewidth'])
		ax1.axhline(0.0,color='white',linestyle='--')
		ax1.set_ylabel(r'$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
		ax1.set_xlabel(r'$\lambda_{\mathrm{rest}}$ ($\mathrm{\AA}$)')
		ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
		plt.tight_layout()
		plt.savefig(run_dir+'max_like_outflows.pdf',fmt='pdf',dpi=150)
		# if 1: sys.exit()
	
	##################################################################################

	if (fit_type=='init') and (output_model==False): # For max. likelihood fitting
		return gmodel
	elif (fit_type=='monte_carlo'):
		return comp_dict
	elif (fit_type=='final') and (output_model==False): # don't output all models
		return gmodel
	elif (fit_type=='final') and (output_model==True): # output all models
		return comp_dict #,weights

########################################################################################################


#### Host-Galaxy Template##############################################################################

# This function is used if we use the Maraston et al. 2009 SDSS composite templates but the absorption
# features are not deep enough to describe NLS1s.
def galaxy_template(lam,age=5):
	if ((age<1) or (age>15)) and (age is not None):
		raise ValueError('\n You must choose an age between (1 Gyr <= age <= 15 Gyr)! \n')
	elif (age is None) and (np.min(lam) >= 3500.) and (np.max(lam) <= 7000.):
		print('\n Using NLS1 template...\n')
		# Use pre-fit NLS1 template (J000338 host-galaxy template)
		df = pd.read_csv('badass_data_files/nls1_template.csv',skiprows=1,sep=',',names=['wave','flux'])

		# if 1: sys.exit()
		wave =  np.array(df['wave'])
		flux =  np.array(df['flux'])
		# Interpolate the template 
		gal_interp = interp1d(wave,flux,kind='cubic',bounds_error=False,fill_value=(flux[0],flux[-1]))
		gal_temp = gal_interp(lam)
		# Normalize by median
		gal_temp = gal_temp/np.median(gal_temp)
		return gal_temp
	elif (age is None) and ((np.min(lam) < 3500.) or (np.max(lam) > 7000.)):
		raise ValueError(' \n Cannot use NLS1 template for wavelength range <3500 A or >7000 A!')
	elif ((age>=1) and (age<=15)):
		print('\n Using Maraston et al. 2009 template... \n ')
		# We use the SDSS template from Maraston et al. 2009 
		# The template is used as a placeholder for the stellar continuum
		# in the initial parameter fit.
		# Open the file and get the appropriate age galaxy
		df = pd.read_csv('badass_data_files/M09_composite_bestfitLRG.csv',skiprows=5,sep=',',names=['t_Gyr','AA','flam'])
		wave =  np.array(df.loc[(df['t_Gyr']==age),'AA'])
		flux =  np.array(df.loc[(df['t_Gyr']==age),'flam'])
		# Interpolate the template 
		gal_interp = interp1d(wave,flux,kind='cubic',bounds_error=False,fill_value=(0,0))
		gal_temp = gal_interp(lam)
		# Normalize by median
		gal_temp = gal_temp/np.median(gal_temp)
		return gal_temp

##################################################################################


#### FeII Templates ##############################################################

def initialize_feii(lam_gal,velscale,fit_reg):
	# Read in template data
	na_feii_table = pd.read_csv('badass_data_files/na_feii_template.csv')
	br_feii_table = pd.read_csv('badass_data_files/br_feii_template.csv')

	# Construct the FeII templates here (do not reconstruct them for every iteration; only amplitude is changing)
	# Initialize the templates with a given guess for the amplitude; we take 10% of the median flux
	feii_voff = 0.0 # velocity offset
	na_feii_amp = 1.0
	na_feii_rel_int  = np.array(na_feii_table['na_relative_intensity'])
	na_feii_center = np.array(na_feii_table['na_wavelength'])
	na_feii_fwhm = 500.0 # km/s; keep fixed
	br_feii_amp = 1.0
	br_feii_rel_int  = np.array(br_feii_table['br_relative_intensity'])
	br_feii_center = np.array(br_feii_table['br_wavelength'])
	br_feii_fwhm = 3000.0 # km/s; keep fixed    
	na_feii_template = feii_template(lam_gal,na_feii_rel_int,na_feii_center,na_feii_amp,na_feii_fwhm,feii_voff,velscale,fit_reg)
	br_feii_template = feii_template(lam_gal,br_feii_rel_int,br_feii_center,br_feii_amp,br_feii_fwhm,feii_voff,velscale,fit_reg)
	
	return na_feii_template, br_feii_template

def feii_template(lam,rel_int,center,amp,sigma,voff,velscale,fit_reg):

	"""
	Produces a gaussian vector the length of
	x with the specified parameters.
	
	Parameters
	----------
	x : array_like
	    the wavelength vector
	A : float
	    the amplitude of the gaussian.
	center: float
	    the mean or center wavelength of the gaussian.
	sigma: float
	    the standard deviation of the gaussian.
	
	Returns
	-------
	g: array
	    a one-dimensional gaussian as a function of x.
	"""
	# Create an empty array in which to store the lines before summing them
	template = []
	for i in range(0,len(rel_int),1):
		if ((((fit_reg[0]+25.) < center[i] < (fit_reg[1]-25.))==True)):
			line = amp*(gaussian(lam,center[i],rel_int[i],sigma,voff,velscale))
			template.append(line)
		else:
			continue
	template = np.sum(template,axis=0)
	
	return template

##################################################################################


#### Power-Law Template ##########################################################

def simple_power_law(x,amp,alpha):
	"""
	Simple power-low function to model
	the AGNs continuum.

	Parameters
	----------
	x     : array_like
		    wavelength vector (angstroms)
	amp   : float 
		    continuum amplitude (flux density units)
	alpha : float
			power-law slope
	xb    : float
		    location of kink in the power-law (angstroms)

	Returns
	----------
	C     : array
		    AGN continuum model the same length as x
	"""
	xb = np.max(x)-((np.max(x)-np.min(x))/2.0) # take to be half of the wavelength range

	C = amp*(x/xb)**alpha # un-normalized
	return C

##################################################################################


##################################################################################

def gaussian(x,center,amp,fwhm,voff,velscale):
    """
    Produces a gaussian vector the length of
    x with the specified parameters.
    
    Parameters
    ----------
    x        : array_like
        	   the wavelength vector in angstroms.
    center   : float
        	   the mean or center wavelength of the gaussian in angstroms.
    sigma    : float
        	   the standard deviation of the gaussian in km/s.
    amp      : float
        	   the amplitude of the gaussian in flux units.
    voff     : the velocity offset (in km/s) from the rest-frame 
          	   line-center (taken from SDSS rest-frame emission
          	   line wavelengths)
    velscale : velocity scale; km/s/pixel

    Returns
    -------
    g        : array_like
        	   a one-dimensional gaussian as a function of x,
        	   where x is measured in PIXELS.
    """
    x_pix = range(len(x))
    wave_interp = interp1d(x,x_pix,kind='cubic',bounds_error=False,fill_value=(0,0))
    center_pix = wave_interp(center) # pixel value corresponding to line center
    sigma = fwhm/2.3548
    sigma_pix = sigma/velscale
    voff_pix = voff/velscale
    center_pix = center_pix + voff_pix
    
    # print center_pix,sigma_pix,voff_pix

    g = amp*np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2)
    
    # Make sure edges of gaussian are zero 
    if (g[0]>1.0e-6) or g[-1]>1.0e-6:
        g = np.zeros(len(g))
    
    return g

##################################################################################


##################################################################################


# pPXF Routines (from Cappellari 2017)


# NAME:
#   GAUSSIAN_FILTER1D
#
# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#       to be used with variable sigma per pixel. MC, Oxford, 10 October 2015

def gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating a template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum

##################################################################################

def log_rebin(lamRange, spec, oversample=1, velscale=None, flux=False):
    """
    Logarithmically rebin a spectrum, while rigorously conserving the flux.
    Basically the photons in the spectrum are simply redistributed according
    to a new grid of pixels, with non-uniform size in the spectral direction.
    
    When the flux keyword is set, this program performs an exact integration 
    of the original spectrum, assumed to be a step function within the 
    linearly-spaced pixels, onto the new logarithmically-spaced pixels. 
    The output was tested to agree with the analytic solution.

    :param lamRange: two elements vector containing the central wavelength
        of the first and last pixels in the spectrum, which is assumed
        to have constant wavelength scale! E.g. from the values in the
        standard FITS keywords: LAMRANGE = CRVAL1 + [0, CDELT1*(NAXIS1 - 1)].
        It must be LAMRANGE[0] < LAMRANGE[1].
    :param spec: input spectrum.
    :param oversample: can be used, not to loose spectral resolution,
        especally for extended wavelength ranges and to avoid aliasing.
        Default: OVERSAMPLE=1 ==> Same number of output pixels as input.
    :param velscale: velocity scale in km/s per pixels. If this variable is
        not defined, then it will contain in output the velocity scale.
        If this variable is defined by the user it will be used
        to set the output number of pixels and wavelength scale.
    :param flux: (boolean) True to preserve total flux. In this case the
        log rebinning changes the pixels flux in proportion to their
        dLam so the following command will show large differences
        beween the spectral shape before and after LOG_REBIN:

           plt.plot(exp(logLam), specNew)  # Plot log-rebinned spectrum
           plt.plot(np.linspace(lamRange[0], lamRange[1], spec.size), spec)

        By defaul, when this is False, the above two lines produce
        two spectra that almost perfectly overlap each other.
    :return: [specNew, logLam, velscale] where logLam is the natural
        logarithm of the wavelength and velscale is in km/s.

    """
    lamRange = np.asarray(lamRange)
    assert len(lamRange) == 2, 'lamRange must contain two elements'
    assert lamRange[0] < lamRange[1], 'It must be lamRange[0] < lamRange[1]'
    assert spec.ndim == 1, 'input spectrum must be a vector'
    n = spec.shape[0]
    m = int(n*oversample)

    dLam = np.diff(lamRange)/(n - 1.)        # Assume constant dLam
    lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
    borders = np.linspace(*lim, num=n+1)     # Linearly
    logLim = np.log(lim)

    c = 299792.458                           # Speed of light in km/s
    if velscale is None:                     # Velocity scale is set by user
        velscale = np.diff(logLim)/m*c       # Only for output
    else:
        logScale = velscale/c
        m = int(np.diff(logLim)/logScale)    # Number of output pixels
        logLim[1] = logLim[0] + m*logScale

    newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
    k = (newBorders - lim[0]).clip(0, n-1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0    # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k])*spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

    return specNew, logLam, velscale

###############################################################################

def rebin(x, factor):
    """
    Rebin a vector, or the first dimension of an array,
    by averaging within groups of "factor" adjacent values.

    """
    if factor == 1:
        xx = x
    else:
        xx = x.reshape(len(x)//factor, factor, -1).mean(1).squeeze()

    return xx

###############################################################################

def template_rfft(templates):
    npix_temp = templates.shape[0]
    templates = templates.reshape(npix_temp, -1)
    npad = fftpack.next_fast_len(npix_temp)
    templates_rfft = np.fft.rfft(templates, npad, axis=0)
    
    return templates_rfft,npad

##################################################################################

def convolve_gauss_hermite(templates_rfft,npad, velscale, start, npix,
                           velscale_ratio=1, sigma_diff=0, vsyst=0):
    """
    Convolve a spectrum, or a set of spectra, arranged into columns of an array,
    with a LOSVD parametrized by the Gauss-Hermite series.

    This is intended to reproduce what pPXF does for the convolution and it
    uses the analytic Fourier Transform of the LOSVD introduced in

        Cappellari (2017) http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

    EXAMPLE:
        ...
        pp = ppxf(templates, galaxy, noise, velscale, start,
                  degree=4, mdegree=4, velscale_ratio=ratio, vsyst=dv)

        spec = convolve_gauss_hermite(templates, velscale, pp.sol, galaxy.size,
                                      velscale_ratio=ratio, vsyst=dv)

        # The spectrum below is equal to pp.bestfit to machine precision

        spectrum = (spec @ pp.weights)*pp.mpoly + pp.apoly

    :param spectra: log rebinned spectra
    :param velscale: velocity scale c*dLogLam in km/s
    :param start: parameters of the LOSVD [vel, sig, h3, h4,...]
    :param npix: number of output pixels
    :return: vector or array with convolved spectra

    """
#     npix_temp = templates.shape[0]
#     templates = templates.reshape(npix_temp, -1)
    start = np.array(start)  # make copy
    start[:2] /= velscale
# #     print start
    vsyst /= velscale

#     npad = fftpack.next_fast_len(npix_temp)
#     templates_rfft = np.fft.rfft(templates, npad, axis=0)
    lvd_rfft = losvd_rfft(start, 1, start.shape, templates_rfft.shape[0],
                          1, vsyst, velscale_ratio, sigma_diff)

    conv_temp = np.fft.irfft(templates_rfft*lvd_rfft[:, 0], npad, axis=0)
    conv_temp = rebin(conv_temp[:npix*velscale_ratio, :], velscale_ratio)

    return conv_temp

##################################################################################

def losvd_rfft(pars, nspec, moments, nl, ncomp, vsyst, factor, sigma_diff):
    """
    Analytic Fourier Transform (of real input) of the Gauss-Hermite LOSVD.
    Equation (38) of Cappellari M., 2017, MNRAS, 466, 798
    http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

    """
    losvd_rfft = np.empty((nl, ncomp, nspec), dtype=complex)
    p = 0
    for j, mom in enumerate(moments):  # loop over kinematic components
        for k in range(nspec):  # nspec=2 for two-sided fitting, otherwise nspec=1
            s = 1 if k == 0 else -1  # s=+1 for left spectrum, s=-1 for right one
            vel, sig = vsyst + s*pars[0 + p], pars[1 + p]
            a, b = [vel, sigma_diff]/sig
            w = np.linspace(0, np.pi*factor*sig, nl)
            losvd_rfft[:, j, k] = np.exp(1j*a*w - 0.5*(1 + b**2)*w**2)

            if mom > 2:
                n = np.arange(3, mom + 1)
                nrm = np.sqrt(special.factorial(n)*2**n)   # vdMF93 Normalization
                coeff = np.append([1, 0, 0], (s*1j)**n * pars[p - 1 + n]/nrm)
                poly = hermite.hermval(w, coeff)
                losvd_rfft[:, j, k] *= poly
        p += mom

    return np.conj(losvd_rfft)

##################################################################################

def nnls(A,b,npoly=0):
    
    m, n = A.shape
    AA = np.hstack([A, -A[:, :npoly]])
    x = optimize.nnls(AA, b)[0]
    x[:npoly] -= x[n:]

    return np.array(x[:n])

####################################################################################
# Fast Non-negative least squares
# def fnnls(A,b,npoly=0):
    
#     m, n = A.shape
#     AA = np.hstack([A, -A[:, :npoly]])

#     AAtAA = np.dot(AA.T,AA)
#     AAtb  = np.dot(AA.T,b)
#     x = FNNLSa(AAtAA, AAtb)[0]
#     x[:npoly] -= x[n:]

#     return np.array(x[:n])

####################################################################################

def run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
              auto_stop,burn_in,write_iter,write_thresh,min_iter,max_iter,threads):

	# Create MCMC_chain.csv if it doesn't exist
	if os.path.exists(run_dir+'MCMC_chain.csv')==False:
		f = open(run_dir+'MCMC_chain.csv','w')
		param_string = ', '.join(str(e) for e in param_names)
		f.write('# iter, ' + param_string) # Write initial parameters
		best_str = ', '.join(str(e) for e in init_params)
		f.write('\n 0, '+best_str)
		f.close()


	# initialize the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args,threads=threads)

	start_time = time.time() # start timer

	# Initialize stuff for autocorrelation analysis
	if (auto_stop==True):
		autocorr = []
		old_tau = np.inf # For testing convergence
		min_samp = 1000 # minimum iterations to use past convergence
		tol = 0.01 # tolerance factor (default is 1% = 0.01)
		ntol = 1 # number of it must achieve tol consecutively
		atol = 100 # additive tolerance; number of iterations > tau*tol for convergence
		ncor_times = 10. # multiplicative tolerance; number of correlation times before which we stop sampling
		itol = 0 # iterator for ntol (do not change!)
		conv_type = 'median' # or 'mean'
		stop_iter = max_iter # stopping iteration; changes once convergence is reached
	
	# Run emcee
	for k, result in enumerate(sampler.sample(pos, iterations=max_iter,storechain=True)):
		if (k>1) and ((k+1) % write_iter == 0):
			print('          Iteration = %d ' % (k+1))
		if ((k+1) % write_iter == 0) and ((k+1)>=write_thresh): # Write every [write_iter] iteration
			# Chain location for each parameter
			# Median of last 100 positions for each walker.
			nwalkers = np.shape(sampler.chain)[0]
			npar = np.shape(sampler.chain)[2]
			
			sampler_chain = sampler.chain[:,:k+1,:]
			new_sampler_chain = []
			for i in range(0,np.shape(sampler_chain)[2],1):
			    pflat = sampler_chain[:,:,i] # flattened along parameter
			    flat  = np.concatenate(np.stack(pflat,axis=1),axis=0)
			    new_sampler_chain.append(flat)
			best = []
			for pp in range(0,npar,1):
				data = new_sampler_chain[pp][-int(nwalkers*write_iter):]
				med = np.median(data)
				best.append(med)
			# write to file
			f = open(run_dir+'MCMC_chain.csv','a')
			best_str = ', '.join(str(e) for e in best)
			f.write('\n'+str(k+1)+', '+best_str)
			f.close()
		if ((k+1) % write_iter == 0) and ((k+1)>=min_iter) and ((k+1)>=write_thresh) and (auto_stop==True):
			# Autocorrelation analysis of chain to determine convergence 
			tau = autocorr_convergence(sampler.chain,param_names,plot=False)
			if (conv_type == 'median'):
				autocorr.append(np.median(tau)) # 
			elif (conv_type == 'mean'):	 
				autocorr.append(np.mean(tau)) # 
			# Count convergences (if ntol >= 1)
			if (conv_type == 'median'):
				if ((np.median(tau) * ncor_times + atol)  < (k+1)) & (np.abs(np.median(old_tau) - np.median(tau)) / np.median(tau) < tol) & (itol < int(ntol)):
					itol += 1
				else: 
					itol = 0 
			elif (conv_type == 'mean'):	
				if ((np.mean(tau) * ncor_times + atol) < (k+1)) & (np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < tol) & (itol < int(ntol)):
					itol += 1
				else: 
					itol = 0
			# Check convergence
			if (conv_type == 'median'):
				if ((np.median(tau) * ncor_times + atol) < (k+1)) & (np.abs(np.median(old_tau) - np.median(tau)) / np.median(tau) < tol) & (itol >= int(ntol)) & (stop_iter == max_iter):
					print('\n Converged at %d iterations. \n' % (k+1))
					burn_in = (k+1)
					stop_iter = (k+1)+min_samp
			elif (conv_type == 'mean'):	
				if ((np.mean(tau) * ncor_times + atol)  < (k+1)) & (np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < tol) & (itol >= int(ntol)) & (stop_iter == max_iter):
					print('\n Converged at %d iterations. \n' % (k+1))
					burn_in = (k+1)
					stop_iter = (k+1)+min_samp
			if ((k+1) == stop_iter):
				break

			if (conv_type == 'median'):
				print("""
				          Current iteration: %d
				          Current autocorrelation time: %d
				          Current tolerance %0.2f%% 

				""" % ((k+1),ncor_times*np.median(tau)+atol,100.*(np.abs(np.median(old_tau) - np.median(tau)) / np.median(tau))))
			elif (conv_type == 'mean'):	 
				print("""
				          Current iteration: %d
				          Current autocorrelation time: %d
				          Current tolerance %0.2f%% 

				""" % ((k+1),ncor_times*np.mean(tau)+atol,100.*(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau))))

			old_tau = tau


			

	elap_time = (time.time() - start_time)	   
	print("\n Runtime = %s. \n" % (time_convert(elap_time)))   

	# Remove excess zeros from sampler chain if emcee converged on a solution
	# in fewer iterations than max_iter
	# Remove zeros from all chains
	a = [] # the zero-trimmed sampler.chain
	for p in range(0,np.shape(sampler.chain)[2],1):
	    c = sampler.chain[:,:,p]
	    c_trimmed = [np.delete(c[i,:],np.argwhere(c[i,:]==0)) for i in range(np.shape(c)[0])] # delete any occurence of zero 
	    a.append(c_trimmed)
	a = np.swapaxes(a,1,0) 
	a = np.swapaxes(a,2,1)

	return a, burn_in


##################################################################################

# Autocorrelation analysis 
##################################################################################

def autocorr_convergence(emcee_chain,param_names,plot=False):
	"""
	My own recipe for convergence.
	
	Description: 
	
	Parameters:
	    emcee_chain : the (non-zero-trimmed, excess zeros NOT removed) sampler chain output from emcee
	    param_names : parameter names 
	    conv_check : convergence checking frequency.  For example, if conv_check=100, the alogorithm will 
	                 check every 100 iterations for convergence.
	    conv_box : convergence box.  The interval over which convergence is checked.
	    conv_samp : (int) sampling of the convergence box.  Default = 10.  
	    stddev_p : (float) precision in (Gaussian) standard deviations for the convergence interval; default = 0.5
	    stdout : (optional; default False)
	    plot : (optional; default False)
	    
	"""
	# Remove zeros from all chains
	sampler_chain = []
	for p in range(0,np.shape(emcee_chain)[2],1):
	    c = emcee_chain[:,:,p]
	    c_trimmed = [np.delete(c[i,:],np.argwhere(c[i,:]==0)) for i in range(np.shape(c)[0])] # delete any occurence of zero 
	    sampler_chain.append(c_trimmed)
	sampler_chain = np.swapaxes(sampler_chain,1,0) 
	sampler_chain = np.swapaxes(sampler_chain,2,1)


	    
	nwalker = np.shape(sampler_chain)[0] # Number of walkers
	niter   = np.shape(sampler_chain)[1] # Number of iterations
	npar    = np.shape(sampler_chain)[2] # Number of parameters
	    
	def autocorr_func(c_x):
	    """"""
	    acf = []
	    for p in range(0,np.shape(c_x)[1],1):
	        x = c_x[:,p]
	        # Subtract mean value
	        rms_x = np.median(x)
	        x = x - rms_x
	        cc = np.correlate(x,x,mode='full')
	        cc = cc[cc.size // 2:]
	        cc = cc/np.max(cc)
	        acf.append(cc)
	    # Flip the array 
	    acf = np.swapaxes(acf,1,0)
	    return acf
	        
	def auto_window(taus, c):
	    """
	    (Adapted from https://github.com/dfm/emcee/blob/master/emcee/autocorr.py)
	    """
	    m = np.arange(len(taus)) < c * taus
	    if np.any(m):
	        return np.argmin(m)
	    return len(taus) - 1
	
	def integrated_time(acf, c=5, tol=0):
	    """Estimate the integrated autocorrelation time of a time series.
	    This estimate uses the iterative procedure described on page 16 of
	    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
	    determine a reasonable window size.
	    Args:
	        acf: The time series. If multidimensional, set the time axis using the
	            ``axis`` keyword argument and the function will be computed for
	            every other axis.
	        c (Optional[float]): The step size for the window search. (default:
	            ``5``)
	        tol (Optional[float]): The minimum number of autocorrelation times
	            needed to trust the estimate. (default: ``0``)
	    Returns:
	        float or array: An estimate of the integrated autocorrelation time of
	            the time series ``x`` computed along the axis ``axis``.
	    (Adapted from https://github.com/dfm/emcee/blob/master/emcee/autocorr.py)
	    """
	    tau_est = np.empty(np.shape(acf)[1])
	    windows = np.empty(np.shape(acf)[1], dtype=int)

	    # Loop over parameters
	    for p in range(0,np.shape(acf)[1],1):
	        taus = 2.0*np.cumsum(acf[:,p])-1.0
	        windows[p] = auto_window(taus, c)
	        tau_est[p] = taus[windows[p]]

	    return tau_est

	c_x = np.mean(sampler_chain[:,:,:],axis=0)
	
	acf = autocorr_func(c_x)
	tau_est = integrated_time(acf)
	    
	if (plot==True):
	    fig = plt.figure(figsize=(14,4))
	    ax1 = fig.add_subplot(2,1,1)
	    ax2 = fig.add_subplot(2,1,2)
	    for c in range(0,np.shape(c_x)[1],1):
	        cn = (c_x[:,c])/(np.median(c_x[:,c]))
	        ax1.plot(cn,alpha=1.,linewidth=0.5)
	    ax1.axhline(1.0,alpha=1.,linewidth=0.5,color='black',linestyle='--')  
	    ax1.set_xlim(0,np.shape(c_x)[0])
	    ax2.plot(range(np.shape(acf)[0]),acf,alpha=1.,linewidth=0.5,label='ACF')
	    ax2.axhline(0.0,alpha=1.,linewidth=0.5)
	    ax2.set_xlim(np.min(range(np.shape(acf)[0])),np.max(range(np.shape(acf)[0])))
	    plt.tight_layout()
	    
	    
	return tau_est

##################################################################################


# Plotting Routines
##################################################################################

def conf_int(x,prob,factor,flat): # Function for calculating an arbitrary confidence interval (%)
    
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx#,array[idx],
    
    max_before = np.max(prob)
    # Normalize (or make sure its normalized)
    prob = prob / np.sum(prob)
    # Interpolate the PDF 
    factor = factor # resampling factor
    xnew = np.linspace(np.min(x),np.max(x),factor*len(x))
    interp_pdf = interp1d(x,prob,kind='cubic',fill_value=(x[0],x[-1]))
    pdfnew = interp_pdf(xnew)
#     pdfnew = shift(pdfnew,shift=-factor/2.0,mode='nearest')
    pdfnew = pdfnew / np.sum(pdfnew) # * factor # Re-sampling increases probability - must renormalize
    max_after = np.max(pdfnew)# maximum after interpolation and renormalization
    scale_factor = max_before/max_after
#     print scale_factor
    # Get the max (most probable) value of the interpolated function
    pdfmax = xnew[np.where(np.isclose(pdfnew,np.max(pdfnew)))][0]
#     print pdfmax,scale_factor
    # Get x-values to the left of the maximum 
    xh_left = xnew[np.where(xnew < pdfmax)]
    yh_left = pdfnew[np.where(xnew < pdfmax)]

    # Get x-values to the right of the maximum 
    xh_right = xnew[np.where(xnew > pdfmax)]
    yh_right = pdfnew[np.where(xnew > pdfmax)]
#     print len(yh_right)
    try:
#     if 1:
        for l in range(0,len(yh_right),1):
            idx = find_nearest(yh_left,yh_right[l])
            xvec = xnew[[i for i,j in enumerate(xnew) if xh_left[idx]<=j<=xh_right[l]]] # x vector for simps
            yvec = pdfnew[[i for i,j in enumerate(xnew) if xh_left[idx]<=j<=xh_right[l]]] # y vector for simps
            integral = simps(y=yvec)#,x=xvec)
#             print l,round(integral,2),np.min(xvec),np.max(xvec)
            if round(integral,2) == 0.68: #68.0/100.0:
                # 1 sigma = 68% confidence interval
                conf_interval_1 = [pdfmax - np.min(xvec),np.max(xvec) - pdfmax]
                
        return pdfmax,conf_interval_1[0],conf_interval_1[1],xvec,yvec*scale_factor
#                 conf_interval_2[0],conf_interval_2[1],\
#                 conf_interval_3[0],conf_interval_3[1],

    except: 
        # print("\n          Error: Cannot determine confidence interval.  Using median and std. dev. from flattened chain...")
        # mode = x[np.where(prob==np.max(prob))]
        # mean = simps(x*prob,x)
        med = np.median(flat)
        # std = np.sqrt(simps((x-mean)**2*prob,x))
        std = np.std(flat)
        return med,std,std,x,np.zeros(len(prob))


def get_bin_centers(bins):
        bins = bins[:-1]
        bin_width = bins[1]-bins[0]
        new_bins =  bins + bin_width/2.0
        return new_bins


def param_plots(param_dict,burn_in,run_dir,save_plots=True):
	# Create a histograms sub-folder
	if (os.path.exists(run_dir + 'param_histograms')==False):
		os.mkdir(run_dir + 'param_histograms')

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])
	for key in param_dict:
		print('          %s' % key)
		# Clear axes
		ax1.clear()
		ax2.clear()
		ax3.clear()

		chain = param_dict[key]['chain'] # shape = (nwalkers,niter)
		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		flat = chain[:,burn_in:]
		flat = flat.flat

		pname = param_dict[key]['label']
		pcol  = param_dict[key]['pcolor']

		# Histogram
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, facecolor=pcol, alpha=0.75)

		pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,10,flat)
		# Store values in dictionary
		param_dict[key]['par_best'] = pdfmax
		param_dict[key]['sig_low']  = low1
		param_dict[key]['sig_upp']  = upp1


		# Plot 1: Histogram plots
		ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.6f$\n' % pdfmax)
		ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.6f$\n' % low1)
		ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.6f$\n' % upp1)
		ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'%s' % pname,fontsize=12)
		ax1.set_ylabel(r'$p$(%s)' % pname,fontsize=12)

		# Plot 2: best fit values
		ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.6f$\n' % pdfmax)
		ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.6f$\n' % low1)
		ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.6f$\n' % upp1)
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Plot 3: Chain plot
		for w in range(0,np.shape(chain)[0],1):
			ax3.plot(range(np.shape(chain)[1]),chain[w,:],color='white',linewidth=0.5,alpha=0.5,zorder=0)
		# Calculate meanand standard deviation of walkers at each iteration 
		c_avg = np.mean(chain,axis=0)
		c_std = np.std(chain,axis=0)
		ax3.plot(range(np.shape(chain)[1]),c_avg,color='xkcd:red',alpha=1.,linewidth=2.0,label='Mean',zorder=10)
		ax3.fill_between(range(np.shape(chain)[1]),c_avg+c_std,c_avg-c_std,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Std. Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in')

		ax3.set_xlim(0,np.shape(chain)[1])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'%s' % pname,fontsize=12)
		ax3.legend(loc='upper left')

		# Save the figure
		if (save_plots==True):
			figname = param_dict[key]['name']
			plt.savefig(run_dir+'param_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=300,fmt='png')

	return param_dict


def emline_flux_plots(burn_in,nwalkers,run_dir,save_plots=True):
	# Create a histograms sub-folder
	if (os.path.exists(run_dir + 'param_histograms')==False):
		os.mkdir(run_dir + 'param_histograms')

	# Open the file containing the line flux data
	df = pd.read_csv(run_dir+'em_line_fluxes.csv')
	keys = list(df.columns.values)[1:]
	df = df.drop(df.columns[0], axis=1)     
	# Create a flux dictionary
	flux_dict = {}
	for i in range(0,len(df.columns),1):
		flux_dict[keys[i]]= {'chain':np.array(df[df.columns[i]])}

	# Burn-in for fluxes and luminosities 
	burn_in = burn_in * nwalkers

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])
	for key in flux_dict:
		print('          %s' % key)
		# Clear axes
		ax1.clear()
		ax2.clear()
		ax3.clear()

		chain = flux_dict[key]['chain'] # shape = (nwalkers,niter)
		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[0]):
			burn_in = int(0.5*np.shape(chain)[0])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		flat = chain[burn_in:]
		flat = flat.flat

		# Histogram
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, alpha=0.75)

		pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,10,flat)
		# Store values in dictionary
		flux_dict[key]['par_best'] = pdfmax
		flux_dict[key]['sig_low']  = low1
		flux_dict[key]['sig_upp']  = upp1


		# Plot 1: Histogram plots
		ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.6f$\n' % pdfmax)
		ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.6f$\n' % low1)
		ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.6f$\n' % upp1)
		ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'%s' % key,fontsize=8)
		ax1.set_ylabel(r'$p$(%s)' % key,fontsize=8)

		# Plot 2: best fit values
		ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.6f$\n' % pdfmax)
		ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.6f$\n' % low1)
		ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.6f$\n' % upp1)
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Plot 3: Chain plot
		ax3.plot(range(np.shape(chain)[0]),chain,color='white',linewidth=0.5,alpha=1.0,zorder=0)
		# Calculate meanand standard deviation of walkers at each iteration 
		c_avg = np.median(chain[burn_in:],axis=0)
		c_std = np.std(chain[burn_in:],axis=0)
		ax3.plot(range(np.shape(chain)[0])[burn_in:],np.full(np.shape(chain[burn_in:])[0],c_avg),color='xkcd:red',alpha=1.,linewidth=2.0,label='Mean',zorder=10)
		ax3.fill_between(range(np.shape(chain)[0])[burn_in:],c_avg+c_std,c_avg-c_std,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Std. Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in')

		ax3.set_xlim(0,np.shape(chain)[0])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'%s' % key,fontsize=8)
		ax3.legend(loc='upper left')

		# Save the figure
		if (save_plots==True):
			figname = key
			plt.savefig(run_dir+'param_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=300,fmt='png')

	return flux_dict

def flux2lum(flux_dict,burn_in,nwalkers,z,run_dir,H0=71.0,Om0=0.27,save_plots=True):
	# Create a histograms sub-folder
	if (os.path.exists(run_dir + 'param_histograms')==False):
		os.mkdir(run_dir + 'param_histograms')

	# Extract flux dict and keys
	# lum_keys = []
	# for key in flux_dict:
	# 	keys.append(key[:-4]+'lum')
   
	# Create a flux dictionary
	lum_dict = {}
	for key in flux_dict:
		flux = (flux_dict[key]['chain']) * 1.0E-17
		# Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
		cosmo = FlatLambdaCDM(H0, Om0)
		d_mpc = cosmo.luminosity_distance(z).value
		d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
		# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
		lum   = (flux * 4*np.pi * d_cm**2    ) / 1.0E+42

		lum_dict[key[:-4]+'lum']= {'chain':lum}

	# Burn-in for fluxes and luminosities 
	burn_in = burn_in * nwalkers

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])
	for key in lum_dict:
		print('          %s' % key)
		# Clear axes
		ax1.clear()
		ax2.clear()
		ax3.clear()

		chain = lum_dict[key]['chain'] # shape = (nwalkers,niter)
		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[0]):
			burn_in = int(0.5*np.shape(chain)[0])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		flat = chain[burn_in:]
		flat = flat.flat

		# Histogram
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, alpha=0.75)

		pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,10,flat)
		# Store values in dictionary
		lum_dict[key]['par_best'] = pdfmax
		lum_dict[key]['sig_low']  = low1
		lum_dict[key]['sig_upp']  = upp1


		# Plot 1: Histogram plots
		ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.6f$\n' % pdfmax)
		ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.6f$\n' % low1)
		ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.6f$\n' % upp1)
		ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'%s' % key,fontsize=8)
		ax1.set_ylabel(r'$p$(%s)' % key,fontsize=8)

		# Plot 2: best fit values
		ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.6f$\n' % pdfmax)
		ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.6f$\n' % low1)
		ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.6f$\n' % upp1)
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Plot 3: Chain plot
		ax3.plot(range(np.shape(chain)[0]),chain,color='white',linewidth=0.5,alpha=1.0,zorder=0)
		# Calculate meanand standard deviation of walkers at each iteration 
		c_avg = np.median(chain[burn_in:],axis=0)
		c_std = np.std(chain[burn_in:],axis=0)
		ax3.plot(range(np.shape(chain)[0])[burn_in:],np.full(np.shape(chain[burn_in:])[0],c_avg),color='xkcd:red',alpha=1.,linewidth=2.0,label='Mean',zorder=10)
		ax3.fill_between(range(np.shape(chain)[0])[burn_in:],c_avg+c_std,c_avg-c_std,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Std. Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in')

		ax3.set_xlim(0,np.shape(chain)[0])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'%s' % key,fontsize=8)
		ax3.legend(loc='upper left')

		# Save the figure
		if (save_plots==True):
			figname = key
			plt.savefig(run_dir+'param_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=300,fmt='png')

	return lum_dict
		

def write_param(param_dict,flux_dict,lum_dict,run_dir):
	# Extract elements from dictionaries
	par_names = []
	par_best  = []
	sig_low   = []
	sig_upp   = []
	for key in param_dict:
	    par_names.append(key)
	    par_best.append(param_dict[key]['par_best'])
	    sig_low.append(param_dict[key]['sig_low'])
	    sig_upp.append(param_dict[key]['sig_upp'])
	for key in flux_dict:
	    par_names.append(key)
	    par_best.append(flux_dict[key]['par_best'])
	    sig_low.append(flux_dict[key]['sig_low'])
	    sig_upp.append(flux_dict[key]['sig_upp'])    
	for key in lum_dict:
	    par_names.append(key)
	    par_best.append(lum_dict[key]['par_best'])
	    sig_low.append(lum_dict[key]['sig_low'])
	    sig_upp.append(lum_dict[key]['sig_upp']) 
	# Print pars 
	if 0: 
	    for i in range(0,len(par_names),1):
	        print par_names[i],par_best[i],sig_low[i],sig_upp[i]
	# Write best-fit paramters to FITS table
	col1 = fits.Column(name='parameter', format='20A', array=par_names)
	col2 = fits.Column(name='best_fit', format='E', array=par_best)
	col3 = fits.Column(name='sigma_low', format='E', array=sig_low)
	col4 = fits.Column(name='sigma_upp', format='E', array=sig_upp)
	cols = fits.ColDefs([col1,col2,col3,col4])
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'par_table.fits',overwrite=True)

def write_chain(param_dict,flux_dict,lum_dict,run_dir):

	cols = []
	# Construct a column for each parameter and chain
	for key in param_dict:
		cols.append(fits.Column(name=key, format='E', array=param_dict[key]['chain'].flat))
	for key in flux_dict:
		cols.append(fits.Column(name=key, format='E', array=flux_dict[key]['chain']))
	for key in lum_dict:
		cols.append(fits.Column(name=key, format='E', array=lum_dict[key]['chain']))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'MCMC_chains.fits',overwrite=True)
    
def plot_best_model(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
                           temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir):

	param_names  = [param_dict[key]['name'] for key in param_dict ]
	par_best       = [param_dict[key]['par_best'] for key in param_dict ]


	output_model = True
	fit_type     = 'final'
	move_temp    = False
	comp_dict = fit_model(par_best,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
			  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
			  fit_type,move_temp,output_model)

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,6)) 
	gs = gridspec.GridSpec(4, 1)
	gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0:3,0])
	ax2  = plt.subplot(gs[3,0])
    
	for key in comp_dict:
		# Galaxy + Best-fit Model
		if (key is not 'residuals') and (key is not 'noise') and (key is not 'wave') and (key is not 'data'):
			ax1.plot(lam_gal,comp_dict[key]['comp'],linewidth=comp_dict[key]['linewidth'],color=comp_dict[key]['pcolor'],label=key,zorder=15)

	ax1.plot(lam_gal,comp_dict['data']['comp'],linewidth=0.5,color='white',label='data',zorder=0)

	ax1.set_xticklabels([])
	ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	ax1.set_ylim(-0.5*np.median(comp_dict['model']['comp']),np.max([comp_dict['data']['comp'],comp_dict['model']['comp']]))
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
	# ax1.legend(loc='best',fontsize=8)
	# Residuals
	sigma_resid = np.std(comp_dict['data']['comp']-comp_dict['model']['comp'])
	sigma_noise = np.median(comp_dict['noise']['comp'])
	ax2.plot(lam_gal,(comp_dict['noise']['comp'])*3,linewidth=comp_dict['noise']['linewidth'],color=comp_dict['noise']['pcolor'],label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
	ax2.plot(lam_gal,(comp_dict['residuals']['comp'])*3,linewidth=comp_dict['residuals']['linewidth'],color=comp_dict['residuals']['pcolor'],label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
	ax2.axhline(0.0,linewidth=1.0,color='black',linestyle='--')
	ax2.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	ax2.set_ylim(ax1.get_ylim())
	ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
	ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
	ax2.set_yticks([0.0])
	ax2.legend(loc='upper right',fontsize=8)
	plt.savefig(run_dir+'bestfit_model.pdf',dpi=150,fmt='png')

	cols = []
	# Construct a column for each parameter and chain
	for key in comp_dict:
		cols.append(fits.Column(name=key, format='E', array=comp_dict[key]['comp']))
	# Write to fits

	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'best_model_components.fits',overwrite=True)

	return None


# Plotting Routines
##################################################################################

def cleanup(run_dir):
	# Remove emission line fluxes csv 
	os.remove(run_dir + 'em_line_fluxes.csv')
	# Remove templates folder if empty
	if not os.listdir(run_dir + 'templates'):
		shutil.rmtree(run_dir + 'templates')
	# Remove param_plots folder if empty
	if not os.listdir(run_dir + 'param_histograms'):
		shutil.rmtree(run_dir + 'param_histograms')

##################################################################################
