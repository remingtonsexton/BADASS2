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
from scipy.stats import kde
from scipy.integrate import simps
from astropy.io import fits
import glob
from time import clock
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

##################################################################################
# Prepare SDSS spectrum for pPXF

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

def sdss_prepare(file,fitreg,temp_dir,run_dir):
    """
    Prepare an SDSS spectrum for pPXF, returning all necessary 
    parameters. 
    
    file: fully-specified path of the spectrum 
    z: the redshift; we use the SDSS-measured redshift
    fitreg: (min,max); tuple specifying the minimum and maximum 
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
    fit_min,fit_max = float(fitreg[0]),float(fitreg[1])
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
    # vazdekis = glob.glob(temp_dir + '/Mun1.30Z*.fits')#[:num_temp]
    vazdekis = glob.glob(temp_dir + '/*.fits')#[:num_temp]

    vazdekis.sort() # Sort them in the order they appear in the directory
    # fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
    fwhm_tem = 1.35 # Indo-US Template Library FWHM

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam_temp = np.array(h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1']))
    # By cropping the templates we save some fitting time
    mask_temp = ( (lam_temp > (fit_min-200.)) & (lam_temp < (fit_max+200.)) )
    ssp = ssp[mask_temp]
    lam_temp = lam_temp[mask_temp]

    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    sspNew = log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
    templates = np.empty((sspNew.size, len(vazdekis)))
    
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

    for j, fname in enumerate(vazdekis):
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

    # Plot the galaxy+ templates
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.step(lam_gal,galaxy,label='Galaxy')
    ax1.fill_between(lam_gal,galaxy-noise,galaxy+noise,color='gray',alpha=0.5)
    ax2.plot(np.exp(loglam_temp),templates[:,-25:],alpha=0.5,label='Template')
    ax1.set_xlabel(r'Wavelength, $\lambda$ ($\mathrm{\AA}$)',fontsize=12)
    ax2.set_xlabel(r'Wavelength, $\lambda$ ($\mathrm{\AA}$)',fontsize=12)
    ax1.set_ylabel(r'$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
    ax2.set_ylabel(r'Normalized Flux',fontsize=12)
    ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig(run_dir+'sdss_prepare.png',dpi=300,fmt='png')
    
    
    return lam_gal,galaxy,templates,noise,fwhm_gal,velscale,dv,vazdekis,z,ebv

##################################################################################

####################### Galactic Extinction Correction ###########################

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

##################################################################################

# pPXF Routines (from Cappellari 2017)
##################################################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

##################################################################################

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

# Maximum Likelihood (initial fitting), Prior, and log Probability functions

def lnlike_init_outflows(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,sigma,run_dir):
    # Create model
    model = init_outflow_model(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,run_dir)
    # Calculate log-likelihood
    l = -0.5*(galaxy-model)**2/(sigma)**2
    l = np.sum(l,axis=0)
    return l #l,model

def lnlike_init_no_outflows(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,sigma,run_dir):
    # Create model
    model = init_no_outflow_model(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,run_dir)
    # Calculate log-likelihood
    l = -0.5*(galaxy-model)**2/(sigma)**2
    l = np.sum(l,axis=0)
    return l #l,model

def max_likelihood_outflows(params,param_limits,args):
    lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir = args
    # This function performs an initial maximum likelihood
    # estimation to acquire robust initial parameters.

    # Define constraint functions for the outflow components
    # Free-parameters (14 total):
    #------------------------------------------
    # [0] - Galaxy template amplitude
    # [1] - Narrow FeII amplitude
    # [2] - Broad FeII amplitude
    # [3] - Na. [OIII]5007 Core Amplitude
    # [4] - Na. [OIII]5007 Core FWHM
    # [5] - Na. [OIII]5007 Core VOFF
    # [6] - Br. [OIII]5007 Outflow amplitude
    # [7] - Br. [OIII]5007 Outflow FWHM
    # [8] - Br. [OIII]5007 Outflow VOFF
    #      - Br. [OIII]4959 Outflow and Na. H-beta outflows are tied to [OIII]5007 outflow
    # [9] - Na. H-beta amplitude
    #      - Na. H-beta FWHM tied to [OIII]5007
    # [10] - Na. H-beta VOFF
    # [11] - Br. H-beta amplitude
    # [12] - Br. H-beta FWHM
    # [13] - Br. H-beta VOFF
    def amp_constraint(p):
        return p[3]-p[6]

    def fwhm_constraint(p):
        return p[7]-p[4]

    def voff_constraint(p):
        return p[5]-p[8]

    cons = [{'type':'ineq','fun': fwhm_constraint },
            {'type':'ineq','fun': voff_constraint },
            {'type':'ineq','fun':  amp_constraint }]
    
    # cons = [{'type':'ineq','fun': fwhm_constraint },
    #         {'type':'ineq','fun': voff_constraint }]

    # Specify bounds for maximum likelihood fit
    bounds = []
    for i in range(0,len(param_limits[0]),1):
        bounds.append((param_limits[0][i],param_limits[1][i]))
    # Perform maximum likelihood estimation for initial guesses of MCMC fit
    nll = lambda *args: -lnlike_init_outflows(*args)
    result = op.minimize(fun = nll, x0 = params, \
             args=(lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir),\
             method='SLSQP', bounds = bounds, constraints=cons, options={'maxiter':2500,'disp': True})
    # result = op.minimize(fun = nll, x0 = params, \
    #          args=(lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir),\
    #          method='L-BFGS-B', bounds = bounds, options={'maxiter':2500,'disp': True})
    return result

def max_likelihood_no_outflows(params,param_limits,args):
    lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir = args
    # This function performs an initial maximum likelihood
    # estimation to acquire robust initial parameters.
    # Specify bounds for maximum likelihood fit
    bounds = []
    for i in range(0,len(param_limits[0]),1):
        bounds.append((param_limits[0][i],param_limits[1][i]))
    # Perform maximum likelihood estimation for initial guesses of MCMC fit
    nll = lambda *args: -lnlike_init_no_outflows(*args)
    result = op.minimize(fun = nll, x0 = params, \
             args=(lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir),\
             method='SLSQP', bounds = bounds,options={'maxiter':2500,'disp': True})
    # result = op.minimize(fun = nll, x0 = params, \
    #          args=(lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir),\
    #          method='L-BFGS-B', bounds = bounds, options={'maxiter':2500,'disp': True})
    return result
   
def lnlike_final_outflows(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,sigma,temp_list,run_dir):
    # Create model
    model = final_outflow_model(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,temp_list,run_dir,move_temp=1)
    # Calculate log-likelihood
    l = -0.5*(galaxy-model)**2/(sigma)**2
    l = np.sum(l,axis=0)
    return l #l,model


def lnlike_final_no_outflows(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,sigma,temp_list,run_dir):
    # Create model
    model = final_no_outflow_model(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,temp_list,run_dir,move_temp=1)
    # Calculate log-likelihood
    l = -0.5*(galaxy-model)**2/(sigma)**2
    l = np.sum(l,axis=0)
    return l #l,model

def lnprior_outflow(bounds,params):  
    # Free-parameters (17 total):
    #------------------------------------------
    # [0]  - Stellar velocity
    # [1]  - Stellar velocity dispersion
    # [2]  - AGN simple power-law slope
    # [3]  - AGN simple power-law amplitude
    # [4]  - Narrow FeII amplitude
    # [5]  - Broad FeII amplitude
    # [6]  - Na. [OIII]5007 Core Amplitude
    # [7]  - Na. [OIII]5007 Core FWHM
    # [8]  - Na. [OIII]5007 Core VOFF
    # [9]  - Br. [OIII]5007 Outflow amplitude
    # [10] - Br. [OIII]5007 Outflow FWHM
    # [11] - Br. [OIII]5007 Outflow VOFF
    # [12] - Na. H-beta amplitude
    # [13] - Na. H-beta VOFF
    # [14] - Br. H-beta amplitude
    # [15] - Br. H-beta FWHM
    # [16] - Br. H-beta VOFF
    # if np.all((params >= bounds[0]) & (params <= bounds[1])):
    #     return 0.0
    if np.all((params >= bounds[0]) & (params <= bounds[1])) & (params[6] > params[9]) & (params[10] > params[7]) & (params[8] > params[11]):
        return 0.0
    return -np.inf 

def lnprior_no_outflow(bounds,params):  
    if np.all((params >= bounds[0]) & (params <= bounds[1])):
        return 0.0
    return -np.inf 

def lnprob(params,bounds,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,sigma,temp_list,run_dir,final_model):
    if (final_model=='outflow'):
        lp = lnprior_outflow(bounds,params)
    elif (final_model=='NO_outflow'):
        lp = lnprior_no_outflow(bounds,params)
    ##############################################
    if not np.isfinite(lp):
        return -np.inf
    elif (final_model=='outflow') and (np.isfinite(lp)==True):
        return lp + lnlike_final_outflows(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,sigma,temp_list,run_dir)
    elif (final_model=='NO_outflow') and (np.isfinite(lp)==True):
        return lp + lnlike_final_no_outflows(params,lam_gal,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,dv,galaxy,sigma,temp_list,run_dir)


####################################################################################

# Model Functions

def initialize_mcmc_init_outflows(lam_gal,galaxy,velscale):
    ################################################################################
    # Initial conditions for some parameters
    max_flux = np.max(galaxy)
    total_flux_init = np.median(galaxy[(lam_gal>5025.) & (lam_gal<5800.)])
    # cont_flux_init = 0.01*(np.median(galaxy))
    feii_flux_init= (0.1*np.median(galaxy))
    hb_amp_init= (np.max(galaxy[(lam_gal>4862-50) & (lam_gal<4862+50)]))
    oiii5007_amp_init = (np.max(galaxy[(lam_gal>5007-50) & (lam_gal<5007+50)]))

    # print('Total flux level: %0.2e' % total_flux_init)
    # print('Continuum flux level: %0.2e' % cont_flux_init )
    # print('FeII flux level: %0.2e' % feii_flux_init)
    
    ################################################################################
    
    mcmc_input = [] # array of parameter dicts
    # [0] - Galaxy template amplitude
    mcmc_input.append({'name':'$A_\mathrm{gal}$','init':0.99*total_flux_init,'pmin':0.0,'pmax':max_flux})
    # [1] - Narrow FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Na\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [2] - Broad FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Br\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [3] - Na. [OIII]5007 Core Amplitude
    mcmc_input.append({'name':'$A_\mathrm{[OIII]5007\;Core}$' ,'init':(oiii5007_amp_init-total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [4] - Na. [OIII]5007 Core FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Core}$','init':250.,'pmin':0.0,'pmax':1000.})
    # [5] - Na. [OIII]5007 Core VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Core}$','init':0.,'pmin':-1000.,'pmax':1000.})
    # [6] - Br. [OIII]5007 Outflow amplitude
    mcmc_input.append({'name':'$A_\mathrm{[OIII]5007\;Outflow}$' ,'init':(oiii5007_amp_init-total_flux_init)/2.,'pmin':0.0,'pmax':max_flux})
    # [7] - Br. [OIII]5007 Outflow FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Outflow}$','init':450.,'pmin':0.0,'pmax':5000.})
    # [8] - Br. [OIII]5007 Outflow VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Outflow}$','init':-50.,'pmin':-2000.,'pmax':2000.})
    # Br. [OIII]4959 Outflow is tied to all components of [OIII]5007 outflow
    # [9] - Na. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Na.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init) ,'pmin':0.0 ,'pmax':max_flux})
    # Na. H-beta FWHM tied to [OIII]5007 FWHM
    # [10] Na. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Na.\;Hb}}$','init':0.,'pmin':-1000 ,'pmax':1000.})
    # [11] - Br. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Br.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init)/2.0  ,'pmin':0.0,'pmax':max_flux})
    # [12] - Br. H-beta FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$','init':2500.,'pmin':0.0,'pmax':10000.})
    # [13] - Br. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Br.\;Hb}}$','init':0.,'pmin':-1000. ,'pmax':1000.})

    param_labels,params,param_limits,param_init = mcmc_structures(mcmc_input)
    
    return param_labels,params,param_limits,param_init
    
    ################################################################################    


def initialize_mcmc_final_outflows(lam_gal,galaxy,templates,velscale):
    ################################################################################
    # Initial conditions for some parameters
    max_flux = np.max(galaxy)
    total_flux_init = np.median(galaxy[(lam_gal>5025.) & (lam_gal<5800.)])
    # cont_flux_init = (0.5*np.median(galaxy))
    feii_flux_init= (0.1*np.median(galaxy))
    hb_amp_init= (np.max(galaxy[(lam_gal>4862-50) & (lam_gal<4862+50)]))
    oiii5007_amp_init = (np.max(galaxy[(lam_gal>5007-50) & (lam_gal<5007+50)]))

    # print('Total flux level: %0.2e' % total_flux_init)
    # print('Continuum flux level: %0.2e' % cont_flux_init )
    # print('FeII flux level: %0.2e' % feii_flux_init)
    
    ################################################################################
    
    mcmc_input = [] # array of parameter dicts
    # [0] - Stellar velocity
    mcmc_input.append({'name':'$V_*$','init':100. ,'pmin':-500. ,'pmax':500.})
    # [1] - Stellar velocity dispersion
    mcmc_input.append({'name':'$\sigma_*$','init':200.0,'pmin':30.0,'pmax':400.})
    # [2] - AGN simple power-law slope
    mcmc_input.append({'name':'$m_\mathrm{cont}$','init':0.0  ,'pmin':-4.0,'pmax':2.0})
    # [3] - AGN simple power-law amplitude
    mcmc_input.append({'name':'$A_\mathrm{cont}$','init':(0.01*total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [4] - Narrow FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Na\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [5] - Broad FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Br\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [6] - Na. [OIII]5007 Core Amplitude
    mcmc_input.append({'name':'$A_\mathrm{[OIII]5007\;Core}$' ,'init':(oiii5007_amp_init-total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [7] - Na. [OIII]5007 Core FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Core}$','init':250.,'pmin':0.0,'pmax':1000.})
    # [8] - Na. [OIII]5007 Core VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Core}$','init':0.,'pmin':-1000.,'pmax':1000.})
    # [9] - Br. [OIII]5007 Outflow amplitude
    mcmc_input.append({'name':'$A_\mathrm{[OIII]5007\;Outflow}$' ,'init':(oiii5007_amp_init-total_flux_init)/2.,'pmin':0.0,'pmax':max_flux})
    # [10] - Br. [OIII]5007 Outflow FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Outflow}$','init':450.,'pmin':0.0,'pmax':5000.})
    # [11] - Br. [OIII]5007 Outflow VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Outflow}$','init':-50.,'pmin':-2000.,'pmax':2000.})
    # Br. [OIII]4959 Outflow is tied to all components of [OIII]5007 outflow
    # [12] - Na. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Na.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init) ,'pmin':0.0 ,'pmax':max_flux})
    # Na. H-beta FWHM tied to [OIII]5007 FWHM
    # [13] Na. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Na.\;Hb}}$','init':0.,'pmin':-1000 ,'pmax':1000.})
    # [14] - Br. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Br.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init)/2.0  ,'pmin':0.0,'pmax':max_flux})
    # [15] - Br. H-beta FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$','init':2500.,'pmin':0.0,'pmax':10000.})
    # [16] - Br. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Br.\;Hb}}$','init':0.,'pmin':-1000. ,'pmax':1000.})

    param_names = ['vel','vel_disp',\
                'cont_slope','cont_amp',\
                'na_feii_amp',\
                'br_feii_amp',\
                'oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',\
                'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff',\
                'na_Hb_amp','na_Hb_voff',\
                'br_Hb_amp','br_Hb_fwhm','br_Hb_voff']

    param_labels,params,param_limits,param_init = mcmc_structures(mcmc_input)

    ################################################################################

    npix = galaxy.shape[0] # number of output pixels
    ntemp = np.shape(templates)[1]# number of templates
    
    # Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
    temp_fft,npad = template_rfft(templates) # we will use this throughout the code
    
    
    ################################################################################    
    
    return param_names,param_labels,params,param_limits,npix,ntemp,temp_fft,npad

def initialize_mcmc_init_no_outflows(lam_gal,galaxy,velscale):
    ################################################################################
    # Initial conditions for some parameters
    max_flux = np.max(galaxy)
    total_flux_init = np.median(galaxy[(lam_gal>5025.) & (lam_gal<5800.)])
    # cont_flux_init = 0.5*(np.median(galaxy))
    feii_flux_init= (0.1*np.median(galaxy))
    hb_amp_init= (np.max(galaxy[(lam_gal>4862-50) & (lam_gal<4862+50)]))
    oiii5007_amp_init = (np.max(galaxy[(lam_gal>5007-50) & (lam_gal<5007+50)]))

    # print('Total flux level: %0.2e' % total_flux_init)
    # print('Continuum flux level: %0.2e' % cont_flux_init )
    # print('FeII flux level: %0.2e' % feii_flux_init)
    
    ################################################################################
    
    mcmc_input = [] # array of parameter dicts
    # [0] - Galaxy template amplitude
    mcmc_input.append({'name':'$A_\mathrm{gal}$','init':(0.99*total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [1] - Narrow FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Na\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [2] - Broad FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Br\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [3] - Na. [OIII]5007 Core Amplitude
    mcmc_input.append({'name':'$A_\mathrm{[OIII]5007\;Core}$' ,'init':(oiii5007_amp_init-total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [4] - Na. [OIII]5007 Core FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Core}$','init':250.,'pmin':0.0,'pmax':1000.})
    # [5] - Na. [OIII]5007 Core VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Core}$','init':0.,'pmin':-1000.,'pmax':1000.})
    # [6] - Na. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Na.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init) ,'pmin':0.0 ,'pmax':max_flux})
    # Na. H-beta FWHM tied to [OIII]5007 FWHM
    # [7] Na. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Na.\;Hb}}$','init':0.,'pmin':-1000 ,'pmax':1000.})
    # [8] - Br. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Br.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init)/2.0  ,'pmin':0.0,'pmax':max_flux})
    # [9] - Br. H-beta FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$','init':2500.,'pmin':0.0,'pmax':10000.})
    # [10] - Br. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Br.\;Hb}}$','init':0.,'pmin':-1000. ,'pmax':1000.})

    param_labels,params,param_limits,param_init = mcmc_structures(mcmc_input)
    
    return param_labels,params,param_limits,param_init
    
    ################################################################################    


def initialize_mcmc_final_no_outflows(lam_gal,galaxy,templates,velscale):
    ################################################################################
    # Initial conditions for some parameters
    max_flux = np.max(galaxy)
    total_flux_init = np.median(galaxy[(lam_gal>5025.) & (lam_gal<5800.)])
    # cont_flux_init = (0.5*np.median(galaxy))
    feii_flux_init= (0.1*np.median(galaxy))
    hb_amp_init= (np.max(galaxy[(lam_gal>4862-50) & (lam_gal<4862+50)]))
    oiii5007_amp_init = (np.max(galaxy[(lam_gal>5007-50) & (lam_gal<5007+50)]))

    # print('Total flux level: %0.2e' % total_flux_init)
    # print('Continuum flux level: %0.2e' % cont_flux_init )
    # print('FeII flux level: %0.2e' % feii_flux_init)
    
    ################################################################################
    
    mcmc_input = [] # array of parameter dicts
    # [0] - Stellar velocity
    mcmc_input.append({'name':'$V_*$','init':100. ,'pmin':-500. ,'pmax':500.})
    # [1] - Stellar velocity dispersion
    mcmc_input.append({'name':'$\sigma_*$','init':200.0,'pmin':30.0,'pmax':400.})
    # [2] - AGN simple power-law slope
    mcmc_input.append({'name':'$m_\mathrm{cont}$','init':0.0  ,'pmin':-4.0,'pmax':2.0})
    # [3] - AGN simple power-law amplitude
    mcmc_input.append({'name':'$A_\mathrm{cont}$','init':(0.01*total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [4] - Narrow FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Na\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [5] - Broad FeII amplitude
    mcmc_input.append({'name':'$A_\mathrm{Br\;FeII}$','init':feii_flux_init,'pmin':0.0,'pmax':total_flux_init})
    # [6] - Na. [OIII]5007 Core Amplitude
    mcmc_input.append({'name':'$A_\mathrm{[OIII]5007\;Core}$' ,'init':(oiii5007_amp_init-total_flux_init),'pmin':0.0,'pmax':max_flux})
    # [7] - Na. [OIII]5007 Core FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_\mathrm{[OIII]5007\;Core}$','init':250.,'pmin':0.0,'pmax':1000.})
    # [8] - Na. [OIII]5007 Core VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_\mathrm{[OIII]5007\;Core}$','init':0.,'pmin':-1000.,'pmax':1000.})
    # [9] - Na. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Na.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init) ,'pmin':0.0 ,'pmax':max_flux})
    # Na. H-beta FWHM tied to [OIII]5007 FWHM
    # [10] Na. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Na.\;Hb}}$','init':0.,'pmin':-1000 ,'pmax':1000.})
    # [11] - Br. H-beta amplitude
    mcmc_input.append({'name':'$A_{\mathrm{Br.\;Hb}}$' ,'init':(hb_amp_init-total_flux_init)/2.0  ,'pmin':0.0,'pmax':max_flux})
    # [12] - Br. H-beta FWHM
    mcmc_input.append({'name':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$','init':2500.,'pmin':0.0,'pmax':10000.})
    # [13] - Br. H-beta VOFF
    mcmc_input.append({'name':'$\mathrm{VOFF}_{\mathrm{Br.\;Hb}}$','init':0.,'pmin':-1000. ,'pmax':1000.})

    param_names = ['vel','vel_disp',\
                'cont_slope','cont_amp',\
                'na_feii_amp',\
                'br_feii_amp',\
                'oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',\
                'na_Hb_amp','na_Hb_voff',\
                'br_Hb_amp','br_Hb_fwhm','br_Hb_voff']

    param_labels,params,param_limits,param_init = mcmc_structures(mcmc_input)

    ################################################################################

    npix = galaxy.shape[0] # number of output pixels
    ntemp = np.shape(templates)[1]# number of templates
    
    # Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
    temp_fft,npad = template_rfft(templates) # we will use this throughout the code
    
    
    ################################################################################    
    
    return param_names,param_labels,params,param_limits,npix,ntemp,temp_fft,npad


##################################################################################

##################################################################################

def mcmc_structures(mcmc_input):
    """
    Returns required array structures for MCMC algorithm.
    
    Parameters
    ----------
    params : array_like
        An array containing each paramter dict.  Each 
        parameter array is of the format
        
        param[i] = {name:,init:,pmin:,pmax:}
        
        where
        
        name:string
            name of the parameter
        init:float
            initial value of parameter
        pmin:float
            minimum value of parameter space
        pmax:float
            maximum value of parameter space
    
    Returns
    -------
    param_names:array
        the names of params
    params:array
        initial params which will be iteratively
        updated.
    param_limits:array
        An array containing the min and max limits of each parameter to serve as a prior.
    param_chain
        An array containing the chains for each parameter.
    """

    param_labels = []
    params      = []
    param_min   = []
    param_max   = []
#     param_limits = []
    param_init = []
    for i in range(0,len(mcmc_input),1):
        param_labels.append(mcmc_input[i]['name'])
        params.append(mcmc_input[i]['init'])
        param_min.append(mcmc_input[i]['pmin'])
        param_max.append(mcmc_input[i]['pmax'])
        param_init.append(mcmc_input[i]['init'])
    
    param_limits = np.array([param_min,param_max])
        
    return np.array(param_labels),np.array(params),np.array(param_limits),np.array(param_init)

##################################################################################

def simple_power_law(x,alpha,amp):
    xb = np.max(x)-((np.max(x)-np.min(x))/2.0) # take to be half of the wavelength range
    C = amp*(x/xb)**alpha # un-normalized
    return C

##################################################################################

def gaussian(x,center,amp,fwhm,voff,velscale):
    """
    Produces a gaussian vector the length of
    x with the specified parameters.
    
    Parameters
    ----------
    x : array_like
        the wavelength vector in angstroms.
    center: float
        the mean or center wavelength of the gaussian in angstroms.
    sigma: float
        the standard deviation of the gaussian in km/s.
    amp : float
        the amplitude of the gaussian in flux units.
    voff: the velocity offset (in km/s) from the rest-frame 
          line-center (taken from SDSS rest-frame emission
          line wavelengths)
    velscale: velocity scale; km/s/pixel
    Returns
    -------
    g: array3
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
    
#     print center_pix,sigma_pix,voff_pix

    g = amp*np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2)
    
    # Make sure edges of gaussian are zero 
    if (g[0]>1.0e-6) or g[-1]>1.0e-6:
        g = np.zeros(len(g))
    
    return g

##################################################################################

def initialize_feii(lam_gal,velscale):
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
    na_feii_template = feii_template(lam_gal,na_feii_rel_int,na_feii_center,na_feii_amp,na_feii_fwhm,feii_voff,velscale)
    br_feii_template = feii_template(lam_gal,br_feii_rel_int,br_feii_center,br_feii_amp,br_feii_fwhm,feii_voff,velscale)
    
    return na_feii_template, br_feii_template

def feii_template(lam,rel_int,center,amp,sigma,voff,velscale):

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
        line = amp*(gaussian(lam,center[i],rel_int[i],sigma,voff,velscale))
        template.append(line)
    template = np.sum(template,axis=0)
    
    return template

##################################################################################

# This function is used if we use the Maraston et al. 2009 SDSS composite templates but the absorption
# features are not deep enough to describe NLS1s.
# def galaxy_template(lam,age=5):
#     if (age<1) or (age>15):
#         print(' You must choose an age between (1 Gyr <= age <= 15 Gyr)!')
#         sys.exit()
#     else:
#         # We use the SDSS template from Maraston et al. 2009 
#         # The template is used as a placeholder for the stellar continuum
#         # in the initial parameter fit.
#         # Open the file and get the appropriate age galaxy (default age=5 Gyr)
#         df = pd.read_csv('badass_data_files/M09_composite_bestfitLRG.csv',skiprows=5,sep=',',names=['t_Gyr','AA','flam'])
#         wave =  np.array(df.loc[(df['t_Gyr']==age),'AA'])
#         flux =  np.array(df.loc[(df['t_Gyr']==age),'flam'])
#         # Interpolate the template 
#         gal_interp = interp1d(wave,flux,kind='cubic',bounds_error=False,fill_value=(0,0))
#         gal_temp = gal_interp(lam)
#         # Normalize by median
#         gal_temp = gal_temp/np.median(gal_temp)
#         return gal_temp

def galaxy_template(lam):
    # We use the SDSS template from Maraston et al. 2009 
    # The template is used as a placeholder for the stellar continuum
    # in the initial parameter fit.
    # Open the file and get the appropriate age galaxy (default age=5 Gyr)
    df = pd.read_csv('badass_data_files/nls1_template.csv',skiprows=1,sep=',',names=['wave','flux'])

    if 1: sys.exit
    wave =  np.array(df['wave'])
    flux =  np.array(df['flux'])
    # Interpolate the template 
    gal_interp = interp1d(wave,flux,kind='cubic',bounds_error=False,fill_value=(flux[0],flux[-1]))
    gal_temp = gal_interp(lam)
    # Normalize by median
    gal_temp = gal_temp/np.median(gal_temp)
    return gal_temp

##################################################################################

def init_outflow_model(pars,lam,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,run_dir,output_model=False):    
    """
    Used for estimating initial paramters using maximum likelihood estimation.
    - omits the stellar continuum fit for simplicity
    """

    def find_nearest_gal_model(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx

    # Free-parameters (14 total):
    #------------------------------------------
    # [0] - Galaxy template amplitude
    # [1] - Narrow FeII amplitude
    # [2] - Broad FeII amplitude
    # [3] - Na. [OIII]5007 Core Amplitude
    # [4] - Na. [OIII]5007 Core FWHM
    # [5] - Na. [OIII]5007 Core VOFF
    # [6] - Br. [OIII]5007 Outflow amplitude
    # [7] - Br. [OIII]5007 Outflow FWHM
    # [8] - Br. [OIII]5007 Outflow VOFF
    #      - Br. [OIII]4959 Outflow and Na. H-beta outflows are tied to [OIII]5007 outflow
    # [9] - Na. H-beta amplitude
    #      - Na. H-beta FWHM tied to [OIII]5007
    # [10] - Na. H-beta VOFF
    # [11] - Br. H-beta amplitude
    # [12] - Br. H-beta FWHM
    # [13] - Br. H-beta VOFF

    ############################# Host-galaxy Component ######################################################
    gal_temp = pars[0]*gal_temp
    host_model = (galaxy) - (gal_temp) # Subtract off continuum from galaxy, since we only want template weights to be fit
    ########################################################################################################    
    
    ############################# Fe II Component ##########################################################
    # Create template model for narrow and broad FeII emission 
    # idx 4 = na_feii_amp, idx 5 = br_feii_amp
    na_feii_amp  = pars[1]
    br_feii_amp  = pars[2]

    na_feii_template = na_feii_amp*na_feii_template
    br_feii_template = br_feii_amp*br_feii_template

    # If including FeII templates, initialize them here
    # na_feii_template,br_feii_template = initialize_feii(lam,velscale,pars[4],pars[5])

    host_model = (host_model) - (na_feii_template) - (br_feii_template)
    ########################################################################################################

    ############################# Emission Lines Component #################################################    
    # Create a template model for emission lines
    # Narrow [OIII]5007 Core; (6,7,8)=(voff,fwhm,amp)
    na_oiii5007_center = 5008.240 # Angstroms
    na_oiii5007_amp  = pars[3] # flux units
    na_oiii5007_res = fwhm_gal[find_nearest_gal_model(lam,na_oiii5007_center)[1]]*300000./na_oiii5007_center # instrumental fwhm resolution at this line
    na_oiii5007_fwhm = np.sqrt(pars[4]**2+(na_oiii5007_res)**2) # km/s
    na_oiii5007_voff = pars[5]  # km/s
    na_oiii5007 = gaussian(lam,na_oiii5007_center,na_oiii5007_amp,na_oiii5007_fwhm,na_oiii5007_voff,velscale)

    # Narrow [OIII]4959 Core; (6,7,9)=(voff,fwhm,amp)
    na_oiii4959_center = 4960.295 # Angstroms
    na_oiii4959_amp  = (1.0/3.0)*pars[3] # flux units
    na_oiii4959_fwhm = na_oiii5007_fwhm # km/s
    na_oiii4959_voff = na_oiii5007_voff  # km/s
    na_oiii4959 = gaussian(lam,na_oiii4959_center,na_oiii4959_amp,na_oiii4959_fwhm,na_oiii4959_voff,velscale)

    # Broad [OIII]5007 Outflow; (10,11,12)=(voff,fwhm,amp)
    br_oiii5007_center = 5008.240 # Angstroms
    br_oiii5007_amp  = pars[6] # flux units
    br_oiii5007_fwhm = np.sqrt(pars[7]**2+(na_oiii5007_res)**2) # km/s
    br_oiii5007_voff = pars[8]  # km/s
    br_oiii5007 = gaussian(lam,br_oiii5007_center,br_oiii5007_amp,br_oiii5007_fwhm,br_oiii5007_voff,velscale)

    # Broad [OIII]4959 Outflow; (10,11,?)=(voff,fwhm,amp)
    br_oiii4959_center = 4960.295 # Angstroms
    br_oiii4959_amp  = br_oiii5007_amp*na_oiii4959_amp/na_oiii5007_amp # flux units
    br_oiii4959_fwhm = br_oiii5007_fwhm # km/s
    br_oiii4959_voff = br_oiii5007_voff  # km/s
    if (br_oiii4959_amp!=br_oiii4959_amp/1.0) or (br_oiii4959_amp==np.inf): br_oiii4959_amp=0.0
    br_oiii4959 = gaussian(lam,br_oiii4959_center,br_oiii4959_amp,br_oiii4959_fwhm,br_oiii4959_voff,velscale)

    # Narrow H-beta; (13,,14)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    na_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    na_hb_amp  = pars[9] # flux units
    na_hb_fwhm = np.sqrt(na_oiii5007_fwhm**2+(na_hb_res)**2) # km/s
    na_hb_voff = pars[10]  # km/s
    na_Hb = gaussian(lam,na_hb_center,na_hb_amp,na_hb_fwhm,na_hb_voff,velscale)

    # Broad H-beta; (15,16,17)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    br_hb_amp  = pars[11] # flux units
    br_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    br_hb_fwhm = np.sqrt(pars[12]**2+(br_hb_res)**2) # km/s
    br_hb_voff = pars[13]  # km/s
    br_Hb = gaussian(lam,na_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)

    # Broad H-beta Outflow; only a model, no free parameters, tied to [OIII]5007
    br_hb_outflow_amp =  br_oiii5007_amp*na_hb_amp/na_oiii5007_amp
    br_hb_outflow_fwhm = np.sqrt(br_oiii5007_fwhm**2+(br_hb_res)**2) # km/s
    br_hb_outflow_voff = na_hb_voff+br_oiii5007_voff
    if (br_hb_outflow_amp!=br_hb_outflow_amp/1.0) or (br_hb_outflow_amp==np.inf): br_hb_outflow_amp=0.0
    br_Hb_outflow = gaussian(lam,na_hb_center,br_hb_outflow_amp,br_hb_outflow_fwhm,br_hb_outflow_voff,velscale)

    host_model = (host_model) - (br_Hb) - (na_Hb) - (br_Hb_outflow) - (na_oiii4959) - (br_oiii4959) - (na_oiii5007)  - (br_oiii5007)

    ########################################################################################################
    
    # The final model 
    gmodel = (gal_temp) + (na_feii_template) + (br_feii_template) + (br_Hb) + (na_Hb)\
     + (br_Hb_outflow) + (na_oiii4959) + (br_oiii4959) + (na_oiii5007)  + (br_oiii5007)

    if output_model==False:
        return gmodel
    elif output_model==True: # output all models
        allmodels = [gal_temp,na_feii_template,br_feii_template,\
                    br_Hb,na_Hb,na_oiii4959,na_oiii5007,br_oiii4959,br_oiii5007,br_Hb_outflow]

        best_model_outflows = (gmodel,allmodels)
        # Plot the model+galaxy
        fig1 = plt.figure(figsize=(14,6)) 
        ax1  = fig1.add_subplot(2,1,1)
        ax2  = fig1.add_subplot(2,1,2)
        ax1.plot(lam,galaxy,linewidth=1.0,color='black',label='Galaxy')
        ax1.plot(lam,best_model_outflows[0],linewidth=1.0,color='red',label='Outflow Model')
        ax1.plot(lam,best_model_outflows[1][0],linewidth=1.0,color='limegreen',label='Host-galaxy')
        ax1.plot(lam,best_model_outflows[1][1],linewidth=1.0,color='orange',label='Na. FeII')
        ax1.plot(lam,best_model_outflows[1][2],linewidth=1.0,color='darkorange',label='Br. FeII')
        ax1.plot(lam,best_model_outflows[1][3],linewidth=1.0,color='blue',label='Broad')
        ax1.plot(lam,best_model_outflows[1][4],linewidth=1.0,color='dodgerblue',label='Narrow')
        ax1.plot(lam,best_model_outflows[1][5],linewidth=1.0,color='dodgerblue',label='')
        ax1.plot(lam,best_model_outflows[1][6],linewidth=1.0,color='dodgerblue')
        ax1.plot(lam,best_model_outflows[1][7],linewidth=1.0,color='orangered',label='Outflow')
        ax1.plot(lam,best_model_outflows[1][8],linewidth=1.0,color='orangered')
        ax1.plot(lam,best_model_outflows[1][9],linewidth=1.0,color='orangered')
        # Perform robust sigma clipping to get a good noise value
        resid_outflow = (galaxy-best_model_outflows[0])
        # sig_clip_resid_outflow = sigma_clip(resid_outflow,sigma=3)
        # outflow_std = round(mad_std(sig_clip_resid_outflow),2)
        outflow_std = round(mad_std(resid_outflow),2)
        # ax2.plot(lam,sig_clip_resid_outflow,linewidth=1.0,color='black',label=r'Residuals, $\sigma=%0.3f$' % outflow_std)
        ax2.plot(lam,resid_outflow,linewidth=1.0,color='black',label=r'Residuals, $\sigma=%0.3f$' % outflow_std)
        ax1.set_ylim(0.0,np.max(np.max([galaxy,best_model_outflows[0]])+(3*outflow_std)))
        ax2.set_ylim(np.min(resid_outflow)-(3*outflow_std),np.max(np.max([galaxy,best_model_outflows[0]])+(3*outflow_std)))
        ax2.axhline(0.0,color='black',linestyle='--')
        ax2.axhline(outflow_std,color='red',linestyle='--',linewidth=0.5)
        ax2.axhline(-outflow_std,color='red',linestyle='--',linewidth=0.5)
        ax1.set_xlim(np.min(lam),np.max(lam))
        ax2.set_xlim(np.min(lam),np.max(lam))
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        plt.savefig(run_dir+'outflow_model.pdf',dpi=300,fmt='pdf')

        return outflow_std#, best_model_outflows[0]

# def outflow_monte_carlo(outflow_std, best_model,best_pars,params,param_limits,args,niter=1000):
#     # args = (lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir)
#     lam_gal,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,noise,run_dir = args
#     mask = np.where((lam_gal > 4700.) & (lam_gal < 5100.))
#     result = np.zeros((niter,len(best_pars))) # This will store the results
#     # print(' Running MC iterations...')
#     for j in range(niter):
#         print(' Running MC iteration %d of %d.' % (j+1,niter))
#         galaxy = best_model[mask]
#         n = np.random.normal(loc=0,scale=1.0*outflow_std,size=np.shape(galaxy))
#         new_galaxy = np.array(galaxy+n)
#         args = (lam_gal[mask],fwhm_gal[mask],na_feii_template[mask],br_feii_template[mask],gal_temp[mask],velscale,new_galaxy[mask],noise[mask],run_dir)
#         result_outflows = max_likelihood_outflows(best_pars,param_limits,args)
#         print result_outflows['x']

    
#     return galaxy,n
        # result[j,:] = pp.sol


def final_outflow_model(pars,lam,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,vsyst,galaxy,temp_list,run_dir,move_temp=False,output_model=False): 	
    """
    Constructs galaxy model by convolving templates with a LOSVD given by 
    a specified set of velocity parameters. 
    
    Parameters:
        pars: parameters of Markov-chain
        lam: wavelength vector used for continuum model
        temp_fft: the Fourier-transformed templates
        npad: 
        velscale: the velocity scale in km/s/pixel
        npix: number of output pixels; must be same as galaxy
        vsyst: dv; the systematic velocity fr
    """

    def find_nearest_gal_model(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx

    # Free-parameters (17 total):
    #------------------------------------------
    # [0] - Stellar velocity
    # [1] - Stellar velocity dispersion
    # [2] - AGN simple power-law slope
    # [3] - AGN simple power-law amplitude
    # [4] - Narrow FeII amplitude
    # [5] - Broad FeII amplitude
    # [6] - Na. [OIII]5007 Core Amplitude
    # [7] - Na. [OIII]5007 Core FWHM
    # [8] - Na. [OIII]5007 Core VOFF
    # [9] - Br. [OIII]5007 Outflow amplitude
    # [10] - Br. [OIII]5007 Outflow FWHM
    # [11] - Br. [OIII]5007 Outflow VOFF
    #      - Br. [OIII]4959 Outflow and Na. H-beta outflows are tied to [OIII]5007 outflow
    # [12] - Na. H-beta amplitude
    #      - Na. H-beta FWHM tied to [OIII]5007
    # [13] - Na. H-beta VOFF
    # [14] - Br. H-beta amplitude
    # [15] - Br. H-beta FWHM
    # [16] - Br. H-beta VOFF

    ############################# Power-law Component ######################################################
    # Create a template model for the power-law continuum
    cont = simple_power_law(lam,pars[2],pars[3]) # ind 2 = alpha, ind 3 = amplitude
    host_model = galaxy - cont # Subtract off continuum from galaxy, since we only want template weights to be fit
    ########################################################################################################
    
    ############################# Fe II Component ##########################################################
    # Create template model for narrow and broad FeII emission 
    # idx 4 = na_feii_amp, idx 5 = br_feii_amp
    na_feii_amp  = pars[4]
    br_feii_amp  = pars[5]

    na_feii_template = na_feii_amp*na_feii_template
    br_feii_template = br_feii_amp*br_feii_template

    # If including FeII templates, initialize them here
    # na_feii_template,br_feii_template = initialize_feii(lam,velscale,pars[4],pars[5])

    host_model = host_model - (na_feii_template) - (br_feii_template)
    ########################################################################################################

    ############################# Emission Lines Component #################################################    
    # Create a template model for emission lines
    # Narrow [OIII]5007 Core; (6,7,8)=(voff,fwhm,amp)
    na_oiii5007_center = 5008.240 # Angstroms
    na_oiii5007_amp  = pars[6] # flux units
    na_oiii5007_res = fwhm_gal[find_nearest_gal_model(lam,na_oiii5007_center)[1]]*300000./na_oiii5007_center # instrumental fwhm resolution at this line
    na_oiii5007_fwhm = np.sqrt(pars[7]**2+(na_oiii5007_res)**2) # km/s
    na_oiii5007_voff = pars[8]  # km/s
    na_oiii5007 = gaussian(lam,na_oiii5007_center,na_oiii5007_amp,na_oiii5007_fwhm,na_oiii5007_voff,velscale)

    # Narrow [OIII]4959 Core; (6,7,9)=(voff,fwhm,amp)
    na_oiii4959_center = 4960.295 # Angstroms
    na_oiii4959_amp  = (1.0/3.0)*pars[6] # flux units
    na_oiii4959_fwhm = na_oiii5007_fwhm # km/s
    na_oiii4959_voff = na_oiii5007_voff  # km/s
    na_oiii4959 = gaussian(lam,na_oiii4959_center,na_oiii4959_amp,na_oiii4959_fwhm,na_oiii4959_voff,velscale)

    # Broad [OIII]5007 Outflow; (10,11,12)=(voff,fwhm,amp)
    br_oiii5007_center = 5008.240 # Angstroms
    br_oiii5007_amp  = pars[9] # flux units
    br_oiii5007_fwhm = np.sqrt(pars[10]**2+(na_oiii5007_res)**2) # km/s
    br_oiii5007_voff = pars[11]  # km/s
    br_oiii5007 = gaussian(lam,br_oiii5007_center,br_oiii5007_amp,br_oiii5007_fwhm,br_oiii5007_voff,velscale)

    # Broad [OIII]4959 Outflow; (10,11,?)=(voff,fwhm,amp)
    br_oiii4959_center = 4960.295 # Angstroms
    br_oiii4959_amp  = br_oiii5007_amp*na_oiii4959_amp/na_oiii5007_amp # flux units
    br_oiii4959_fwhm = br_oiii5007_fwhm # km/s
    br_oiii4959_voff = br_oiii5007_voff  # km/s
    if (br_oiii4959_amp!=br_oiii4959_amp/1.0) or (br_oiii4959_amp==np.inf): br_oiii4959_amp=0.0
    br_oiii4959 = gaussian(lam,br_oiii4959_center,br_oiii4959_amp,br_oiii4959_fwhm,br_oiii4959_voff,velscale)

    # Narrow H-beta; (13,,14)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    na_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    na_hb_amp  = pars[12] # flux units
    na_hb_fwhm = np.sqrt(na_oiii5007_fwhm**2+(na_hb_res)**2) # km/s
    na_hb_voff = pars[13]  # km/s
    na_Hb = gaussian(lam,na_hb_center,na_hb_amp,na_hb_fwhm,na_hb_voff,velscale)

    # Broad H-beta; (15,16,17)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    br_hb_amp  = pars[14] # flux units
    br_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    br_hb_fwhm = np.sqrt(pars[15]**2+(br_hb_res)**2) # km/s
    br_hb_voff = pars[16]  # km/s
    br_Hb = gaussian(lam,na_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)

    # Broad H-beta Outflow; only a model, no free parameters, tied to [OIII]5007
    br_hb_outflow_amp =  br_oiii5007_amp*na_hb_amp/na_oiii5007_amp
    br_hb_outflow_fwhm = np.sqrt(br_oiii5007_fwhm**2+(br_hb_res)**2) # km/s
    br_hb_outflow_voff = na_hb_voff+br_oiii5007_voff
    if (br_hb_outflow_amp!=br_hb_outflow_amp/1.0) or (br_hb_outflow_amp==np.inf): br_hb_outflow_amp=0.0
    br_Hb_outflow = gaussian(lam,na_hb_center,br_hb_outflow_amp,br_hb_outflow_fwhm,br_hb_outflow_voff,velscale)

    host_model = (host_model) - (br_Hb) - (na_Hb) - (br_Hb_outflow) - (na_oiii4959) - (br_oiii4959) - (na_oiii5007)  - (br_oiii5007)

    ########################################################################################################
    
    ############################# Host-galaxy Component ####################################################    
    # Convolve the templates with a LOSVD
    losvd_params = [pars[0],pars[1]] # ind 0 = velocity*, ind 1 = sigma*
    conv_temp = convolve_gauss_hermite(temp_fft,npad,float(velscale),\
                losvd_params,npix,velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
    
    # Fitted weights of all templates using Non-negative Least Squares (NNLS)
#     host_model = galaxy
    weights = nnls(conv_temp,host_model)

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
                print(' Moving template %s' % temp_name)
                shutil.copyfile(temp_list[i],run_dir+'templates/'+temp_name)


    # if 1: sys.exit()

    ########################################################################################################
    
    # The final model is sum(stellar templates*weights) + continuum model
    gmodel = (np.sum(weights*conv_temp,axis=1)) + cont + (na_feii_template) + (br_feii_template) + (br_Hb) + (na_Hb)\
     + (br_Hb_outflow) + (na_oiii4959) + (br_oiii4959) + (na_oiii5007)  + (br_oiii5007)

    ########################## Measure Emission Line Fluxes #################################################

    na_feii_flux       = simps(na_feii_template,lam)
    br_feii_flux       = simps(br_feii_template,lam)
    br_Hb_flux         = simps(br_Hb,lam)
    na_Hb_flux         = simps(na_Hb,lam)
    br_Hb_outflow_flux = simps(br_Hb_outflow,lam)
    na_oiii4959_flux   = simps(na_oiii4959,lam)
    br_oiii4959_flux   = simps(br_oiii4959,lam)
    na_oiii5007_flux   = simps(na_oiii5007,lam)
    br_oiii5007_flux   = simps(br_oiii5007,lam)
    # Put em_line_fluxes into dataframe
    em_line_fluxes = {'na_feii_flux':[na_feii_flux],'br_feii_flux':[br_feii_flux],'br_Hb_flux':[br_Hb_flux],'na_Hb_flux':[na_Hb_flux],\
    'br_Hb_outflow_flux':[br_Hb_outflow_flux],'na_oiii4959_flux':[na_oiii4959_flux],'br_oiii4959_flux':[br_oiii4959_flux],\
    'na_oiii5007_flux':[na_oiii5007_flux],'br_oiii5007_flux':[br_oiii5007_flux]}
    em_line_cols = ['na_feii_flux','br_feii_flux','br_Hb_flux','na_Hb_flux','br_Hb_outflow_flux','na_oiii4959_flux','br_oiii4959_flux','na_oiii5007_flux','br_oiii5007_flux']
    df_fluxes = pd.DataFrame(data=em_line_fluxes,columns=em_line_cols)

    # Write to csv
    # Create a folder in the working directory to save plots and data to
    if os.path.exists(run_dir+'em_line_fluxes.csv')==False:
    #     # If file has not been created, create it 
        # print(' File does not exist!')
        df_fluxes.to_csv(run_dir+'em_line_fluxes.csv',sep=',')
    if os.path.exists(run_dir+'em_line_fluxes.csv')==True:
    #     # If file has been created, append df_fluxes to existing file
        # print( ' File exists!')
        with open(run_dir+'em_line_fluxes.csv', 'a') as f:
            df_fluxes.to_csv(f, header=False)


    # if 1: sys.exit()
    ########################################################################################################

    if output_model==False:
    	return gmodel
    elif output_model==True: # output all models
    	allmodels = [np.sum(weights*conv_temp,axis=1),cont,na_feii_template,br_feii_template,\
                    br_Hb,na_Hb,na_oiii4959,na_oiii5007,br_oiii4959,br_oiii5007,br_Hb_outflow]
    	return gmodel,allmodels,weights

##################################################################################

def init_no_outflow_model(pars,lam,fwhm_gal,na_feii_template,br_feii_template,gal_temp,velscale,galaxy,run_dir,output_model=False):    
    """
    Used for estimating initial paramters using maximum likelihood estimation.
    - omits the stellar continuum fit for simplicity
    """

    def find_nearest_gal_model(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx

    # Free-parameters (13 total):
    #------------------------------------------
    # [0] - Galaxy template amplitude    
    # [1] - Narrow FeII amplitude
    # [2] - Broad FeII amplitude
    # [3] - Na. [OIII]5007 Core Amplitude
    # [4] - Na. [OIII]5007 Core FWHM
    # [5] - Na. [OIII]5007 Core VOFF
    # [6] - Na. H-beta amplitude
    #      - Na. H-beta FWHM tied to [OIII]5007
    # [7] - Na. H-beta VOFF
    # [8] - Br. H-beta amplitude
    # [9] - Br. H-beta FWHM
    # [10] - Br. H-beta VOFF

    ############################# Host-galaxy Component ######################################################
    gal_temp = pars[0]*gal_temp
    host_model = (galaxy) - (gal_temp) # Subtract off continuum from galaxy, since we only want template weights to be fit
    ########################################################################################################   
    
    ############################# Fe II Component ##########################################################
    # Create template model for narrow and broad FeII emission 
    # idx 4 = na_feii_amp, idx 5 = br_feii_amp
    na_feii_amp  = pars[1]
    br_feii_amp  = pars[2]

    na_feii_template = na_feii_amp*na_feii_template
    br_feii_template = br_feii_amp*br_feii_template

    # If including FeII templates, initialize them here
    # na_feii_template,br_feii_template = initialize_feii(lam,velscale,pars[4],pars[5])

    host_model = (host_model) - (na_feii_template) - (br_feii_template)
    ########################################################################################################

    ############################# Emission Lines Component #################################################    
    # Create a template model for emission lines
    # Narrow [OIII]5007 Core; (6,7,8)=(voff,fwhm,amp)
    na_oiii5007_center = 5008.240 # Angstroms
    na_oiii5007_amp  = pars[3] # flux units
    na_oiii5007_res = fwhm_gal[find_nearest_gal_model(lam,na_oiii5007_center)[1]]*300000./na_oiii5007_center # instrumental fwhm resolution at this line
    na_oiii5007_fwhm = np.sqrt(pars[4]**2+(na_oiii5007_res)**2) # km/s
    na_oiii5007_voff = pars[5]  # km/s
    na_oiii5007 = gaussian(lam,na_oiii5007_center,na_oiii5007_amp,na_oiii5007_fwhm,na_oiii5007_voff,velscale)

    # Narrow [OIII]4959 Core; (6,7,9)=(voff,fwhm,amp)
    na_oiii4959_center = 4960.295 # Angstroms
    na_oiii4959_amp  = (1.0/3.0)*pars[3] # flux units
    na_oiii4959_fwhm = na_oiii5007_fwhm # km/s
    na_oiii4959_voff = na_oiii5007_voff  # km/s
    na_oiii4959 = gaussian(lam,na_oiii4959_center,na_oiii4959_amp,na_oiii4959_fwhm,na_oiii4959_voff,velscale)

    # Narrow H-beta; (13,,14)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    na_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    na_hb_amp  = pars[6] # flux units
    na_hb_fwhm = np.sqrt(na_oiii5007_fwhm**2+(na_hb_res)**2) # km/s
    na_hb_voff = pars[7]  # km/s
    na_Hb = gaussian(lam,na_hb_center,na_hb_amp,na_hb_fwhm,na_hb_voff,velscale)

    # Broad H-beta; (15,16,17)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    br_hb_amp  = pars[8] # flux units
    br_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    br_hb_fwhm = np.sqrt(pars[9]**2+(br_hb_res)**2) # km/s
    br_hb_voff = pars[10]  # km/s
    br_Hb = gaussian(lam,na_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)

    host_model = (host_model) - (br_Hb) - (na_Hb) - (na_oiii4959) - (na_oiii5007) 

    ########################################################################################################
    
    # The final model 
    gmodel = (gal_temp) + (na_feii_template) + (br_feii_template) + (br_Hb) + (na_Hb)\
     + (na_oiii4959) + (na_oiii5007) 

    if output_model==False:
        return gmodel
    elif output_model==True: # output all models
        allmodels = [gal_temp,na_feii_template,br_feii_template,\
                    br_Hb,na_Hb,na_oiii4959,na_oiii5007]

        best_model_no_outflows = (gmodel,allmodels)
        # Plot the model+galaxy
        fig1 = plt.figure(figsize=(14,6)) 
        ax1  = plt.subplot(2,1,1)
        ax2  = plt.subplot(2,1,2)
        ax1.plot(lam,galaxy,linewidth=1.0,color='black',label='Galaxy')
        ax1.plot(lam,best_model_no_outflows[0],linewidth=1.0,color='red',label='NO Outflow Model')
        ax1.plot(lam,best_model_no_outflows[1][0],linewidth=1.0,color='limegreen',label='Host-galaxy')
        ax1.plot(lam,best_model_no_outflows[1][1],linewidth=1.0,color='orange',label='Na. FeII')
        ax1.plot(lam,best_model_no_outflows[1][2],linewidth=1.0,color='darkorange',label='Br. FeII')
        ax1.plot(lam,best_model_no_outflows[1][3],linewidth=1.0,color='blue',label='Broad')
        ax1.plot(lam,best_model_no_outflows[1][4],linewidth=1.0,color='dodgerblue',label='Narrow')
        ax1.plot(lam,best_model_no_outflows[1][5],linewidth=1.0,color='dodgerblue',label='')
        ax1.plot(lam,best_model_no_outflows[1][6],linewidth=1.0,color='dodgerblue')
        # Perform robust sigma clipping to get a good noise value
        resid_no_outflow = (galaxy-best_model_no_outflows[0])
        # sig_clip_resid_no_outflow = sigma_clip(resid_no_outflow,sigma=3)
        # no_outflow_std = round(mad_std(sig_clip_resid_no_outflow),2)
        no_outflow_std = round(mad_std(resid_no_outflow),2)
        # ax2.plot(lam,sig_clip_resid_no_outflow,linewidth=1.0,color='black',label=r'Residuals, $\sigma=%0.3f$' % no_outflow_std)
        ax2.plot(lam,resid_no_outflow,linewidth=1.0,color='black',label=r'Residuals, $\sigma=%0.3f$' % no_outflow_std)
        ax1.set_ylim(0.0,np.max(np.max([galaxy,best_model_no_outflows[0]])+(3*no_outflow_std)))
        ax2.set_ylim(np.min(resid_no_outflow)-(3*no_outflow_std),np.max(np.max([galaxy,best_model_no_outflows[0]])+(3*no_outflow_std)))
        ax2.axhline(0.0,color='black',linestyle='--')
        ax2.axhline(no_outflow_std,color='red',linestyle='--',linewidth=0.5)
        ax2.axhline(-no_outflow_std,color='red',linestyle='--',linewidth=0.5)
        ax1.set_xlim(np.min(lam),np.max(lam))
        ax2.set_xlim(np.min(lam),np.max(lam))
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        plt.savefig(run_dir+'no_outflow_model.pdf',dpi=300,fmt='pdf')

        return no_outflow_std

def final_no_outflow_model(pars,lam,fwhm_gal,na_feii_template,br_feii_template,temp_fft,npad,velscale,npix,vsyst,galaxy,temp_list,run_dir,move_temp=False,output_model=False):    
    """
    Constructs galaxy model by convolving templates with a LOSVD given by 
    a specified set of velocity parameters. 
    
    Parameters:
        pars: parameters of Markov-chain
        lam: wavelength vector used for continuum model
        temp_fft: the Fourier-transformed templates
        npad: 
        velscale: the velocity scale in km/s/pixel
        npix: number of output pixels; must be same as galaxy
        vsyst: dv; the systematic velocity fr
    """

    def find_nearest_gal_model(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx

    # Free-parameters (14 total):
    #------------------------------------------
    # [0] - Stellar velocity
    # [1] - Stellar velocity dispersion
    # [2] - AGN simple power-law slope
    # [3] - AGN simple power-law amplitude
    # [4] - Narrow FeII amplitude
    # [5] - Broad FeII amplitude
    # [6] - Na. [OIII]5007 Core Amplitude
    # [7] - Na. [OIII]5007 Core FWHM
    # [8] - Na. [OIII]5007 Core VOFF
    # [9] - Na. H-beta amplitude
    #      - Na. H-beta FWHM tied to [OIII]5007
    # [10] - Na. H-beta VOFF
    # [11] - Br. H-beta amplitude
    # [12] - Br. H-beta FWHM
    # [13] - Br. H-beta VOFF
    ############################# Power-law Component ######################################################
    # Create a template model for the power-law continuum
    cont = simple_power_law(lam,pars[2],pars[3]) # ind 2 = alpha, ind 3 = amplitude
    host_model = galaxy - cont # Subtract off continuum from galaxy, since we only want template weights to be fit
    ########################################################################################################
    
    ############################# Fe II Component ##########################################################
    # Create template model for narrow and broad FeII emission 
    # idx 4 = na_feii_amp, idx 5 = br_feii_amp
    na_feii_amp  = pars[4]
    br_feii_amp  = pars[5]

    na_feii_template = na_feii_amp*na_feii_template
    br_feii_template = br_feii_amp*br_feii_template

    # If including FeII templates, initialize them here
    # na_feii_template,br_feii_template = initialize_feii(lam,velscale,pars[4],pars[5])

    host_model = host_model - (na_feii_template) - (br_feii_template)
    ########################################################################################################

    ############################# Emission Lines Component #################################################    
    # Create a template model for emission lines
    # Narrow [OIII]5007 Core; (6,7,8)=(voff,fwhm,amp)
    na_oiii5007_center = 5008.240 # Angstroms
    na_oiii5007_amp  = pars[6] # flux units
    na_oiii5007_res = fwhm_gal[find_nearest_gal_model(lam,na_oiii5007_center)[1]]*300000./na_oiii5007_center # instrumental fwhm resolution at this line
    na_oiii5007_fwhm = np.sqrt(pars[7]**2+(na_oiii5007_res)**2) # km/s
    na_oiii5007_voff = pars[8]  # km/s
    na_oiii5007 = gaussian(lam,na_oiii5007_center,na_oiii5007_amp,na_oiii5007_fwhm,na_oiii5007_voff,velscale)

    # Narrow [OIII]4959 Core; (6,7,9)=(voff,fwhm,amp)
    na_oiii4959_center = 4960.295 # Angstroms
    na_oiii4959_amp  = (1.0/3.0)*pars[6] # flux units
    na_oiii4959_fwhm = na_oiii5007_fwhm # km/s
    na_oiii4959_voff = na_oiii5007_voff  # km/s
    na_oiii4959 = gaussian(lam,na_oiii4959_center,na_oiii4959_amp,na_oiii4959_fwhm,na_oiii4959_voff,velscale)

    # Narrow H-beta; (13,,14)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    na_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    na_hb_amp  = pars[9] # flux units
    na_hb_fwhm = np.sqrt(na_oiii5007_fwhm**2+(na_hb_res)**2) # km/s
    na_hb_voff = pars[10]  # km/s
    na_Hb = gaussian(lam,na_hb_center,na_hb_amp,na_hb_fwhm,na_hb_voff,velscale)

    # Broad H-beta; (15,16,17)=(voff,fwhm,amp)
    na_hb_center = 4862.68 # Angstroms
    br_hb_amp  = pars[11] # flux units
    br_hb_res = fwhm_gal[find_nearest_gal_model(lam,na_hb_center)[1]]*300000./na_hb_center # instrumental fwhm resolution at this line
    br_hb_fwhm = np.sqrt(pars[12]**2+(br_hb_res)**2) # km/s
    br_hb_voff = pars[13]  # km/s
    br_Hb = gaussian(lam,na_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)

    host_model = (host_model) - (br_Hb) - (na_Hb) - (na_oiii4959) - (na_oiii5007) 

    ########################################################################################################
    
    ############################# Host-galaxy Component ####################################################    
    # Convolve the templates with a LOSVD
    losvd_params = [pars[0],pars[1]] # ind 0 = velocity*, ind 1 = sigma*
    conv_temp = convolve_gauss_hermite(temp_fft,npad,float(velscale),\
                losvd_params,npix,velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
    
    # Fitted weights of all templates using Non-negative Least Squares (NNLS)
#     host_model = galaxy
    weights = nnls(conv_temp,host_model)

    if move_temp==True:
        # Templates that are used are placed in templates folder
        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

        a =  np.where(weights>0)[0]
        for i in a:
            s = temp_list[i]
            temp_name = path_leaf(temp_list[i])
            # print s,temp_name,work_dir+'templates/'+temp_name
            if os.path.exists(run_dir+'templates/'+temp_name)==False:
                # Check if template file was already copied to template folder
                # If not, copy it to templates folder for each object
                print(' Moving template %s' % temp_name)
                shutil.copyfile(temp_list[i],run_dir+'templates/'+temp_name)


    # if 1: sys.exit()

    ########################################################################################################
    
    # The final model is sum(stellar templates*weights) + continuum model
    gmodel = (np.sum(weights*conv_temp,axis=1)) + cont + (na_feii_template) + (br_feii_template) + (br_Hb) + (na_Hb)\
     + (na_oiii4959) + (na_oiii5007) 

    ########################## Measure Emission Line Fluxes #################################################

    na_feii_flux       = simps(na_feii_template,lam)
    br_feii_flux       = simps(br_feii_template,lam)
    br_Hb_flux         = simps(br_Hb,lam)
    na_Hb_flux         = simps(na_Hb,lam)
    na_oiii4959_flux   = simps(na_oiii4959,lam)
    na_oiii5007_flux   = simps(na_oiii5007,lam)
    # Put em_line_fluxes into dataframe
    em_line_fluxes = {'na_feii_flux':[na_feii_flux],'br_feii_flux':[br_feii_flux],'br_Hb_flux':[br_Hb_flux],'na_Hb_flux':[na_Hb_flux],\
    'na_oiii4959_flux':[na_oiii4959_flux],'na_oiii5007_flux':[na_oiii5007_flux]}
    em_line_cols = ['na_feii_flux','br_feii_flux','br_Hb_flux','na_Hb_flux','na_oiii4959_flux','na_oiii5007_flux']
    df_fluxes = pd.DataFrame(data=em_line_fluxes,columns=em_line_cols)

    # Write to csv
    # Create a folder in the working directory to save plots and data to
    if os.path.exists(run_dir+'em_line_fluxes.csv')==False:
    #     # If file has not been created, create it 
        # print(work_dir+'MCMC_output/em_line_fluxes.csv')
        # print(' File does not exist!')
        df_fluxes.to_csv(run_dir+'em_line_fluxes.csv',sep=',')
    if os.path.exists(run_dir+'em_line_fluxes.csv')==True:
    #     # If file has been created, append df_fluxes to existing file
        # print( ' File exists!')
        with open(run_dir+'em_line_fluxes.csv', 'a') as f:
            df_fluxes.to_csv(f, header=False)


    # if 1: sys.exit()
    ########################################################################################################

    if output_model==False:
        return gmodel
    elif output_model==True: # output all models
        allmodels = [np.sum(weights*conv_temp,axis=1),cont,na_feii_template,br_feii_template,\
                    br_Hb,na_Hb,na_oiii4959,na_oiii5007]
        return gmodel,allmodels,weights

##################################################################################

def likelihood(params,lam_gal,temp_fft,npad,velscale,npix,dv,galaxy,sigma):
    # Create model
    model = galaxy_model(params,lam_gal,temp_fft,npad,velscale,npix,dv,galaxy)
    # Calculate log-likelihood
    l = -0.5*(galaxy-model)**2/(sigma)**2#-0.5*( np.log(2*np.pi*sigma**2)+(galaxy-model)**2/(sigma)**2 )
    l = np.sum(l,axis=0)
    return l,model

##################################################################################

# Plotting Routines
##################################################################################

def grid(x, y, z, resX=25, resY=25):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    X, Y = meshgrid(xi, yi)
    return X, Y, zi

def conf_int(x,prob,factor): # Function for calculating an arbitrary confidence interval (%)
    
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
#                 print conf_interval_1
#             elif round(integral,2) == 0.95: #95.0/100.0:    
#                 # 2 sigma = 95% confidence interval
#                 conf_interval_2 = [pdfmax - np.min(xvec),np.max(xvec) - pdfmax]
#                 print conf_interval_2
#             elif round(integral,3) == 0.997: #99.7/100.0:    
#                 # 3 sigma = 99.7% confidence interval
#                 conf_interval_3 = [pdfmax - np.min(xvec),np.max(xvec) - pdfmax]
#                 print conf_interval_3
#                 break
                
        return pdfmax,conf_interval_1[0],conf_interval_1[1],xvec,yvec*scale_factor
#                 conf_interval_2[0],conf_interval_2[1],\
#                 conf_interval_3[0],conf_interval_3[1],

    except: 
        print("\n Error: Cannot determine confidence interval.")
        mode = x[np.where(prob==np.max(prob))]
        mean = simps(x*prob,x)
        std = np.sqrt(simps((x-mean)**2*prob,x))
        return mode[0],std,std,x,np.zeros(len(prob))

def determine_hist_bins(chains):
        """
        Determines the maxmimum number of bins for a histogram 
        that is unimodal.
        
        Parameters
        ----------
        chain : array_like
            MCMC chain of a parameter.  Recommended 
            minimum length of 10000.
    
        Returns
        -------
        nbins:int
            The maximum number of bins that still
            results in a unimodal distribution.
        """
        chain_bins = []
        for c in range(0,len(chains),1):
            max_bins = 200 # maximum number of bins to consider
            min_bins = 4 # minimum number of bins to consider
            for i in range(min_bins,max_bins,1):
                n,bins = np.histogram(chains[c],i)
                new_n = np.r_[True, n[1:] > n[:-1]] & np.r_[n[:-1] > n[1:], True]
                if len(new_n[new_n==True])>1 and i==min_bins:
                    # Really bad sampling of parameter space, default to 3 bins
                    nbins = min_bins
                elif len(new_n[new_n==True])==1 and i>min_bins:
                    # Increase bin size and check again
                    continue
                elif len(new_n[new_n==True])>1 and i>min_bins:
                    # Max.# of bins for global maximum reached, stop here
                    nbins = i-1
                    chain_bins.append(nbins)
                    break
        return chain_bins

def get_bin_centers(bins):
        bins = bins[:-1]
        bin_width = bins[1]-bins[0]
        new_bins =  bins + bin_width/2.0
        return new_bins

def remove_outliers(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



def param_plots(params,param_labels,burnin,sampler_chain,run_dir,model_type):
    # Store best-fit parameter values from histogram fitting
    par_best = [] # best fit parameter
    sig_low  = [] # lower-bound 1-sigma error in best-fit
    sig_upp  = [] # upper-bound 1-sigma error in best-fit
    if model_type=='outflow':
    	# Colors for each parameter
    	pcolor = ['blue','dodgerblue',\
    	        'salmon','red',\
    	        'sandybrown',\
    	        'darkorange',\
    	        'green','limegreen','palegreen',\
    	        'mediumpurple','darkorchid','orchid',\
    	        'gold','yellow',\
    	        'steelblue','royalblue','turquoise']
    	# Savefig name
    	figname = ['vel','vel_disp',\
    	        'cont_slope','cont_amp',\
    	        'na_feii_amp',\
    	        'br_feii_amp',\
    	        'oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',\
    	        'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff',\
    	        'na_Hb_amp','na_Hb_voff',\
    	        'br_Hb_amp','br_Hb_fwhm','br_Hb_voff']
    elif model_type=='NO_outflow':
    	# Colors for each parameter
    	pcolor = ['blue','dodgerblue',\
    	        'salmon','red',\
    	        'sandybrown',\
    	        'darkorange',\
    	        'green','limegreen','palegreen',\
    	        'gold','yellow',\
    	        'steelblue','royalblue','turquoise']
    	# Savefig name
    	figname = ['vel','vel_disp',\
    	        'cont_slope','cont_amp',\
    	        'na_feii_amp',\
    	        'br_feii_amp',\
    	        'oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',\
    	        'na_Hb_amp','na_Hb_voff',\
    	        'br_Hb_amp','br_Hb_fwhm','br_Hb_voff']

    nwalkers = np.shape(sampler_chain)[0]
    nsteps = np.shape(sampler_chain)[1]
    ndim = len(params)
    # Initialize figures and axes
    # Make an updating plot of the chain
    fig = plt.figure(figsize=(10,8)) 
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
    ax1  = plt.subplot(gs[0,0])
    ax2  = plt.subplot(gs[0,1])
    ax3  = plt.subplot(gs[1,0:2])
    for p in range(0,len(params),1):
        print p
        # Clear axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        par = p
        pcol = pcolor[p]
        flat = sampler_chain[:, burnin:, :].reshape((-1, ndim))
        flat = flat[:,par]
        print('OLD FLAT: %s ' % np.shape(flat) )
        # Remove large outliers here
        flat = flat[~remove_outliers(flat)] 
        print('NEW FLAT: %s ' % np.shape(flat) )
        # print np.mean(flat), np.std(flat)
        pname = param_labels[par]
        hbins = determine_hist_bins([flat])
        # Histogram
        n, bins, patches = ax1.hist(flat, hbins[0], normed=True, facecolor=pcol, alpha=0.75)
        pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,100)
        # pmax.append(pdfmax)
        # pmin_std.append(low1)
        # pmax_std.append(upp1)
        ax1.axvline(pdfmax,linestyle='--',color='black',label='$\mu=%0.6f$' % pdfmax)
        ax1.axvline(pdfmax-low1,linestyle=':',color='black',label='$\sigma_-=%0.6f$' % low1)
        ax1.axvline(pdfmax+upp1,linestyle=':',color='black',label='$\sigma_+=%0.6f$' % upp1)
        ax1.plot(xvec,yvec,color='k')
        # Output bestfit parameters in axis 2
        ax2.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.6f$' % pdfmax)
        ax2.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.6f$' % low1)
        ax2.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.6f$' % upp1)
        # Store values
        par_best.append(pdfmax)
        sig_low.append(low1)
        sig_upp.append(upp1)
        ax2.legend(loc='upper left',fontsize=14)
        ax1.set_xlabel(r'%s' % pname,fontsize=12)
        ax1.set_ylabel(r'$p$(%s)' % pname,fontsize=12)
        # Parameter chain
        for i in range(0,nwalkers,1):
            ax3.plot(range(nsteps),sampler_chain[i,:,par],color='black',linewidth=0.5,alpha=0.5)
        avg_chain = []
        for i in range(0,np.shape(sampler_chain)[1],1):
        #     print i, np.median(sampler.chain[:,i,par])
            # avg_chain.append(np.mean(sampler.chain[:,i,par]))
            avg_chain.append(np.mean(sampler_chain[:,i,par]))
        ax3.plot(range(nsteps),avg_chain,color='red',linewidth=2.0,label='Average')
        ax3.axvline(burnin,color='dodgerblue',linestyle='--',linewidth=2,label='Burn-in = %d' % burnin)
        ax3.set_xlim(0,nsteps)
        ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
        ax3.set_ylabel(r'%s' % pname,fontsize=12)
        ax3.legend(loc='upper right')
        # plt.tight_layout()
        plt.savefig(run_dir+'%s_MCMC.png' % (figname[p]) ,dpi=300,fmt='png')

    return figname,par_best,sig_low,sig_upp


def emline_flux_plots(run_dir,model_type):
    # Store best-fit parameter values from histogram fitting
    par_best = [] # best fit parameter
    sig_low  = [] # lower-bound 1-sigma error in best-fit
    sig_upp  = [] # upper-bound 1-sigma error in best-fit
    if model_type=='outflow':
        # Labels for each parameter 
        plabels = ['$f_\mathrm{Na\;FeII}$','$f_\mathrm{Br\;FeII}$',\
                  '$f_{\mathrm{Br\;Hb}}$','$f_{\mathrm{Na\;Hb}}$','$f_{\mathrm{Hb\;Outflow}}$',\
                  '$f_\mathrm{[OIII]4959\;Core}$','$f_\mathrm{[OIII]4959\;Outflow}$',\
                  '$f_\mathrm{[OIII]5007\;Core}$','$f_\mathrm{[OIII]5007\;Outflow}$']
        # Colors for each parameter
        pcolor = ['sandybrown','darkorange',\
                  'yellow','steelblue','turquoise',\
                  'green','limegreen',\
                  'mediumpurple','darkorchid']
        # Savefig name
        figname = ['na_feii_flux','br_feii_flux',\
                   'br_Hb_flux','na_Hb_flux','br_Hb_outflow_flux',\
                   'oiii4959_core_flux','oiii4959_outflow_flux',\
                   'oiii5007_core_flux','oiii5007_outflow_flux']
    elif model_type=='NO_outflow':
        # Labels for each parameter 
        plabels = ['$f_\mathrm{Na\;FeII}$','$f_\mathrm{Br\;FeII}$',\
                  '$f_{\mathrm{Br\;Hb}}$','$f_{\mathrm{Na\;Hb}}$',\
                  '$f_\mathrm{[OIII]4959\;Core}$',\
                  '$f_\mathrm{[OIII]5007\;Core}$']
        # Colors for each parameter
        pcolor = ['sandybrown','darkorange',\
                  'yellow','steelblue',\
                  'green',\
                  'mediumpurple']
        # Savefig name
        figname = ['na_feii_flux','br_feii_flux',\
                   'br_Hb_flux','na_Hb_flux',\
                   'oiii4959_core_flux',\
                   'oiii5007_core_flux']

    # Open the file containing the line flux data
    df = pd.read_csv(run_dir+'em_line_fluxes.csv')
    df = df.drop(df.columns[0], axis=1)
    nsteps = len(np.array(df[df.columns[0]]))
    burnin = int(nsteps - nsteps*0.25)       
    
    # Initialize figures and axes
    # Make an updating plot of the chain
    fig = plt.figure(figsize=(10,8)) 
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
    ax1  = plt.subplot(gs[0,0])
    ax2  = plt.subplot(gs[0,1])
    ax3  = plt.subplot(gs[1,0:2])
    for p in range(0,len(df.columns),1):
        print p
        # Clear axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        par = p
        pcol = pcolor[p]
        flat = np.array(df[df.columns[p]])

        pname = plabels[p]
        hbins = determine_hist_bins([flat[burnin:]])
        
        # Histogram
        n, bins, patches = ax1.hist(flat[burnin:], hbins[0], normed=True, facecolor=pcol, alpha=0.75)
        pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,100)
        # pmax.append(pdfmax)
        # pmin_std.append(low1)
        # pmax_std.append(upp1)
        ax1.axvline(pdfmax,linestyle='--',color='black',label='$\mu=%0.6f$' % pdfmax)
        ax1.axvline(pdfmax-low1,linestyle=':',color='black',label='$\sigma_-=%0.6f$' % low1)
        ax1.axvline(pdfmax+upp1,linestyle=':',color='black',label='$\sigma_+=%0.6f$' % upp1)
        ax1.plot(xvec,yvec,color='k')
        # Output bestfit parameters in axis 2
        ax2.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.6f$' % pdfmax)
        ax2.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.6f$' % low1)
        ax2.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.6f$' % upp1)
        # Store values
        par_best.append(pdfmax)
        sig_low.append(low1)
        sig_upp.append(upp1)
        ax2.legend(loc='upper left',fontsize=14)
        ax1.set_xlabel(r'%s' % pname,fontsize=12)
        ax1.set_ylabel(r'$p$(%s)' % pname,fontsize=12)
        # Parameter chain
        ax3.plot(range(nsteps),flat,color='black',linewidth=0.5,alpha=0.5)
        ax3.axhline(pdfmax,color='red',linewidth=2.0,label='Average')
        ax3.axvline(burnin,color='dodgerblue',linestyle='--',linewidth=2,label='Burn-in = %d' % burnin)
        ax3.set_xlim(0,nsteps)
        ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
        ax3.set_ylabel(r'%s' % pname,fontsize=12)
        ax3.legend(loc='upper right')
        # plt.tight_layout()
        plt.savefig(run_dir+'%s_MCMC.png' % (figname[p]) ,dpi=300,fmt='png')

    return figname,par_best,sig_low,sig_upp

def flux2lum(eline_flux_figname,eline_flux_best,eline_flux_sig_low,eline_flux_sig_upp,z,run_dir,H0=71.0,Om0=0.27):
    # Rename fignames _flux to _lum
    figname = []
    for name in eline_flux_figname:
        figname.append(name[:-4]+'lum')
    print figname
    # The SDSS spectra are normalized by 10^(-17), so now multiply each flux by 10^(-17)
    eline_flux_best    = np.array(eline_flux_best) * 1.0E-17
    eline_flux_sig_low = np.array(eline_flux_sig_low) * 1.0E-17
    eline_flux_sig_upp = np.array(eline_flux_sig_upp) * 1.0E-17
    # Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
    cosmo = FlatLambdaCDM(H0, Om0)
    d_mpc = cosmo.luminosity_distance(z) 
    d_cm = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
    # Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
    eline_lum_best    = (eline_flux_best * 4*np.pi * d_cm**2    )/1.0E+42
    eline_lum_sig_low = (eline_flux_sig_low * 4*np.pi * d_cm**2 )/1.0E+42
    eline_lum_sig_upp = (eline_flux_sig_upp * 4*np.pi* d_cm**2)/1.0E+42
    
    return figname,eline_lum_best,eline_lum_sig_low,eline_lum_sig_upp

def write_fits_table(figname,par_best,sig_low,sig_upp,run_dir,model_type):
    # Write best-fit paramters to FITS table
    col1 = fits.Column(name='parameter', format='20A', array=figname)
    col2 = fits.Column(name='best_fit', format='E', array=par_best)
    col3 = fits.Column(name='sigma_low', format='E', array=sig_low)
    col4 = fits.Column(name='sigma_upp', format='E', array=sig_upp)
    cols = fits.ColDefs([col1,col2,col3,col4])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(run_dir+'par_table.fits',overwrite=True)

def write_MCMC_data(sampler_chain,run_dir,model_type):
	nwalkers = np.shape(sampler_chain)[0]
	nsteps = np.shape(sampler_chain)[1]
	ndim = np.shape(sampler_chain)[2]
	# Write MCMC data to csv
	flat = sampler_chain[:, :, :].reshape((-1, ndim))
    
	if model_type=='outflow':
		d = {'vel':flat[:,0],'vel_disp':flat[:,1],\
		     'cont_slope':flat[:,2],'cont_amp':flat[:,3],\
		     'na_feii_amp':flat[:,4],\
		     'br_feii_amp':flat[:,5],\
		     'oiii5007_core_amp':flat[:,6],'oiii5007_core_fwhm':flat[:,7],'oiii5007_core_voff':flat[:,8],\
		     'oiii5007_outflow_amp':flat[:,9],'oiii5007_outflow_fwhm':flat[:,10],'oiii5007_outflow_voff':flat[:,11],\
		     'na_Hb_amp':flat[:,12],'na_Hb_voff':flat[:,13],\
		     'br_Hb_amp':flat[:,14],'br_Hb_fwhm':flat[:,15],'br_Hb_voff':flat[:,16]}
		
		df = pd.DataFrame(data=d,columns=['vel','vel_disp',\
			  'cont_slope','cont_amp',\
			  'na_feii_amp',\
			  'br_feii_amp',\
			  'oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',\
			  'oiii5007_outflow_amp','oiii5007_outflow_fwhm','oiii5007_outflow_voff',\
			  'na_Hb_amp','na_Hb_voff',\
			  'br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])
	elif model_type=='NO_outflow':
		d = {'vel':flat[:,0],'vel_disp':flat[:,1],\
		     'cont_slope':flat[:,2],'cont_amp':flat[:,3],\
		     'na_feii_amp':flat[:,4],\
		     'br_feii_amp':flat[:,5],\
		     'oiii5007_core_amp':flat[:,6],'oiii5007_core_fwhm':flat[:,7],'oiii5007_core_voff':flat[:,8],\
		     'na_Hb_amp':flat[:,9],'na_Hb_voff':flat[:,10],\
		     'br_Hb_amp':flat[:,11],'br_Hb_fwhm':flat[:,12],'br_Hb_voff':flat[:,13]}
		
		df = pd.DataFrame(data=d,columns=['vel','vel_disp',\
		    'cont_slope','cont_amp',\
		    'na_feii_amp',\
		    'br_feii_amp',\
		    'oiii5007_core_amp','oiii5007_core_fwhm','oiii5007_core_voff',\
		    'na_Hb_amp','na_Hb_voff',\
		    'br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])

	df.to_csv(run_dir+'MCMC_data.csv',sep=',')
    
def plot_best_model(lam_gal,galaxy,best_model,run_dir,model_type):
    # Initialize figures and axes
	# Make an updating plot of the chain
    fig = plt.figure(figsize=(10,6)) 
    gs = gridspec.GridSpec(4, 1)
    gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
    ax1  = plt.subplot(gs[0:3,0])
    ax2  = plt.subplot(gs[3,0])
    
    if model_type=='outflow':
    	# Galaxy + Best-fit Model
    	ax1.plot(lam_gal,galaxy,linewidth=1.0,color='black',label='Data')
    	ax1.plot(lam_gal,best_model[0],linewidth=1.0,color='red',label='Model')
    	ax1.plot(lam_gal,best_model[1][0],linewidth=1.0,color='limegreen',label='Host')
    	ax1.plot(lam_gal,best_model[1][1],linewidth=1.0,color='violet',label='AGN Cont.')
    	ax1.plot(lam_gal,best_model[1][2],linewidth=1.0,color='orange',label='Na. FeII')
    	ax1.plot(lam_gal,best_model[1][3],linewidth=1.0,color='darkorange',label='Br. FeII')
    	ax1.plot(lam_gal,best_model[1][4],linewidth=1.0,color='blue',label='Broad')
    	ax1.plot(lam_gal,best_model[1][5],linewidth=1.0,color='dodgerblue',label='Narrow')
    	ax1.plot(lam_gal,best_model[1][6],linewidth=1.0,color='dodgerblue',label='')
    	ax1.plot(lam_gal,best_model[1][7],linewidth=1.0,color='dodgerblue')
    	ax1.plot(lam_gal,best_model[1][8],linewidth=1.0,color='orangered',label='Outflow')
    	ax1.plot(lam_gal,best_model[1][9],linewidth=1.0,color='orangered')
    	ax1.plot(lam_gal,best_model[1][10],linewidth=1.0,color='orangered')
    elif model_type=='NO_outflow':
    	# Galaxy + Best-fit Model
    	ax1.plot(lam_gal,galaxy,linewidth=1.0,color='black',label='Data')
    	ax1.plot(lam_gal,best_model[0],linewidth=1.0,color='red',label='Model')
    	ax1.plot(lam_gal,best_model[1][0],linewidth=1.0,color='limegreen',label='Host')
    	ax1.plot(lam_gal,best_model[1][1],linewidth=1.0,color='violet',label='AGN Cont.')
    	ax1.plot(lam_gal,best_model[1][2],linewidth=1.0,color='orange',label='Na. FeII')
    	ax1.plot(lam_gal,best_model[1][3],linewidth=1.0,color='darkorange',label='Br. FeII')
    	ax1.plot(lam_gal,best_model[1][4],linewidth=1.0,color='blue',label='Broad')
    	ax1.plot(lam_gal,best_model[1][5],linewidth=1.0,color='dodgerblue',label='Narrow')
    	ax1.plot(lam_gal,best_model[1][6],linewidth=1.0,color='dodgerblue',label='')
    	ax1.plot(lam_gal,best_model[1][7],linewidth=1.0,color='dodgerblue')

    ax1.set_xticklabels([])
    ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
    ax1.set_ylim(-0.5*np.median(best_model[0]),np.max([np.max(galaxy),np.max(best_model[0])]))
    ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
    ax1.legend(loc='best',fontsize=8)
    # Residuals
    sigma_resid = np.std(galaxy-best_model[0])
    ax2.plot(lam_gal,(galaxy-best_model[0])*3,linewidth=1.0,color='red',label='$\sigma=%0.4f$' % (sigma_resid))
    ax2.axhline(0.0,linewidth=1.0,color='black',linestyle='--')
    ax2.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
    ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
    ax2.set_yticks([0.0])
    ax2.legend(loc='best',fontsize=8)
    plt.savefig(run_dir+'bestfit_model.png',dpi=300,fmt='png')

    # Write out full models to FITS files 
    if model_type=='outflow':
        #cont,na_feii_template,br_feii_template,\
        # br_Hb,na_Hb,na_oiii4959,na_oiii5007,br_oiii4959,br_oiii5007,br_Hb_outflow
        hdu1  = fits.PrimaryHDU(galaxy) # original spectrum 
        hdu2  = fits.ImageHDU(lam_gal) # rest wavelength
        hdu3  = fits.ImageHDU(best_model[0]) # best-fit model'

        hdu4  = fits.ImageHDU(best_model[1][0]) # stellar-continuum
        hdu5  = fits.ImageHDU(best_model[1][1]) # agn continuum
        hdu6  = fits.ImageHDU(best_model[1][2]) # na. FeII emission
        hdu7  = fits.ImageHDU(best_model[1][3]) # br. FeII emission
        hdu8  = fits.ImageHDU(best_model[1][4]) # Br. Hb
        hdu9  = fits.ImageHDU(best_model[1][5]) # Na. Hb Core
        hdu10 = fits.ImageHDU(best_model[1][6]) # Na. [OIII]4959 Core
        hdu11 = fits.ImageHDU(best_model[1][7]) # Na. [OIII]5007 Core
        hdu12 = fits.ImageHDU(best_model[1][8]) # Br. [OIII]4960 Outflow
        hdu13 = fits.ImageHDU(best_model[1][9]) # Br. [OIII]5007 Outflow
        hdu14 = fits.ImageHDU(best_model[1][10]) # Br. Hb Outflow

        hdulist= fits.HDUList([hdu1,hdu2,hdu3,hdu4,hdu5,hdu6,hdu7,hdu8,hdu9,hdu10,hdu11,hdu12,hdu13,hdu14])
        hdulist.writeto(run_dir+'best_fit_models.fits',overwrite=True)
        hdulist.close()
    elif model_type=='NO_outflow':
        hdu1  = fits.PrimaryHDU(galaxy) # original spectrum 
        hdu2  = fits.ImageHDU(lam_gal) # rest wavelength
        hdu3  = fits.ImageHDU(best_model[0]) # best-fit model
        hdu4  = fits.ImageHDU(best_model[1][0]) # stellar-continuum
        hdu5  = fits.ImageHDU(best_model[1][1]) # agn continuum
        hdu6  = fits.ImageHDU(best_model[1][2]) # na. FeII emission
        hdu7  = fits.ImageHDU(best_model[1][3]) # br. FeII emission
        hdu8  = fits.ImageHDU(best_model[1][4]) # Br. Hb
        hdu9  = fits.ImageHDU(best_model[1][5]) # Na. Hb Core
        hdu10 = fits.ImageHDU(best_model[1][6]) # Na. [OIII]4959 Core
        hdu11 = fits.ImageHDU(best_model[1][7]) # Na. [OIII]5007 Core

        hdulist= fits.HDUList([hdu1,hdu2,hdu3,hdu4,hdu5,hdu6,hdu7,hdu8,hdu9,hdu10,hdu11])
        hdulist.writeto(run_dir+'best_fit_models.fits',overwrite=True)
        hdulist.close()


