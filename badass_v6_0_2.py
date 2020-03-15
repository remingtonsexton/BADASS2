#####################################################

# Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS,version 6.0.1)
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
import datetime
from os import path
import os
import shutil
import sys
import emcee
from astropy.stats import mad_std
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM
import re
import natsort
plt.style.use('dark_background') # For cool tron-style dark plots
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 100000
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import gc # Garbage collector
##################################################################################


#### Run BADASS ##################################################################

def run_BADASS(file,run_dir,temp_dir,
			   fit_reg=(4400,5800),good_thresh=0.60,test_outflows=True,mcbs_niter=10,
			   fit_feii=True,fit_losvd=True,fit_host=True,fit_power=True,
               fit_broad=True,fit_narrow=True,fit_outflows=True,tie_narrow=False,
			   nwalkers=100,auto_stop=True,conv_type='all',min_samp=2500,ncor_times=2.5,autocorr_tol=5.0,
			   write_iter=100,write_thresh=100,burn_in=23500,min_iter=2500,max_iter=25000,threads=1,
			   ):
	# This is the main function responsible for executing the fitting steps in order
	# by calling all functions below.

	# Determine fitting region
	fit_reg,good_frac = determine_fit_reg(file,good_thresh,run_dir,fit_reg)
	if (fit_reg is None):
	    print('\n Fit region too small! Moving to next object... \n')
	    return None
	elif (good_frac < good_thresh) and (fit_reg is not None): # if fraction of good pixels is less than good_threshold, then move to next object
	    print('\n Not enough good channels above threshold! Moving onto next object... \n')
	    return None
	elif (good_frac >= good_thresh) and (fit_reg is not None):
	    print('          Fitting region: (%d,%d)' % (fit_reg[0],fit_reg[1]))
	    print('          Fraction of good channels = %0.2f' % (good_frac))
	
	# Prepare SDSS spectrum for fitting
	lam_gal,galaxy,noise,velscale,vsyst,temp_list,z,ebv,npix,ntemp,temp_fft,npad = sdss_prepare(file,fit_reg,temp_dir,run_dir,plot=True)
	print('          z = %0.4f' % z)
	print('          E(B-V) =  %0.4f' % ebv)
	print('          Velocity Scale = %0.4f (km/s/pixel)' % velscale)
	print('----------------------------------------------------------------------------------------------------')
	###########################################################################################################
	

	# Testing for outflows 
	if (test_outflows==True) & (fit_outflows==True) & ((fit_reg[0]<=4400.)==True) & ((fit_reg[1] >=5800.)==True) & (mcbs_niter>0): # Only do this for Hb Region
	    print('\n Running outflow tests...')
	    print('----------------------------------------------------------------------------------------------------')

	    fit_outflows,outflow_res,res_sigma = outflow_test(lam_gal,galaxy,noise,run_dir,velscale,mcbs_niter)
	    if fit_outflows==True:
	        print('  Outflows detected: including outflow components in fit...')
	    elif fit_outflows==False:
	        print('  Outflows not detected: disabling outflow components from fit...')


	# Initialize maximum likelihood parameters
	param_dict = initialize_mcmc(lam_gal,galaxy,run_dir,fit_reg=fit_reg,fit_type='init',
	                             fit_feii=fit_feii,fit_losvd=fit_losvd,fit_host=fit_host,
	                             fit_power=fit_power,fit_broad=fit_broad,
	                             fit_narrow=fit_narrow,fit_outflows=fit_outflows,
	                             tie_narrow=tie_narrow)

	# By generating the galaxy and FeII templates before, instead of generating them with each iteration, we save a lot of time
	gal_temp = galaxy_template(lam_gal,age=5.0) # 'age=None' option selections a NLS1 template
	if (fit_feii==True):
	    na_feii_temp,br_feii_temp = initialize_feii(lam_gal,velscale,fit_reg)
	elif (fit_feii==False):
	    na_feii_temp,br_feii_temp = None,None

	# Peform maximum likelihood
	result_dict, sn = max_likelihood(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
	                                 temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,monte_carlo=False)
	
	# If the total continuum level is < 10, disable the host-galaxy and losvd components 
	# (there is no point in fitting them because the fits will be horrible)
	if (sn<0.1):
	    print('\n Continuum S/N = %0.2f' % sn)
	    print('\n Total continuum level < 10.  Disabling host_galaxy/LOSVD components...\n')
	    fit_host  = True
	    fit_losvd = False

	# Initialize parameters for emcee
	param_dict = initialize_mcmc(lam_gal,galaxy,run_dir,fit_reg=fit_reg,fit_type='final',
	                             fit_feii=fit_feii,fit_losvd=fit_losvd,fit_host=fit_host,
	                             fit_power=fit_power,fit_broad=fit_broad,
	                             fit_narrow=fit_narrow,fit_outflows=fit_outflows,tie_narrow=tie_narrow)
	
	# Replace initial conditions with best fit max. likelihood parameters (the old switcharoo)
	for key in result_dict:
	    if key in param_dict:
	        param_dict[key]['init']=result_dict[key]['res']
	print('\n     Final parameters and their initial guesses:')
	print('----------------------------------------------------------------------------------------------------')

	for key in param_dict:
	    print('          %s = %0.2f' % (key,param_dict[key]['init']) )
	#######################################################################################################

	# Run emcee
	print('\n Performing MCMC iterations...')
	print('----------------------------------------------------------------------------------------------------')

	# Extract relevant stuff from dicts
	param_names  = [param_dict[key]['name'] for key in param_dict ]
	init_params  = [param_dict[key]['init'] for key in param_dict ]
	bounds       = [param_dict[key]['plim'] for key in param_dict ]

	# Check number of walkers
	# If number of walkers < 2*(# of params) (the minimum required), then set it to that
	if nwalkers<2*len(param_names):
		print('\n Number of walkers < 2 x (# of parameters)!.  Setting nwalkers = %d' % (2*len(param_names)))
	
	ndim, nwalkers = len(init_params), nwalkers # minimum walkers = 2*len(params)
	# initialize walker starting positions based on parameter estimation from Maximum Likelihood fitting
	pos = [init_params + 1.0e-4*np.random.randn(ndim) for i in range(nwalkers)]
	# Run emcee
	# args = arguments of lnprob (log-probability function)
	lnprob_args=(param_names,bounds,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
	      temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir)
	
	sampler_chain,burn_in = run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
	                                  auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,
	                                  burn_in,min_iter,max_iter,threads=threads)
	
	# Add chains to each parameter in param dictionary
	for k,key in enumerate(param_names):
	    if key in param_dict:
	        param_dict[key]['chain']=sampler_chain[:,:,k]
	    
	print('\n Fitting chains...')
	print('----------------------------------------------------------------------------------------------------')
	param_dict = param_plots(param_dict,burn_in,run_dir,save_plots=True)

	flux_dict  = emline_flux_plots(burn_in,nwalkers,run_dir,save_plots=True)
	
	lum_dict   = flux2lum(flux_dict,burn_in,nwalkers,z,run_dir,H0=71.0,Om0=0.27,save_plots=True)

	# If stellar velocity is fit, estimate the systemic velocity of the galaxy;
	# SDSS redshifts are based on average emission line redshifts.
	if ('stel_vel' in param_dict):
		print('\n Estimating systemic velocity of galaxy...')
		print('----------------------------------------------------------------------------------------------------')
		z_best = systemic_vel_est(z,param_dict)
		print('\n     Best-fit Systemic Redshift = %0.6f (-%0.6f,+%0.6f)' %  (z_best[0],z_best[1],z_best[2]))
		# Write to log file
		write_log(z_best,70,run_dir)
	# If broadlines are fit, estimate BH mass
	if ('br_Hb_fwhm' in param_dict) and ('br_Hb_lum' in lum_dict):
		print('\n Estimating black hole mass from Broad H-beta FWHM and luminosity...')
		print('----------------------------------------------------------------------------------------------------')
		L5100_Hb = hbeta_to_agn_lum(lum_dict['br_Hb_lum']['par_best'],lum_dict['br_Hb_lum']['sig_low'],lum_dict['br_Hb_lum']['sig_upp'])
		print('\n     AGN Luminosity: log10(L5100) = %0.2f (-%0.2f, +%0.2f)' % (L5100_Hb[0],L5100_Hb[1],L5100_Hb[2]))
		log_MBH_Hbeta = estimate_BH_mass_hbeta(param_dict['br_Hb_fwhm']['par_best'],param_dict['br_Hb_fwhm']['sig_low'],param_dict['br_Hb_fwhm']['sig_upp'],L5100_Hb[3])
		print('\n     BH Mass:         log10(M_BH) = %0.2f (-%0.2f, +%0.2f)' % (log_MBH_Hbeta[0],log_MBH_Hbeta[1],log_MBH_Hbeta[2]))
		# Write to log file
		write_log((L5100_Hb,log_MBH_Hbeta),60,run_dir)
	if ('br_Ha_fwhm' in param_dict) and ('br_Ha_lum' in lum_dict):
		print('\n Estimating black hole mass from Broad H-alpha FWHM and luminosity...')
		print('----------------------------------------------------------------------------------------------------')
		L5100_Ha = halpha_to_agn_lum(lum_dict['br_Ha_lum']['par_best'],lum_dict['br_Ha_lum']['sig_low'],lum_dict['br_Ha_lum']['sig_upp'])
		print('\n     AGN Luminosity: log10(L5100) = %0.2f (-%0.2f, +%0.2f)' % (L5100_Ha[0],L5100_Ha[1],L5100_Ha[2]))
		log_MBH_Halpha = estimate_BH_mass_halpha(param_dict['br_Ha_fwhm']['par_best'],param_dict['br_Ha_fwhm']['sig_low'],param_dict['br_Ha_fwhm']['sig_upp'],L5100_Ha[3])
		print('\n     BH Mass:         log10(M_BH) = %0.2f (-%0.2f, +%0.2f)' % (log_MBH_Halpha[0],log_MBH_Halpha[1],log_MBH_Halpha[2]))
		# Write to log files
		write_log((L5100_Ha,log_MBH_Halpha),61,run_dir)

	# If BPT lines are fit, output a BPT diagram
	if all(elem in lum_dict for elem in ('na_Hb_core_lum','na_oiii5007_core_lum','na_Ha_core_lum','na_nii6585_core_lum')):
		print('\n Generating BPT diagram...')
		print('----------------------------------------------------------------------------------------------------')
		bpt_diagram(lum_dict,run_dir)

	print('\n Saving Data...')
	print('----------------------------------------------------------------------------------------------------')
	# Write best fit parameters to fits table
	write_param(param_dict,flux_dict,lum_dict,run_dir)
	# Write all chains to a fits table
	write_chain(param_dict,flux_dict,lum_dict,run_dir)
	# Plot and save the best fit model and all sub-components
	plot_best_model(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
	                       temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir)
	
	print('\n Cleaning up...')
	print('----------------------------------------------------------------------------------------------------')
	# Delete redundant files to cut down on space
	cleanup(run_dir)
	print('----------------------------------------------------------------------------------------------------')
	# delete variables and manually collect garbage
	del fit_reg
	del good_frac
	del lam_gal
	del galaxy
	del noise
	del velscale
	del vsyst
	del temp_list
	del z
	del ebv
	del npix
	del ntemp
	del temp_fft
	del npad
	del gal_temp
	del na_feii_temp
	del br_feii_temp
	del result_dict
	del sn
	del param_names
	del init_params
	del bounds
	del ndim
	del nwalkers
	del pos
	del lnprob_args
	del sampler_chain
	del burn_in
	del param_dict
	del flux_dict
	# del lum_dict

	gc.collect()

	return None

#### Calculate Sysetemic Velocity ################################################

def systemic_vel_est(z,param_dict):
	c = 299792.458   
	# Get measured stellar velocity
	stel_vel     = param_dict['stel_vel']['par_best']
	stel_vel_low = param_dict['stel_vel']['sig_low']
	stel_vel_upp = param_dict['stel_vel']['sig_upp']
	# Calculate new redshift
	z_best     = (z+1)*(1+stel_vel/c)-1
	z_best_low = z_best - ( (z+1)*(1+(stel_vel-stel_vel_low)/c)-1 )
	z_best_upp = ( (z+1)*(1+(stel_vel+stel_vel_upp)/c)-1 ) - z_best	
	#
	return (z_best,z_best_low,z_best_upp)



##################################################################################

#### BPT Diagram #################################################################

def bpt_diagram(lum_dict,run_dir):
	# Get luminosities (or fluxes)
	hb_lum       = lum_dict['na_Hb_core_lum']['par_best']
	hb_lum_low   = lum_dict['na_Hb_core_lum']['sig_low']
	hb_lum_upp   = lum_dict['na_Hb_core_lum']['sig_upp']
	#
	oiii_lum     = lum_dict['na_oiii5007_core_lum']['par_best']
	oiii_lum_low = lum_dict['na_oiii5007_core_lum']['sig_low']
	oiii_lum_upp = lum_dict['na_oiii5007_core_lum']['sig_upp']
	#
	ha_lum       = lum_dict['na_Ha_core_lum']['par_best']
	ha_lum_low   = lum_dict['na_Ha_core_lum']['sig_low']
	ha_lum_upp   = lum_dict['na_Ha_core_lum']['sig_upp']
	#
	nii_lum     = lum_dict['na_nii6585_core_lum']['par_best']
	nii_lum_low = lum_dict['na_nii6585_core_lum']['sig_low']
	nii_lum_upp = lum_dict['na_nii6585_core_lum']['sig_upp']
	#
	# Calculate log ratios 
	log_oiii_hb_ratio = np.log10(oiii_lum/hb_lum)
	log_nii_ha_ratio  = np.log10(nii_lum/ha_lum)
	# Calculate uncertainnties
	log_oiii_hb_ratio_low = 0.434*((np.sqrt((oiii_lum_low/oiii_lum)**2+(hb_lum_low/hb_lum)**2))/(oiii_lum/hb_lum))
	log_oiii_hb_ratio_upp = 0.434*((np.sqrt((oiii_lum_upp/oiii_lum)**2+(hb_lum_upp/hb_lum)**2))/(oiii_lum/hb_lum))
	log_nii_ha_ratio_low  = 0.434*((np.sqrt((nii_lum_low/nii_lum)**2+(ha_lum_low/ha_lum)**2))/(nii_lum/ha_lum))
	log_nii_ha_ratio_upp  = 0.434*((np.sqrt((nii_lum_upp/nii_lum)**2+(ha_lum_upp/ha_lum)**2))/(nii_lum/ha_lum))
	# Plot and save figure
	fig = plt.figure(figsize=(8,8))
	ax1 = fig.add_subplot(1,1,1)
	#
	ax1.errorbar(log_nii_ha_ratio,log_oiii_hb_ratio,
				xerr = [[log_nii_ha_ratio_low],[log_nii_ha_ratio_upp]],
				yerr = [[log_oiii_hb_ratio_low],[log_oiii_hb_ratio_upp]],
				# xerr = 0.1,
				# yerr = 0.1,
				color='xkcd:black',marker='.',markersize=15,
				elinewidth=1,ecolor='white',markeredgewidth=1,markerfacecolor='red',
				capsize=5,linestyle='None',zorder=15)
	# Kewley et al. 2001
	x_k01 = np.arange(-2.0,0.30,0.01)
	y_k01 = 0.61 / (x_k01 - 0.47) + 1.19
	ax1.plot(x_k01,y_k01,linewidth=2,linestyle='--',color='xkcd:turquoise',label='Kewley et al. 2001')
	# Kauffmann et al. 2003 
	x_k03 = np.arange(-2.0,0.0,0.01)
	y_k03 = 0.61 / (x_k03 - 0.05) + 1.3
	ax1.plot(x_k03,y_k03,linewidth=2,linestyle='--',color='xkcd:lime green',label='Kauffmann et al. 2003')
	#
	ax1.annotate('LINER', xy=(0.90, 0.25),  xycoords='axes fraction',
            xytext=(0.90, 0.25), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14
            )
	ax1.annotate('AGN', xy=(0.75, 0.75),  xycoords='axes fraction',
            xytext=(0.75, 0.75), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14
            )
	ax1.annotate('STARFORMING', xy=(0.35, 0.35),  xycoords='axes fraction',
            xytext=(0.35, 0.35), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14
            )
	ax1.annotate('COMP.', xy=(0.70, 0.10),  xycoords='axes fraction',
            xytext=(0.70, 0.10), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14
            )
	#
	ax1.set_ylabel(r'$\log_{10}(\rm{[OIII]}/\rm{H}\beta)$',fontsize=14)
	ax1.set_xlabel(r'$\log_{10}(\rm{[NII]}/\rm{H}\alpha)$',fontsize=14)
	ax1.tick_params(axis="x", labelsize=14)
	ax1.tick_params(axis="y", labelsize=14)
	ax1.set_title('BPT Diagnostic',fontsize=16)
	ax1.set_xlim(-1.50,0.75)
	ax1.set_ylim(-1.25,1.50)
	ax1.legend(fontsize=14)
	# 
	plt.tight_layout()
	plt.savefig(run_dir+'bpt_classification.png',dpi=150,fmt='png')
	fig.clear()
	plt.close()
	#
	return None

##################################################################################

#### BH Mass Estimation ##########################################################

# The following functions are required for BH mass estimation
def normal_dist(x,mean,sigma):
    return 1.0/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mean)**2/(2.0*sigma**2))

def asymmetric_gaussian(mean,sigma_low,sigma_upp):
    # Creates an asymmetric gaussian from which we can sample
    # Create a linearly-space axis that goes out to N-sigma past the mean in either direction
    n_sigma = 3.
    n_samp = 100.
    x_low = np.linspace(mean-n_sigma*sigma_low,mean+n_sigma*sigma_low,n_samp)
    x_upp = np.linspace(mean-n_sigma*sigma_upp,mean+n_sigma*sigma_upp,n_samp)
    # Generate gaussian distributions for each side
    g_low = normal_dist(x_low,mean,sigma_low)
    g_upp = normal_dist(x_upp,mean,sigma_upp)
    # Normalize each gaussian to 1.0
    g_low_norm = g_low/np.max(g_low)
    g_upp_norm = g_upp/np.max(g_upp)
    # index closest to the maximum of each gaussian
    g_low_max = find_nearest(g_low_norm,np.max(g_low_norm))[1]
    g_upp_max = find_nearest(g_upp_norm,np.max(g_upp_norm))[1]
    # Split each gaussian into its respective half
    g_low_half = g_low_norm[:g_low_max]
    x_low_half = x_low[:g_low_max]
    g_upp_half = g_upp_norm[g_low_max+1:]
    x_upp_half = x_upp[g_low_max+1:]
    # Concatenate the two halves together 
    g_merged = np.concatenate([g_low_half,g_upp_half])
    x_merged = np.concatenate([x_low_half,x_upp_half])
    # Interpolate the merged gaussian 
    g_interp = interp1d(x_merged,g_merged,kind='cubic',fill_value=0.0)
    # Create new x axis to interpolate the new gaussian onto
    x_new = np.linspace(x_merged[0],x_merged[-1],n_samp)
    g_new = g_interp(x_new)
    # truncate
    cutoff = 0
    return g_new[g_new>=cutoff],x_new[g_new>=cutoff]

def hbeta_to_agn_lum(L,L_low,L_upp,n_samp=1000):
	# The equation used to convert broad H-beta luminosities to AGN luminosity 
	# can be found in Greene & Ho 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...630..122G/abstract)
	# 
	# Eq. (2): L_hbeta = (1.425  +/- 0.007 ) * 1.e+42 * (L_5100  / 1.e+44)^(1.133  +/- 0.005 )
	# 		   L_5100  = (0.7315 +/- 0.0042) * 1.e+44 * (L_hbeta / 1.e+42)^(0.8826 +/- 0.0050)
	#
	# Define variables
	A      = 0.7315 
	dA_low = 0.0042
	dA_upp = 0.0042
	#
	B      = 0.8826
	dB_low = 0.0050
	dB_upp = 0.0050
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	#
	# fig = plt.figure(figsize=(5,5))
	# ax1 = fig.add_subplot(1,1,1)
	# ax1.hist(A_,bins='doane')
	# plt.show()
	#
	# Generate samples of L_hbeta
	p_L,x_L = asymmetric_gaussian(L,L_low,L_upp) 
	p_L = p_L/p_L.sum() 
	L_hbeta = np.random.choice(a=x_L,size=n_samp,p=p_L,replace=True) * 1.e+42
	#
	L5100_ = A_ * 1.e+44 * (L_hbeta/1.e+42)**(B_)
	L5100_[L5100_/L5100_ != 1] = 0.0
	# Make distribution and get best-fit MBH and uncertainties
	n, bins = np.histogram(L5100_,bins='doane', density=True)#, facecolor='xkcd:cerulean blue', alpha=0.75)
	pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,100,L5100_)
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax),L5100_

def halpha_to_agn_lum(L,L_low,L_upp,n_samp=1000):
	# The equation used to convert broad H-alpha luminosities to AGN luminosity 
	# can be found in Greene & Ho 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...630..122G/abstract)
	# 
	# Eq. (1): L_alpha = (5.25   +/- 0.02  ) * 1.e+42 * (L_5100  / 1.e+44)^(1.157  +/- 0.005 )
	#          L_5100  = (0.2385 +/- 0.0023) * 1.e+44 * (L_alpha / 1.e+42)^(0.8643 +/- 0.0050)
	#
	# Define variables
	A      = 0.2385
	dA_low = 0.0023
	dA_upp = 0.0023
	#
	B      = 0.8643
	dB_low = 0.0050
	dB_upp = 0.0050
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	#
	# fig = plt.figure(figsize=(5,5))
	# ax1 = fig.add_subplot(1,1,1)
	# ax1.hist(A_,bins='doane')
	# plt.show()
	#
	# Generate samples of L_hbeta
	p_L,x_L = asymmetric_gaussian(L,L_low,L_upp) 
	p_L = p_L/p_L.sum() 
	L_halpha = np.random.choice(a=x_L,size=n_samp,p=p_L,replace=True) * 1.e+42
	#
	L5100_ = A_ * 1.e+44 * (L_halpha/1.e+42)**(B_)
	L5100_[L5100_/L5100_ != 1] = 0.0
	# Make distribution and get best-fit MBH and uncertainties
	n, bins = np.histogram(L5100_,bins='doane', density=True)#, facecolor='xkcd:cerulean blue', alpha=0.75)
	pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,100,L5100_)
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax),L5100_

def estimate_BH_mass_hbeta(fwhm,fwhm_low,fwhm_upp,L5100,n_samp=1000):
	# Generate samples of FWHM_Hbeta
	p_FWHM,x_FWHM = asymmetric_gaussian(fwhm,fwhm_low,fwhm_upp) 
	p_FWHM = p_FWHM/p_FWHM.sum() 
	FWHM_Hb = np.random.choice(a=x_FWHM,size=n_samp,p=p_FWHM,replace=True)
	#
	# Calculate BH Mass using the Sexton et al. 2019 relation (bassed on Woo et al. 2015 recalibration)
	# (https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)
	#
	# Eq. (4): M_BH = 10^(6.867,-0.153,+0.155)*(FWHM_Hb/1.e+3)^2 * (L5100/1.e+44)^(0.533,-0.033,+0.035)
 	#
	# Define variables
	A      = 6.867
	dA_low = 0.153
	dA_upp = 0.155
	#
	B      = 0.533
	dB_low = 0.033
	dB_upp = 0.035
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	#
	MBH_ = 10**(A_) * (FWHM_Hb/1.e+3)**2 * (L5100/1.e+44)**B_
	#
	# Make distribution and get best-fit MBH and uncertainties
	n, bins = np.histogram(MBH_,bins='doane', density=True)#, facecolor='xkcd:cerulean blue', alpha=0.75)
	pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,100,MBH_)
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax)

def estimate_BH_mass_halpha(fwhm,fwhm_low,fwhm_upp,L5100,n_samp=1000):
	# Generate samples of FWHM_Halpha
	p_FWHM,x_FWHM = asymmetric_gaussian(fwhm,fwhm_low,fwhm_upp) 
	p_FWHM = p_FWHM/p_FWHM.sum() 
	FWHM_Ha = np.random.choice(a=x_FWHM,size=n_samp,p=p_FWHM,replace=True)
	#
	# Calculate BH Mass using the Woo et al. 2015 relation (and after a bit of math)
	# (https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract)
	#
	# M_BH = (0.8437,-0.3232,+0.5121) * 1.e+7 * (FWHM_Ha/1.e+3)^(2.06 +/- 0.06) * (L5100/1.e+44)^(0.533,-0.033,+0.035)
 	# 
 	#
	# Define variables
	#
	A      = 0.8437
	dA_low = 0.3232
	dA_upp = 0.5121
	#
	B      = 2.06
	dB_low = 0.06
	dB_upp = 0.06
	#
	C      = 0.533
	dC_low = 0.033
	dC_upp = 0.035
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	p_C,x_C = asymmetric_gaussian(C,dC_low,dC_upp)
	p_C = p_C/p_C.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	C_ = np.random.choice(a=x_C,size=n_samp,p=p_C,replace=True)
	#
	MBH_ = (A_)* 1.e+7 * (FWHM_Ha/1.e+3)**(B_) * (L5100/1.e+44)**(C_)
	#
	# Make distribution and get best-fit MBH and uncertainties
	n, bins = np.histogram(MBH_,bins='doane', density=True)#, facecolor='xkcd:cerulean blue', alpha=0.75)
	pdfmax,low1,upp1,xvec,yvec = conf_int(get_bin_centers(bins),n,100,MBH_)
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax)

##################################################################################

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
        os.mkdir(new_fold)
        run_dir = new_fold
        if os.path.exists(prev_fold+'MCMC_chain.csv')==True:
            prev_dir = prev_fold
        else:
            prev_dir = prev_fold
        print(' Storing MCMC_output in %s' % run_dir)

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

	gal  = t['flux']
	ivar = t['ivar']
	and_mask = t['and_mask']
	# Edges of wavelength vector
	first_good = lam_gal[0]
	last_good  = lam_gal[-1]

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
		fig.clear()
		plt.close()
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
	Adapted from example from Cappellari's pPXF (Cappellari et al. 2004,2017)
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

	val,idx = find_nearest(lam_gal,5175)

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
	    hdu.close()
	
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
		plt.savefig(run_dir+'sdss_prepare.pdf',dpi=150,fmt='pdf')
		fig.clear()
		plt.close()
	################################################################################
	# Close the fits file
	hdu.close()
	################################################################################
	# Write to Log 
	write_log((file,ra,dec,z,fit_min,fit_max,velscale,ebv),0,run_dir)
	################################################################################
	# Collect garbage
	del fig
	del ax1
	del ax2
	del templates
	gc.collect()
	################################################################################

	
	return lam_gal,galaxy,noise,velscale,vsyst,temp_list,z,ebv,npix,ntemp,temp_fft,npad

##################################################################################

#### Initialize Parameters #######################################################


def initialize_mcmc(lam_gal,galaxy,run_dir,fit_reg,fit_type='init',fit_feii=True,fit_losvd=True,fit_host=True,
					fit_power=True,fit_broad=True,fit_narrow=True,fit_outflows=True,tie_narrow=True):
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
		                			   'plim':(1.0e-3,max_flux),
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
		                   			'plim':(10.0,400.),
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
		                   		   'plim':(1.0e-3,max_flux),
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
		                   			  'label':'$A_{\mathrm{Na\;FeII}}$',
		                   			  'init':feii_flux_init,
		                   			  'plim':(1.0e-3,total_flux_init),
		                   			  'pcolor':'sandybrown',
		                   			  })
		# Broad FeII amplitude
		mcmc_input['br_feii_amp'] = ({'name':'br_feii_amp',
		                   			  'label':'$A_{\mathrm{Br\;FeII}}$',
		                   			  'init':feii_flux_init,
		                   			  'plim':(1.0e-3,total_flux_init),
		                   			  'pcolor':'darkorange',
		                   			  })
	##############################################################################

	#### Emission Lines ##########################################################

	#### Narrow [OII] Doublet ##############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 3727.092 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow H-delta emission.')
		# Na. [OII]3727 Core Amplitude
		mcmc_input['na_oii3727_core_amp'] = ({'name':'na_oii3727_core_amp',
		                   				    'label':'$A_{\mathrm{[OII]3727}}$',
		                   				    'init':(oii_amp_init-total_flux_init),
		                   				    'plim':(1.0e-3,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. [OII]3727 Core FWHM
			mcmc_input['na_oii3727_core_fwhm'] = ({'name':'na_oii3727_core_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[OII]3727}}$',
			                   					 'init':250.,
			                   					 'plim':(1.0e-3,1000.),
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
		                   				    'plim':(1.0e-3,max_flux),
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
		                   				    'plim':(1.0e-3,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. [NIII]3870 Core FWHM
			mcmc_input['na_neiii_core_fwhm'] = ({'name':'na_neiii_core_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{[NeIII]}}$',
			                   					 'init':250.,
			                   					 'plim':(1.0e-3,1000.),
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
		                   				    'plim':(1.0e-3,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		if (tie_narrow==False):
			# Na. H-delta Core FWHM
			mcmc_input['na_Hd_fwhm'] = ({'name':'na_Hd_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\delta}$',
			                   					 'init':250.,
			                   					 'plim':(1.0e-3,1000.),
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

	#### Narrow H-gamma/[OIII]4363 ###############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 4341.68 < (fit_reg[1]-25.))==True))):
		print(' Fitting narrow H-gamma/[OIII]4363 emission.')

		if (tie_narrow==False):
			# Na. H-gamma Core FWHM
			mcmc_input['na_Hg_fwhm'] = ({'name':'na_Hg_fwhm',
			                   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\gamma}$',
			                   					 'init':250.,
			                   					 'plim':(1.0e-3,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
			# Na. [OIII]4363 Core FWHM
			mcmc_input['na_oiii4363_core_fwhm'] = ({'name':'na_oiii4363_core_fwhm',
			                   					 'label':'$\mathrm{FWHM}_\mathrm{[OIII]4363\;Core}$',
			                   					 'init':250.,
			                   					 'plim':(1.0e-3,1000.),
			                   					 'pcolor':'limegreen',
			                   					 })
		# Na. H-gamma Core Amplitude
		mcmc_input['na_Hg_amp'] = ({'name':'na_Hg_amp',
		                   				    'label':'$A_{\mathrm{H}\gamma}$',
		                   				    'init':(hg_amp_init-total_flux_init),
		                   				    'plim':(1.0e-3,max_flux),
		                   				    'pcolor':'green',
		                   				    })

		# Na. H-gamma Core VOFF
		mcmc_input['na_Hg_voff'] = ({'name':'na_Hg_voff',
		                   					 'label':'$\mathrm{VOFF}_{\mathrm{H}\gamma}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })
		# Na. [OIII]4363 Core Amplitude
		mcmc_input['na_oiii4363_core_amp'] = ({'name':'na_oiii4363_core_amp',
		                   				    'label':'$A_\mathrm{[OIII]4363\;Core}$',
		                   				    'init':(hg_amp_init-total_flux_init),
		                   				    'plim':(1.0e-3,max_flux),
		                   				    'pcolor':'green',
		                   				    })			
		# Na. [OIII]4363 Core VOFF
		mcmc_input['na_oiii4363_core_voff'] = ({'name':'na_oiii4363_core_voff',
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
		                   			'plim':(1.0e-3,max_flux),
		                   			'pcolor':'steelblue',
		                   			})
		# Br. H-beta FWHM
		mcmc_input['br_Hg_fwhm'] = ({'name':'br_Hg_fwhm',
		               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hg}}$',
		               	   			 'init':2500.,
		               	   			 'plim':(1.0e-3,10000.),
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
		mcmc_input['na_oiii5007_core_amp'] = ({'name':'na_oiii5007_core_amp',
		                   				    'label':'$A_{\mathrm{[OIII]5007\;Core}}$',
		                   				    'init':(oiii5007_amp_init-total_flux_init),
		                   				    'plim':(1.0e-3,max_flux),
		                   				    'pcolor':'green',
		                   				    })
		# If tie_narrow=True, then all line widths are tied to [OIII]5007, as well as their outflows (currently H-alpha/[NII]/[SII] outflows only)
		# Na. [OIII]5007 Core FWHM
		mcmc_input['na_oiii5007_core_fwhm'] = ({'name':'na_oiii5007_core_fwhm',
		                   					 'label':'$\mathrm{FWHM}_{\mathrm{[OIII]5007\;Core}}$',
		                   					 'init':250.,
		                   					 'plim':(1.0e-3,1000.),
		                   					 'pcolor':'limegreen',
		                   					 })
		# Na. [OIII]5007 Core VOFF
		mcmc_input['na_oiii5007_core_voff'] = ({'name':'na_oiii5007_core_voff',
		                   					 'label':'$\mathrm{VOFF}_{\mathrm{[OIII]5007\;Core}}$',
		                   					 'init':0.,
		                   					 'plim':(-1000.,1000.),
		                   					 'pcolor':'palegreen',
		                   					 })
		# Na. H-beta amplitude
		mcmc_input['na_Hb_core_amp'] = ({'name':'na_Hb_core_amp',
		                   		 	     'label':'$A_{\mathrm{Na.\;Hb}}$' ,
		                   		 	     'init':(hb_amp_init-total_flux_init) ,
		                   		 	     'plim':(1.0e-3,max_flux),
		                   		 	     'pcolor':'gold',
		                   		 	     })
		# Na. H-beta FWHM tied to [OIII]5007 FWHM
		# Na. H-beta VOFF
		mcmc_input['na_Hb_core_voff'] = ({'name':'na_Hb_core_voff',
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
		mcmc_input['na_oiii5007_outflow_amp'] = ({'name':'na_oiii5007_outflow_amp',
		                   					   'label':'$A_{\mathrm{[OIII]5007\;Outflow}}$' ,
		                   					   'init':(oiii5007_amp_init-total_flux_init)/2.,
		                   					   'plim':(1.0e-3,max_flux),
		                   					   'pcolor':'mediumpurple',
		                   					   })
		# Br. [OIII]5007 Outflow FWHM
		mcmc_input['na_oiii5007_outflow_fwhm'] = ({'name':'na_oiii5007_outflow_fwhm',
		                   						'label':'$\mathrm{FWHM}_{\mathrm{[OIII]5007\;Outflow}}$',
		                   						'init':450.,
		                   						'plim':(1.0e-3,2500.),
		                   						'pcolor':'darkorchid',
		                   						})
		# Br. [OIII]5007 Outflow VOFF
		mcmc_input['na_oiii5007_outflow_voff'] = ({'name':'na_oiii5007_outflow_voff',
		                   						'label':'$\mathrm{VOFF}_{\mathrm{[OIII]5007\;Outflow}}$',
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
		                   			'plim':(1.0e-3,max_flux),
		                   			'pcolor':'steelblue',
		                   			})
		# Br. H-beta FWHM
		mcmc_input['br_Hb_fwhm'] = ({'name':'br_Hb_fwhm',
		               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$',
		               	   			 'init':2500.,
		               	   			 'plim':(1.0e-3,10000.),
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
	if ((fit_narrow==True) and ((((fit_reg[0]+150.) <  6564.61< (fit_reg[1]-150.))==True))):
		print(' Fitting narrow Ha/[NII]/[SII] emission.')
		
		# If we aren't tying all narrow line widths, include the FWHM for H-alpha region
		if (tie_narrow==False):
			# Na. H-alpha FWHM
			mcmc_input['na_Ha_core_fwhm'] = ({'name':'na_Ha_core_fwhm',
		                   				   'label':'$\mathrm{FWHM}_{\mathrm{Na.\;Ha}}$',
		                   				   'init': 250,
		                   				   'plim':(1.0e-3,1000.),
		                   				   'pcolor':'limegreen',
		                   				   })
		# Na. H-alpha amplitude
		mcmc_input['na_Ha_core_amp'] = ({'name':'na_Ha_core_amp',
		                   		 	'label':'$A_{\mathrm{Na.\;Ha}}$' ,
		                   		 	'init':(ha_amp_init-total_flux_init) ,
		                   		 	'plim':(1.0e-3,max_flux),
		                   		 	'pcolor':'gold',
		                   		 	})
		# Na. H-alpha VOFF
		mcmc_input['na_Ha_core_voff'] = ({'name':'na_Ha_core_voff',
		                   			 'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Ha}}$',
		                   			 'init':0.,
		                   			 'plim':(-1000,1000.),
		                   			 'pcolor':'yellow',
		                   			 })
		# Na. [NII]6585 Amp.
		mcmc_input['nii6585_core_amp'] = ({'name':'nii6585_core_amp',
		                   				   'label':'$A_{\mathrm{[NII]6585\;Core}}$',
		                   				   'init':(ha_amp_init-total_flux_init)*0.75,
		                   				   'plim':(1.0e-3,max_flux),
		                   				   'pcolor':'green',
		                   				   })
		# Na. [SII]6718 Amp.
		mcmc_input['sii6718_core_amp'] = ({'name':'sii6718_core_amp',
		                   				   'label':'$A_{\mathrm{[SII]6718\;Core}}$',
		                   				   'init':(sii_amp_init-total_flux_init),
		                   				   'plim':(1.0e-3,max_flux),
		                   				   'pcolor':'green',
		                   				   })
		# Na. [SII]6732 Amp.
		mcmc_input['sii6732_core_amp'] = ({'name':'sii6732_core_amp',
		                   				   'label':'$A_{\mathrm{[SII]6732\;Core}}$',
		                   				   'init':(sii_amp_init-total_flux_init),
		                   				   'plim':(1.0e-3,max_flux),
		                   				   'pcolor':'green',
		                   				   })

	#### Ha/[NII]/[SII] Outflows ######################################################
	# As it turns out, the Type 1 H-alpha broad line severely limits the ability of outflows in this region to be 
	# fit, and lead to very inconsistent results.  In order to fit outflows in this region in the presence of broad
	# lines, you must constrain the outflow components using the [OIII]5007 outflow, which is much easier to detect.
	# Therefore, if we wish to fit outflows in H-alpha, one must include the H-beta/[OIII] region for constraints.
	# 
	# If you wish to fit H-alpha outflow independently, uncomment below:
	# if ((fit_narrow==True) and (fit_outflows==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
	# 	print(' Fitting Ha/[NII]/[SII] outflows.')
	# 	# Br. [OIII]5007 Outflow amplitude
	# 	mcmc_input['na_Ha_outflow_amp'] = ({'name':'na_Ha_outflow_amp',
	# 	                   					   'label':'$A_{\mathrm{Na.\;Ha\;Outflow}}$' ,
	# 	                   					   'init':(ha_amp_init-total_flux_init)*0.25,
	# 	                   					   'plim':(1.0e-3,max_flux),
	# 	                   					   'pcolor':'mediumpurple',
	# 	                   					   })
	# 	if (tie_narrow==False):
	# 		# Br. [OIII]5007 Outflow FWHM
	# 		mcmc_input['na_Ha_outflow_fwhm'] = ({'name':'na_Ha_outflow_fwhm',
	# 		                   						'label':'$\mathrm{FWHM}_{\mathrm{Na.\;Ha\;Outflow}}$',
	# 		                   						'init':450.,
	# 		                   						'plim':(1.0e-3,2500.),
	# 		                   						'pcolor':'darkorchid',
	# 		                   						})
	# 	# Br. [OIII]5007 Outflow VOFF
	# 	mcmc_input['na_Ha_outflow_voff'] = ({'name':'na_Ha_outflow_voff',
	# 	                   						'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Ha\;Outflow}}$',
	# 	                   						'init':-50.,
	# 	                   						'plim':(-2000.,2000.),
	# 	                   						'pcolor':'orchid',
	# 	                   						})
	#	# All components [NII]6585 of outflow are tied to all outflows of the Ha/[NII]/[SII] region

	##############################################################################

	#### Broad Line H-alpha ###########################################################

	if ((fit_broad==True) and ((((fit_reg[0]+150.) < 6564.61 < (fit_reg[1]-150.))==True))):
		print(' Fitting broadline H-alpha.')
		# Br. H-alpha amplitude
		mcmc_input['br_Ha_amp'] = ({'name':'br_Ha_amp',
		                   			'label':'$A_{\mathrm{Br.\;Ha}}$' ,
		                   			'init':(ha_amp_init-total_flux_init)/2.0  ,
		                   			'plim':(1.0e-3,max_flux),
		                   			'pcolor':'steelblue',
		                   			})
		# Br. H-alpha FWHM
		mcmc_input['br_Ha_fwhm'] = ({'name':'br_Ha_fwhm',
		               	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Ha}}$',
		               	   			 'init':2500.,
		               	   			 'plim':(1.0e-3,10000.),
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
	# Write to log 
	if (fit_type=='final'):
		write_log(mcmc_input,3,run_dir)
	##############################################################################
	
	# return param_names,param_labels,params,param_limits,param_init,param_colors
	return mcmc_input

##################################################################################

#### Outflow Test ################################################################
def outflow_test(lam_gal,galaxy,noise,run_dir,velscale,mcbs_niter):
	fit_reg = (4400,5800)
	param_dict = initialize_mcmc(lam_gal,galaxy,run_dir,fit_reg=fit_reg,fit_type='init',
	                             fit_feii=True,fit_losvd=True,
	                             fit_power=True,fit_broad=True,
	                             fit_narrow=True,fit_outflows=True)

	gal_temp = galaxy_template(lam_gal,age=5.0)
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

	write_log((rdict,sigma),'outflow_test',run_dir)

	# Determine the significance of outflows
	# Outflow criteria:
	#	(1) FWHM test: (FWHM_outflow - dFWHM_outflow) > (FWHM_core + dFWHM_core)
	cond1 = ((rdict['na_oiii5007_outflow_fwhm']['med']-rdict['na_oiii5007_outflow_fwhm']['std']) > (rdict['na_oiii5007_core_fwhm']['med']+rdict['na_oiii5007_core_fwhm']['std']))
	if (cond1==True):
		print('          Outflow FWHM condition: Pass')
	elif (cond1==False):
		print('          Outflow FWHM condition: Fail')
	#	(2) VOFF test: (VOFF_outflow + dVOFF_outflow) < (VOFF_core - dVOFF_core)
	cond2 = ((rdict['na_oiii5007_outflow_voff']['med']+rdict['na_oiii5007_outflow_voff']['std']) < (rdict['na_oiii5007_core_voff']['med']-rdict['na_oiii5007_core_voff']['std']))
	if (cond2==True):
		print('          Outflow VOFF condition: Pass')
	elif (cond2==False):
		print('          Outflow VOFF condition: Fail')
	#	(3) Amp. test: (AMP_outflow - dAMP_outflow) > sigma
	cond3 = ((rdict['na_oiii5007_outflow_amp']['med']-rdict['na_oiii5007_outflow_amp']['std']) > (3.0*sigma) )
	if (cond3==True):
		print('          Outflow amplitude condition: Pass')
	elif (cond3==False):
		print('          Outflow amplitude condition: Fail')

	if (all([cond1,cond2,cond3])==True):
		write_log((cond1,cond2,cond3),20,run_dir)
		return True,rdict,sigma
	elif (all([cond1,cond2,cond3])==False):
		write_log((cond1,cond2,cond3),21,run_dir)
		return False,rdict,sigma

##################################################################################


#### Maximum Likelihood Fitting ##################################################

def max_likelihood(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
				   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
				   fit_type='init',output_model=False,
				   monte_carlo=False,niter=25):
	
	# This function performs an initial maximum likelihood
	# estimation to acquire robust initial parameters.
	param_names  = [param_dict[key]['name'] for key in param_dict ]
	params       = [param_dict[key]['init'] for key in param_dict ]
	bounds       = [param_dict[key]['plim'] for key in param_dict ]

	# Constraints for Outflow components
	def oiii_amp_constraint(p):
		# (core_amp >= outflow_amp) OR (core_amp - outflow_amp >= 0)
	    return p[param_names.index('na_oiii5007_core_amp')]-p[param_names.index('na_oiii5007_outflow_amp')]

	def oiii_fwhm_constraint(p):
		# (outflow_fwhm >= core_fwhm) OR (outflow_fwhm - core_fwhm >= 0)
		return p[param_names.index('na_oiii5007_outflow_fwhm')]-p[param_names.index('na_oiii5007_core_fwhm')]
	def oiii_voff_constraint(p):
		# (core_voff >= outflow_voff) OR (core_voff - outflow_voff >= 0)
	    return p[param_names.index('na_oiii5007_core_voff')]-p[param_names.index('na_oiii5007_outflow_voff')]
	def nii_amp_constraint(p):
		# (core_amp >= outflow_amp) OR (core_amp - outflow_amp >= 0)
	    return p[param_names.index('na_Ha_core_amp')]-p[param_names.index('na_Ha_outflow_amp')]
	def nii_fwhm_constraint(p):
		# (outflow_fwhm >= core_fwhm) OR (outflow_fwhm - core_fwhm >= 0)
		return p[param_names.index('na_Ha_outflow_fwhm')]-p[param_names.index('na_Ha_core_fwhm')]
	def nii_voff_constraint(p):
		# (core_voff >= outflow_voff) OR (core_voff - outflow_voff >= 0)
	    return p[param_names.index('na_Ha_core_voff')]-p[param_names.index('na_Ha_outflow_voff')]

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
	# if model includes narrow lines and outflows

	# Model 1: [OIII]/Hb Region with outflows (excludes Ha/[NII]/[SII] region and outflows)
	if (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											 'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True) and \
	   (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   										 'nii6585_core_amp',
	   										 'sii6732_core_amp','sii6718_core_amp',
	   										 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==False):
		print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] region...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,output_model),\
		     				 method='SLSQP', bounds = bounds, constraints=cons1, options={'ftol':1.0e-6,'maxiter':1000,'disp': True})

	# Model 2: Ha/[NII]/[SII] region with outflows (excludes [OIII]/Hb region and outflows)
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==False) and \
	     (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   										   'nii6585_core_amp',
	   										   'sii6732_core_amp','sii6718_core_amp',
	   										   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		print('\n Performing max. likelihood fitting.  Constraining outflow components for Ha/[NII]/[SII] region...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,output_model),\
		     				 method='SLSQP', bounds = bounds, constraints=cons2, options={'ftol':1.0e-6,'maxiter':1000,'disp': True})

	# Model 3: Both [OIII]/Hb and Ha/[NII]/[SII] regions with outflows
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'nii6585_core_amp',
											   'sii6732_core_amp','sii6718_core_amp',
											   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] and Ha/[NII]/[SII] regions...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,output_model),\
		     				 method='SLSQP', bounds = bounds, constraints=cons3, options={'ftol':1.0e-6,'maxiter':1000,'disp': True})




	elif all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											  'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==False:
		print('\n Not fitting outflow components...\n ')
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(fun = nll, x0 = params, \
		     				 args=(param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		     				 	   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		     				 	   fit_type,output_model),\
		     				 method='SLSQP', bounds = bounds, options={'maxiter':1000,'disp': True})
	elap_time = (time.time() - start_time)

	###### Monte Carlo Simulations for Outflows ###############################################################

	if ((monte_carlo==True) ):
		
		par_best     = result['x']
		fit_type     = 'monte_carlo'
		output_model = False

		comp_dict = fit_model(par_best,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
							  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
							  fit_type,output_model)
	
		# Compute the total standard deviation at each pixel 
		# To take into account the uncertainty of the model, we compute the median absolute deviation of
		# the residuals from the initial fit, and add it in quadrature to the per-pixel standard deviation
		# of the sdss spectra.
		model_std = mad_std(comp_dict['residuals']['comp'])
	
		sigma = np.sqrt(mad_std(noise)**2 + model_std**2)

		model = comp_dict['model']['comp']

		mcpars = np.empty((len(par_best), niter)) # stores best-fit parameters of each MC iteration

		if (na_feii_temp is None) or (br_feii_temp is None):
			na_feii_temp = np.full_like(lam_gal,0)
			br_feii_temp = np.full_like(lam_gal,0)


		print( '\n Performing Monte Carlo bootstrap simulations  to determine if outflows are present...')
		print( '\n       Approximate time for %d iterations: %s \n' % (niter,time_convert(elap_time*niter))  )
		for n in range(0,niter,1):
			print('       Completed %d of %d iterations.' % (n+1,niter) )
			# Generate an array of random normally distributed data using sigma
			rand  = np.random.normal(0.0,sigma,len(lam_gal))
	
			mcgal = model + rand
			fit_type     = 'init'
			output_model = False

			if all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
													'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = params, \
	         				 		 args=(param_names,lam_gal,mcgal,noise,gal_temp,na_feii_temp,br_feii_temp,
	         				 	   		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
	         				 	   		   fit_type,output_model),\
	         				 		 method='SLSQP', bounds = bounds, constraints=cons1, options={'maxiter':500,'disp': False})

			elif all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
													  'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==False:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = params, \
	         				 		 args=(param_names,lam_gal,mcgal,noise,gal_temp,na_feii_temp,br_feii_temp,
	         				 	   		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
	         				 	   		   fit_type,output_model),\
	         				 		 method='SLSQP', bounds = bounds, options={'maxiter':500,'disp': False})
			mcpars[:,n] = resultmc['x']
	
			# For testing: plots every max. likelihood iteration
			if 0:
				output_model = True
				fit_model(resultmc['x'],param_names,lam_gal,mcgal,noise,gal_temp,na_feii_temp,br_feii_temp,
			  			  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
			  			  fit_type,output_model)
	
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
					              fit_type,output_model)
			plt.savefig(run_dir+'outflow_mcbs.pdf',fmt='pdf',dpi=150)
			# Close the plot
			plt.close()

		# Outflow determination and LOSVD for final model: 
		# determine if the object has outflows and if stellar velocity dispersion should be fit

		return pdict, sigma

	elif (monte_carlo==False):

		if 1: # Set to 1 to plot and stop
			output_model = True
			comp_dict = fit_model(result['x'],param_names,
					              lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
					              temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
					              fit_type,output_model)
		# Put best-fit params into dictionary
		pdict = {}
		for k in range(0,len(param_names),1):
			pdict[param_names[k]] = {'res':result['x'][k]}
		# Close the plot
		plt.close()

		# Get S/N of region (5050,5800) to determine if LOSVD/host-galaxy should be fit
		# If S/N < 10 do not fit
		if all(key in comp_dict for key in ['gal_temp','power'])==True:
			sn = np.median(comp_dict['gal_temp']['comp']+comp_dict['power']['comp'])/np.std(comp_dict['residuals']['comp'])
			print(' Signal-to-noise of host-galaxy continuum: %0.2f' % sn)
			return pdict, sn
		else:	
			sn = np.median(comp_dict['data']['comp'])/np.std(comp_dict['residuals']['comp'])
			print(' Signal-to-noise of object continuum: %0.2f' % sn)
			return pdict, sn

#### Likelihood function #########################################################

# Maximum Likelihood (initial fitting), Prior, and log Probability functions
def lnlike(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		   temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		   fit_type,output_model):

	# Create model
	model = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
					  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
					  fit_type,output_model)
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

	pdict = {}
	for k in range(0,len(param_names),1):
			pdict[param_names[k]] = {'p':params[k]}

	if (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											 'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True) and \
	   (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   										 'nii6585_core_amp',
	   										 'sii6732_core_amp','sii6718_core_amp',
	   										 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==False):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
		    (pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
		    (pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
		    (pdict['na_oiii5007_core_voff']['p'] >= pdict['na_oiii5007_outflow_voff']['p']):
		    return 0.0
		else: return -np.inf
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==False) and \
		   (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
		   										 'nii6585_core_amp',
		   										 'sii6732_core_amp','sii6718_core_amp',
		   										 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['na_Ha_core_amp']['p'] >= pdict['na_Ha_outflow_amp']['p']) & \
			(pdict['na_Ha_outflow_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']) & \
			(pdict['na_Ha_core_voff']['p'] >= pdict['na_Ha_outflow_voff']['p']):
			return 0.0
		else: return -np.inf
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
													  'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
													  'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
													  'nii6585_core_amp',
													  'sii6732_core_amp','sii6718_core_amp',
													  'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
		    (pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
		    (pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
		    (pdict['na_oiii5007_core_voff']['p'] >= pdict['na_oiii5007_outflow_voff']['p']) & \
		    (pdict['na_Ha_core_amp']['p'] >= pdict['na_Ha_outflow_amp']['p']) & \
			(pdict['na_Ha_outflow_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']) & \
			(pdict['na_Ha_core_voff']['p'] >= pdict['na_Ha_outflow_voff']['p']):
			return 0.0
		else: return -np.inf
	elif all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											  'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==False:
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
		output_model = False
		return lp + lnlike(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
		   					temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
		   					fit_type,output_model)

####################################################################################




#### Model Function ##############################################################

def fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
			  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
			  fit_type,output_model):


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
	c = 299792.458 # speed of light
	host_model = np.copy(galaxy)
	comp_dict  = {} 

	############################# Power-law Component ######################################################

	# if all(comp in param_names for comp in ['power_amp','power_slope','power_break'])==True:
	if all(comp in param_names for comp in ['power_amp','power_slope'])==True:

		# Create a template model for the power-law continuum
		# power = simple_power_law(lam_gal,p['power_amp'],p['power_slope'],p['power_break']) # 
		power = simple_power_law(lam_gal,p['power_amp'],p['power_slope']) # 

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

    #### [OII]3727,3729 #################################################################################
	if all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_fwhm','na_oii3727_core_voff','na_oii3729_core_amp'])==True:
		# Narrow [OII]3727
		na_oii3727_core_center        = 3727.092 # Angstroms
		na_oii3727_core_amp           = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm          = np.sqrt(p['na_oii3727_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oii3727_core_voff          = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core               = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model                    = host_model - na_oii3727_core
		comp_dict['na_oii3727_core']  = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729
		na_oii3729_core_center        = 3729.875 # Angstroms
		na_oii3729_core_amp           = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm          = na_oii3727_core_fwhm # km/s
		na_oii3729_core_voff          = na_oii3727_core_voff  # km/s
		na_oii3729_core               = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model                    = host_model - na_oii3729_core
		comp_dict['na_oii3729_core']  = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif (all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_voff','na_oii3729_core_amp','na_oiii5007_core_fwhm'])==True) & ('na_oii3727_core_fwhm' not in param_names):
		# Narrow [OII]3727
		na_oii3727_core_center        = 3727.092 # Angstroms
		na_oii3727_core_amp           = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm          = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oii3727_core_voff          = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core               = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model                    = host_model - na_oii3727_core
		comp_dict['na_oii3727_core']  = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729 
		na_oii3729_core_center        = 3729.875 # Angstroms
		na_oii3729_core_amp           = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm          = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oii3729_core_voff          = na_oii3727_core_voff  # km/s
		na_oii3729_core               = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model                    = host_model - na_oii3729_core
		comp_dict['na_oii3729_core'] = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### [NeIII]3870 #################################################################################
	if all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_fwhm','na_neiii_core_voff'])==True:
		# Narrow H-gamma
		na_neiii_core_center          = 3869.810 # Angstroms
		na_neiii_core_amp             = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm            = np.sqrt(p['na_neiii_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_neiii_core_voff            = p['na_neiii_core_voff']  # km/s
		na_neiii_core                 = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model                    = host_model - na_neiii_core
		comp_dict['na_neiii_core']   = {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif (all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_voff','na_oiii5007_core_fwhm'])==True) & ('na_neiii_core_fwhm' not in param_names):
		# Narrow H-gamma
		na_neiii_core_center          = 3869.810 # Angstroms
		na_neiii_core_amp             = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm            = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_neiii_core_voff            = p['na_neiii_core_voff']  # km/s
		na_neiii_core                 = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model                    = host_model - na_neiii_core
		comp_dict['na_neiii_core']    = {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### H-delta #####################################################################################
	if all(comp in param_names for comp in ['na_Hd_amp','na_Hd_fwhm','na_Hd_voff'])==True:
		# Narrow H-gamma
		na_hd_core_center             = 4102.890 # Angstroms
		na_hd_core_amp                = p['na_Hd_amp'] # flux units
		na_hd_core_fwhm               = np.sqrt(p['na_Hd_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hd_core_voff               = p['na_Hd_voff']  # km/s
		na_Hd_core                    = gaussian(lam_gal,na_hd_core_center,na_hd_core_amp,na_hd_core_fwhm,na_hd_core_voff,velscale)
		host_model                    = host_model - na_Hd_core
		comp_dict['na_Hd_core']       = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif (all(comp in param_names for comp in ['na_Hd_amp','na_Hd_voff','na_oiii5007_core_fwhm'])==True) & ('na_Hd_fwhm' not in param_names):
		# Narrow H-gamma
		na_hd_core_center             = 4102.890 # Angstroms
		na_hd_core_amp                = p['na_Hd_amp'] # flux units
		na_hd_core_fwhm               = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hd_core_voff               = p['na_Hd_voff']  # km/s
		na_Hd_core                    = gaussian(lam_gal,na_hd_core_center,na_hd_core_amp,na_hd_core_fwhm,na_hd_core_voff,velscale)
		host_model                    = host_model - na_Hd_core
		comp_dict['na_Hd_core']       = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### H-gamma/[OIII]4363 ##########################################################################
	if all(comp in param_names for comp in ['na_Hg_amp','na_Hg_fwhm','na_Hg_voff','na_oiii4363_core_amp','na_oiii4363_core_fwhm','na_oiii4363_core_voff'])==True:
		# Narrow H-gamma
		na_hg_core_center             = 4341.680 # Angstroms
		na_hg_core_amp                = p['na_Hg_amp'] # flux units
		na_hg_core_fwhm               = np.sqrt(p['na_Hg_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hg_core_voff               = p['na_Hg_voff']  # km/s
		na_Hg_core                    = gaussian(lam_gal,na_hg_core_center,na_hg_core_amp,na_hg_core_fwhm,na_hg_core_voff,velscale)
		host_model                    = host_model - na_Hg_core
		comp_dict['na_Hg_core']       = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_core_center       = 4364.436 # Angstroms
		na_oiii4363_core_amp          =  p['na_oiii4363_core_amp'] # flux units
		na_oiii4363_core_fwhm         =  np.sqrt(p['na_oiii4363_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii4363_core_voff         =  p['na_oiii4363_core_voff'] # km/s
		na_oiii4363_core              = gaussian(lam_gal,na_oiii4363_core_center,na_oiii4363_core_amp,na_oiii4363_core_fwhm,na_oiii4363_core_voff,velscale)
		host_model                    = host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	elif (all(comp in param_names for comp in ['na_Hg_amp','na_Hg_voff','na_oiii4363_core_amp','na_oiii4363_core_voff','na_oiii5007_core_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_Hg_fwhm','oiii4363_core_fwhm'])==True):
		# Narrow H-gamma
		na_hg_core_center             = 4341.680 # Angstroms
		na_hg_core_amp                = p['na_Hg_amp'] # flux units
		na_hg_core_fwhm               = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_hg_core_voff               = p['na_Hg_voff']  # km/s
		na_Hg_core                    = gaussian(lam_gal,na_hg_core_center,na_hg_core_amp,na_hg_core_fwhm,na_hg_core_voff,velscale)
		host_model                    = host_model - na_Hg_core
		comp_dict['na_Hg_core']       = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_core_center       = 4364.436 # Angstroms
		na_oiii4363_core_amp          =  p['na_oiii4363_core_amp'] # flux units
		na_oiii4363_core_fwhm         =  np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii4363_core_voff         =  p['na_oiii4363_core_voff'] # km/s
		na_oiii4363_core              = gaussian(lam_gal,na_oiii4363_core_center,na_oiii4363_core_amp,na_oiii4363_core_fwhm,na_oiii4363_core_voff,velscale)
		host_model                    = host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    #### H-beta/[OIII] #########################################################################################
	if all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											'na_Hb_core_amp','na_Hb_core_voff'])==True:
		# Narrow [OIII]5007 Core
		na_oiii5007_core_center       = 5008.240 # Angstroms
		na_oiii5007_core_amp          = p['na_oiii5007_core_amp'] # flux units
		na_oiii5007_core_fwhm         = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii5007_core_voff         = p['na_oiii5007_core_voff']  # km/s
		na_oiii5007_core              = gaussian(lam_gal,na_oiii5007_core_center,na_oiii5007_core_amp,na_oiii5007_core_fwhm,na_oiii5007_core_voff,velscale)
		host_model                    = host_model - na_oiii5007_core
		comp_dict['na_oiii5007_core'] = {'comp':na_oiii5007_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    	# Narrow [OIII]4959 Core
		na_oiii4959_core_center       = 4960.295 # Angstroms
		na_oiii4959_core_amp          = (1.0/3.0)*na_oiii5007_core_amp # flux units
		na_oiii4959_core_fwhm         = na_oiii5007_core_fwhm # km/s
		na_oiii4959_core_voff         = na_oiii5007_core_voff  # km/s
		na_oiii4959_core              = gaussian(lam_gal,na_oiii4959_core_center,na_oiii4959_core_amp,na_oiii4959_core_fwhm,na_oiii4959_core_voff,velscale)
		host_model                    = host_model - na_oiii4959_core
		comp_dict['na_oiii4959_core'] = {'comp':na_oiii4959_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow H-beta
		na_hb_core_center             = 4862.680 # Angstroms
		na_hb_core_amp                = p['na_Hb_core_amp'] # flux units
		na_hb_core_fwhm               = na_oiii5007_core_fwhm # km/s
		na_hb_core_voff               = p['na_Hb_core_voff']  # km/s
		na_Hb_core                    = gaussian(lam_gal,na_hb_core_center,na_hb_core_amp,na_hb_core_fwhm,na_hb_core_voff,velscale)
		host_model                    = host_model - na_Hb_core
		comp_dict['na_Hb_core']       = {'comp':na_Hb_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	#### H-alpha/[NII]/[SII] ####################################################################################
	if all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											'nii6585_core_amp',
											'sii6732_core_amp','sii6718_core_amp'])==True:
		# Narrow H-alpha
		na_ha_core_center             = 6564.610 # Angstroms
		na_ha_core_amp                = p['na_Ha_core_amp'] # flux units
		na_ha_core_fwhm               = np.sqrt(p['na_Ha_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_ha_core_voff               = p['na_Ha_core_voff']  # km/s
		na_Ha_core                    = gaussian(lam_gal,na_ha_core_center,na_ha_core_amp,na_ha_core_fwhm,na_ha_core_voff,velscale)
		host_model                    = host_model - na_Ha_core
		comp_dict['na_Ha_core']       = {'comp':na_Ha_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [NII]6585 Core
		na_nii6585_core_center 		  = 6585.270 # Angstroms
		na_nii6585_core_amp    		  = p['nii6585_core_amp'] # flux units
		na_nii6585_core_fwhm   		  = na_ha_core_fwhm
		na_nii6585_core_voff   		  = na_ha_core_voff
		na_nii6585_core   		      = gaussian(lam_gal,na_nii6585_core_center,na_nii6585_core_amp,na_nii6585_core_fwhm,na_nii6585_core_voff,velscale)
		host_model        		      = host_model - na_nii6585_core
		comp_dict['na_nii6585_core']  = {'comp':na_nii6585_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    	# Narrow [NII]6549 Core
		na_nii6549_core_center        = 6549.860 # Angstroms
		na_nii6549_core_amp           = (1.0/2.93)*na_nii6585_core_amp # flux units
		na_nii6549_core_fwhm          = na_ha_core_fwhm # km/s
		na_nii6549_core_voff          = na_ha_core_voff
		na_nii6549_core               = gaussian(lam_gal,na_nii6549_core_center,na_nii6549_core_amp,na_nii6549_core_fwhm,na_nii6549_core_voff,velscale)
		host_model                    = host_model - na_nii6549_core
		comp_dict['na_nii6549_core']  = {'comp':na_nii6549_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6718
		na_sii6718_core_center        = 6718.290 # Angstroms
		na_sii6718_core_amp           = p['sii6718_core_amp'] # flux units
		na_sii6718_core_fwhm          = na_ha_core_fwhm #na_sii6732_fwhm # km/s
		na_sii6718_core_voff          = na_ha_core_voff
		na_sii6718_core               = gaussian(lam_gal,na_sii6718_core_center,na_sii6718_core_amp,na_sii6718_core_fwhm,na_sii6718_core_voff,velscale)
		host_model                    = host_model - na_sii6718_core
		comp_dict['na_sii6718_core']  = {'comp':na_sii6718_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6732
		na_sii6732_core_center        = 6732.670 # Angstroms
		na_sii6732_core_amp           = p['sii6732_core_amp'] # flux units
		na_sii6732_core_fwhm          = na_ha_core_fwhm #np.sqrt(p['sii6732_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_sii6732_core_voff          = na_ha_core_voff
		na_sii6732_core               = gaussian(lam_gal,na_sii6732_core_center,na_sii6732_core_amp,na_sii6732_core_fwhm,na_sii6732_core_voff,velscale)
		host_model                    = host_model - na_sii6732_core
		comp_dict['na_sii6732_core']  = {'comp':na_sii6732_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}

	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_voff',
											   'nii6585_core_amp',
											   'sii6732_core_amp','sii6718_core_amp',
											   'na_oiii5007_core_fwhm'])==True) & ('na_Ha_core_fwhm' not in param_names):

		# If all narrow line widths are tied to [OIII]5007 FWHM...
		# Narrow H-alpha
		na_ha_core_center             = 6564.610 # Angstroms
		na_ha_core_amp                = p['na_Ha_core_amp'] # flux units
		na_ha_core_fwhm               = np.sqrt(p['na_oiii5007_core_fwhm']**2+(2.355*velscale)**2) # km/s
		na_ha_core_voff               = p['na_Ha_core_voff']  # km/s
		na_Ha_core                    = gaussian(lam_gal,na_ha_core_center,na_ha_core_amp,na_ha_core_fwhm,na_ha_core_voff,velscale)
		host_model                    = host_model - na_Ha_core
		comp_dict['na_Ha_core']       = {'comp':na_Ha_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [NII]6585 Core
		na_nii6585_core_center        = 6585.270 # Angstroms
		na_nii6585_core_amp           = p['nii6585_core_amp'] # flux units
		na_nii6585_core_fwhm          = na_ha_core_fwhm
		na_nii6585_core_voff          = na_ha_core_voff
		na_nii6585_core               = gaussian(lam_gal,na_nii6585_core_center,na_nii6585_core_amp,na_nii6585_core_fwhm,na_nii6585_core_voff,velscale)
		host_model                    = host_model - na_nii6585_core
		comp_dict['na_nii6585_core']  = {'comp':na_nii6585_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
    	# Narrow [NII]6549 Core
		na_nii6549_core_center        = 6549.860 # Angstroms
		na_nii6549_core_amp           = (1.0/2.93)*na_nii6585_core_amp # flux units
		na_nii6549_core_fwhm          = na_ha_core_fwhm
		na_nii6549_core_voff          = na_ha_core_voff
		na_nii6549_core               = gaussian(lam_gal,na_nii6549_core_center,na_nii6549_core_amp,na_nii6549_core_fwhm,na_nii6549_core_voff,velscale)
		host_model                    = host_model - na_nii6549_core
		comp_dict['na_nii6549_core']  = {'comp':na_nii6549_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6732
		na_sii6732_core_center        = 6732.670 # Angstroms
		na_sii6732_core_amp           = p['sii6732_core_amp'] # flux units
		na_sii6732_core_fwhm          = na_ha_core_fwhm
		na_sii6732_core_voff          = na_ha_core_voff
		na_sii6732_core               = gaussian(lam_gal,na_sii6732_core_center,na_sii6732_core_amp,na_sii6732_core_fwhm,na_sii6732_core_voff,velscale)
		host_model                    = host_model - na_sii6732_core
		comp_dict['na_sii6732_core']  = {'comp':na_sii6732_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6718
		na_sii6718_core_center        = 6718.290 # Angstroms
		na_sii6718_core_amp           = p['sii6718_core_amp'] # flux units
		na_sii6718_core_fwhm          = na_ha_core_fwhm
		na_sii6718_core_voff          = na_ha_core_voff
		na_sii6718_core               = gaussian(lam_gal,na_sii6718_core_center,na_sii6718_core_amp,na_sii6718_core_fwhm,na_sii6718_core_voff,velscale)
		host_model                    = host_model - na_sii6718_core
		comp_dict['na_sii6718_core']  = {'comp':na_sii6718_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	########################################################################################################

	# Outflow Components
    #### Hb/[OIII] outflows ################################################################################
	if (all(comp in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True):
		# Broad [OIII]5007 Outflow;
		na_oiii5007_outflow_center       = 5008.240 # Angstroms
		na_oiii5007_outflow_amp          = p['na_oiii5007_outflow_amp'] # flux units
		na_oiii5007_outflow_fwhm         = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(2.355*velscale)**2) # km/s
		na_oiii5007_outflow_voff         = p['na_oiii5007_outflow_voff']  # km/s
		na_oiii5007_outflow              = gaussian(lam_gal,na_oiii5007_outflow_center,na_oiii5007_outflow_amp,na_oiii5007_outflow_fwhm,na_oiii5007_outflow_voff,velscale)
		host_model                       = host_model - na_oiii5007_outflow
		comp_dict['na_oiii5007_outflow'] = {'comp':na_oiii5007_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
    	# Broad [OIII]4959 Outflow; 
		na_oiii4959_outflow_center       = 4960.295 # Angstroms
		na_oiii4959_outflow_amp          = na_oiii4959_core_amp*na_oiii5007_outflow_amp/na_oiii5007_core_amp # flux units
		na_oiii4959_outflow_fwhm         = na_oiii5007_outflow_fwhm # km/s
		na_oiii4959_outflow_voff         = na_oiii5007_outflow_voff  # km/s
		if (na_oiii4959_outflow_amp!=na_oiii4959_outflow_amp/1.0) or (na_oiii4959_outflow_amp==np.inf): na_oiii4959_outflow_amp=0.0
		na_oiii4959_outflow              = gaussian(lam_gal,na_oiii4959_outflow_center,na_oiii4959_outflow_amp,na_oiii4959_outflow_fwhm,na_oiii4959_outflow_voff,velscale)
		host_model                       = host_model - na_oiii4959_outflow
		comp_dict['na_oiii4959_outflow'] = {'comp':na_oiii4959_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# Broad H-beta Outflow; only a model, no free parameters, tied to [OIII]5007
		na_hb_outflow_amp                = na_hb_core_amp*na_oiii5007_outflow_amp/na_oiii5007_core_amp
		na_hb_outflow_fwhm               = na_oiii5007_outflow_fwhm # km/s
		na_hb_outflow_voff               = na_hb_core_voff+na_oiii5007_outflow_voff  # km/s
		if (na_hb_outflow_amp!=na_hb_outflow_amp/1.0) or (na_hb_outflow_amp==np.inf): na_hb_outflow_amp=0.0
		na_Hb_outflow                    = gaussian(lam_gal,na_hb_core_center,na_hb_outflow_amp,na_hb_outflow_fwhm,na_hb_outflow_voff,velscale)
		host_model                       = host_model - na_Hb_outflow
		comp_dict['na_Hb_outflow']       = {'comp':na_Hb_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	#### Ha/[NII]/[SII] outflows ###########################################################################
		# Outflows in H-alpha/[NII] are poorly constrained due to the presence of a broad line, therefore
		# we tie all outflows in this region together with the H-beta/[OIII] outflows.
	# 
	if (all(comp in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											 'na_Ha_core_amp','na_Ha_core_voff','nii6585_core_amp','sii6732_core_amp','sii6718_core_amp'])==True):
		# H-alpha Outflow; 
		na_ha_outflow_center             = 6564.610 # Angstroms
		na_ha_outflow_amp                = p['na_Ha_core_amp']*p['na_oiii5007_outflow_amp']/p['na_oiii5007_core_amp'] # flux units
		na_ha_outflow_fwhm               = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(2.355*velscale)**2) # km/s
		na_ha_outflow_voff               = p['na_oiii5007_outflow_voff']  # km/s  # km/s
		if (na_ha_outflow_amp!=na_ha_outflow_amp/1.0) or (na_ha_outflow_amp==np.inf): na_ha_outflow_amp=0.0
		na_Ha_outflow                    = gaussian(lam_gal,na_ha_outflow_center,na_ha_outflow_amp,na_ha_outflow_fwhm,na_ha_outflow_voff,velscale)
		host_model                       = host_model - na_Ha_outflow
		comp_dict['na_Ha_outflow']       = {'comp':na_Ha_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6585 Outflow;
		na_nii6585_outflow_center        = 6585.270 # Angstroms
		na_nii6585_outflow_amp           = na_nii6585_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6585_outflow_fwhm          = na_ha_outflow_fwhm
		na_nii6585_outflow_voff          = na_ha_outflow_voff
		if (na_nii6585_outflow_amp!=na_nii6585_outflow_amp/1.0) or (na_nii6585_outflow_amp==np.inf): na_nii6585_outflow_amp=0.0
		na_nii6585_outflow               = gaussian(lam_gal,na_nii6585_outflow_center,na_nii6585_outflow_amp,na_nii6585_outflow_fwhm,na_nii6585_outflow_voff,velscale)
		host_model                       = host_model - na_nii6585_outflow
		comp_dict['na_nii6585_outflow']  = {'comp':na_nii6585_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6549 Outflow; 
		na_nii6549_outflow_center        = 6549.860 # Angstroms
		na_nii6549_outflow_amp           = na_nii6549_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6549_outflow_fwhm          = na_ha_outflow_fwhm # km/s
		na_nii6549_outflow_voff          = na_ha_outflow_voff  # km/s
		if (na_nii6549_outflow_amp!=na_nii6549_outflow_amp/1.0) or (na_nii6549_outflow_amp==np.inf): na_nii6549_outflow_amp=0.0
		na_nii6549_outflow               = gaussian(lam_gal,na_nii6549_outflow_center,na_nii6549_outflow_amp,na_nii6549_outflow_fwhm,na_nii6549_outflow_voff,velscale)
		host_model                       = host_model - na_nii6549_outflow
		comp_dict['na_nii6549_outflow']  = {'comp':na_nii6549_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# Broad [SII]6718 Outflow; 
		na_sii6718_outflow_center        = 6718.290 # Angstroms
		na_sii6718_outflow_amp           = na_sii6718_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6718_outflow_fwhm          = na_ha_outflow_fwhm # km/s
		na_sii6718_outflow_voff          = na_ha_outflow_voff  # km/s
		if (na_sii6718_outflow_amp!=na_sii6718_outflow_amp/1.0) or (na_sii6718_outflow_amp==np.inf): na_sii6718_outflow_amp=0.0
		na_sii6718_outflow               = gaussian(lam_gal,na_sii6718_outflow_center,na_sii6718_outflow_amp,na_sii6718_outflow_fwhm,na_sii6718_outflow_voff,velscale)
		host_model                       = host_model - na_sii6718_outflow
		comp_dict['na_sii6718_outflow']  = {'comp':na_sii6718_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [SII]6732 Outflow; 
		na_sii6732_outflow_center        = 6732.670 # Angstroms
		na_sii6732_outflow_amp           = na_sii6732_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6732_outflow_fwhm          = na_ha_outflow_fwhm # km/s
		na_sii6732_outflow_voff          = na_ha_outflow_voff  # km/s
		if (na_sii6732_outflow_amp!=na_sii6732_outflow_amp/1.0) or (na_sii6732_outflow_amp==np.inf): na_sii6732_outflow_amp=0.0
		na_sii6732_outflow               = gaussian(lam_gal,na_sii6732_outflow_center,na_sii6732_outflow_amp,na_sii6732_outflow_fwhm,na_sii6732_outflow_voff,velscale)
		host_model                       = host_model - na_sii6732_outflow
		comp_dict['na_sii6732_outflow']  = {'comp':na_sii6732_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	elif (all(comp in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		# H-alpha Outflow; 
		na_ha_outflow_center             = 6564.610 # Angstroms
		na_ha_outflow_amp                = p['na_Ha_outflow_amp'] # flux units
		na_ha_outflow_fwhm               = np.sqrt(p['na_Ha_outflow_fwhm']**2+(2.355*velscale)**2) # km/s
		na_ha_outflow_voff               = p['na_Ha_outflow_voff']  # km/s  # km/s
		if (na_ha_outflow_amp!=na_ha_outflow_amp/1.0) or (na_ha_outflow_amp==np.inf): na_ha_outflow_amp=0.0
		na_Ha_outflow                    = gaussian(lam_gal,na_ha_outflow_center,na_ha_outflow_amp,na_ha_outflow_fwhm,na_ha_outflow_voff,velscale)
		host_model                       = host_model - na_Ha_outflow
		comp_dict['na_Ha_outflow']       = {'comp':na_Ha_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6585 Outflow;
		na_nii6585_outflow_center        = 6585.270 # Angstroms
		na_nii6585_outflow_amp           = na_nii6585_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6585_outflow_fwhm          = na_ha_outflow_fwhm
		na_nii6585_outflow_voff          = na_ha_outflow_voff
		if (na_nii6585_outflow_amp!=na_nii6585_outflow_amp/1.0) or (na_nii6585_outflow_amp==np.inf): na_nii6585_outflow_amp=0.0
		na_nii6585_outflow               = gaussian(lam_gal,na_nii6585_outflow_center,na_nii6585_outflow_amp,na_nii6585_outflow_fwhm,na_nii6585_outflow_voff,velscale)
		host_model                       = host_model - na_nii6585_outflow
		comp_dict['na_nii6585_outflow']  = {'comp':na_nii6585_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6549 Outflow; 
		na_nii6549_outflow_center        = 6549.860 # Angstroms
		na_nii6549_outflow_amp           = na_nii6549_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6549_outflow_fwhm          = na_ha_outflow_fwhm # km/s
		na_nii6549_outflow_voff          = na_ha_outflow_voff  # km/s
		if (na_nii6549_outflow_amp!=na_nii6549_outflow_amp/1.0) or (na_nii6549_outflow_amp==np.inf): na_nii6549_outflow_amp=0.0
		na_nii6549_outflow               = gaussian(lam_gal,na_nii6549_outflow_center,na_nii6549_outflow_amp,na_nii6549_outflow_fwhm,na_nii6549_outflow_voff,velscale)
		host_model                       = host_model - na_nii6549_outflow
		comp_dict['na_nii6549_outflow']  = {'comp':na_nii6549_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# Broad [SII]6718 Outflow; 
		na_sii6718_outflow_center        = 6718.290 # Angstroms
		na_sii6718_outflow_amp           = na_sii6718_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6718_outflow_fwhm          = na_ha_outflow_fwhm # km/s
		na_sii6718_outflow_voff          = na_ha_outflow_voff  # km/s
		if (na_sii6718_outflow_amp!=na_sii6718_outflow_amp/1.0) or (na_sii6718_outflow_amp==np.inf): na_sii6718_outflow_amp=0.0
		na_sii6718_outflow               = gaussian(lam_gal,na_sii6718_outflow_center,na_sii6718_outflow_amp,na_sii6718_outflow_fwhm,na_sii6718_outflow_voff,velscale)
		host_model                       = host_model - na_sii6718_outflow
		comp_dict['na_sii6718_outflow']  = {'comp':na_sii6718_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [SII]6732 Outflow; 
		na_sii6732_outflow_center        = 6732.670 # Angstroms
		na_sii6732_outflow_amp           = na_sii6732_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6732_outflow_fwhm          = na_ha_outflow_fwhm # km/s
		na_sii6732_outflow_voff          = na_ha_outflow_voff  # km/s
		if (na_sii6732_outflow_amp!=na_sii6732_outflow_amp/1.0) or (na_sii6732_outflow_amp==np.inf): na_sii6732_outflow_amp=0.0
		na_sii6732_outflow               = gaussian(lam_gal,na_sii6732_outflow_center,na_sii6732_outflow_amp,na_sii6732_outflow_fwhm,na_sii6732_outflow_voff,velscale)
		host_model                       = host_model - na_sii6732_outflow
		comp_dict['na_sii6732_outflow']  = {'comp':na_sii6732_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}


	########################################################################################################

	# Broad Lines
	#### Br. H-gamma #######################################################################################
	if all(comp in param_names for comp in ['br_Hg_amp','br_Hg_fwhm','br_Hg_voff'])==True:
		br_hg_center       = 4341.680 # Angstroms
		br_hg_amp          = p['br_Hg_amp'] # flux units
		br_hg_fwhm         = np.sqrt(p['br_Hg_fwhm']**2+(2.355*velscale)**2) # km/s
		br_hg_voff         = p['br_Hg_voff']  # km/s
		br_Hg              = gaussian(lam_gal,br_hg_center,br_hg_amp,br_hg_fwhm,br_hg_voff,velscale)
		host_model         = host_model - br_Hg
		comp_dict['br_Hg'] = {'comp':br_Hg,'pcolor':'xkcd:blue','linewidth':1.0}
	#### Br. H-beta ########################################################################################
	if all(comp in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])==True:
		br_hb_center       = 4862.68 # Angstroms
		br_hb_amp          = p['br_Hb_amp'] # flux units
		br_hb_fwhm         = np.sqrt(p['br_Hb_fwhm']**2+(2.355*velscale)**2) # km/s
		br_hb_voff         = p['br_Hb_voff']  # km/s
		br_Hb              = gaussian(lam_gal,br_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)
		host_model         = host_model - br_Hb
		comp_dict['br_Hb'] = {'comp':br_Hb,'pcolor':'xkcd:blue','linewidth':1.0}
 
	#### Br. H-alpha #######################################################################################
	if all(comp in param_names for comp in ['br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True:
		br_ha_center       = 6564.610 # Angstroms
		br_ha_amp          = p['br_Ha_amp'] # flux units
		br_ha_fwhm         = np.sqrt(p['br_Ha_fwhm']**2+(2.355*velscale)**2) # km/s
		br_ha_voff         = p['br_Ha_voff']  # km/s
		br_Ha              = gaussian(lam_gal,br_ha_center,br_ha_amp,br_ha_fwhm,br_ha_voff,velscale)
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
		weights     = nnls(conv_temp,host_model) # scipy.optimize Non-negative Least Squares
		host_galaxy = (np.sum(weights*conv_temp,axis=1)) 
		comp_dict['host_galaxy'] = {'comp':host_galaxy,'pcolor':'xkcd:lime green','linewidth':1.0}

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
	
	##################################################################################

	if (fit_type=='init') and (output_model==False): # For max. likelihood fitting
		return gmodel
	if (fit_type=='init') and (output_model==True): # For max. likelihood fitting
		return comp_dict
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
def galaxy_template(lam,age=5.0):
	ages = [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]

	if ((age<0.1) or (age>15.0)):
		print('\n You must choose an age between (1 Gyr <= age <= 15 Gyr)! Using 5.0 Gyr template instead... \n')
		fname = 'M09_'+str(age)+'.csv'
		df = pd.read_csv('badass_data_files/M09_ssp_templates/'+fname,skiprows=5,sep=',',names=['t_Gyr','Z','lam','flam'])
		wave = np.array(df['lam'])
		flux = np.array(df['flam'])
		flux = flux/flux[wave==5500.] # Normalize to 1.0 at 5500 A
		# Interpolate the template 
		gal_interp = interp1d(wave,flux,kind='cubic',bounds_error=False,fill_value=(0,0))
		gal_temp = gal_interp(lam)
		# Normalize by median
		gal_temp = gal_temp/np.median(gal_temp)
		return gal_temp
	elif ((age>=0.1) and (age<=15.0)):
		# Get nearest user-input age
		age, aidx =find_nearest(ages,age)
		print('\n Using Maraston et al. 2009 %0.1f Gyr template... \n ' % age)
		fname = 'M09_'+str(age)+'.csv'
		df = pd.read_csv('badass_data_files/M09_ssp_templates/'+fname,skiprows=5,sep=',',names=['t_Gyr','Z','lam','flam'])
		wave = np.array(df['lam'])
		flux = np.array(df['flam'])
		flux = flux/flux[wave==5500.] # Normalize to 1.0 at 5500 A
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
	na_feii_table = pd.read_csv('badass_data_files/feii_templates/na_feii_template.csv')
	br_feii_table = pd.read_csv('badass_data_files/feii_templates/br_feii_template.csv')

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

def simple_power_law(x,amp,alpha):#,xb)
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
		    location of break in the power-law (angstroms)

	Returns
	----------
	C     : array
		    AGN continuum model the same length as x
	"""
	xb = np.max(x)-(0.5*(np.max(x)-np.min(x))) # take to be half of the wavelength range

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

def run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
              auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,burn_in,min_iter,max_iter,threads):
	# Keep original burn_in and max_iter to reset convergence if jumps out of convergence
	orig_burn_in  = burn_in
	orig_max_iter = max_iter
	# Sorted parameter names
	param_names = np.array(param_names)
	i_sort = np.argsort(param_names) # this array gives the ordered indices of parameter names (alphabetical)
	# Create MCMC_chain.csv if it doesn't exist
	if os.path.exists(run_dir+'log/MCMC_chain.csv')==False:
		f = open(run_dir+'log/MCMC_chain.csv','w')
		param_string = ', '.join(str(e) for e in param_names)
		f.write('# iter, ' + param_string) # Write initial parameters
		best_str = ', '.join(str(e) for e in init_params)
		f.write('\n 0, '+best_str)
		f.close()


	# initialize the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args,threads=threads)

	start_time = time.time() # start timer

	write_log((ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter,threads),40,run_dir)

	# Initialize stuff for autocorrelation analysis
	if (auto_stop==True):
		autocorr_chain = []
		if (isinstance(conv_type,tuple)==True):
			old_tau = np.full(len(param_names),np.inf)
		elif (conv_type == 'mean') or (conv_type == 'median') or (conv_type == 'all'):
			old_tau = np.full(len(param_names),np.inf)
		min_samp     = min_samp # minimum iterations to use past convergence
		ncor_times   = ncor_times # multiplicative tolerance; number of correlation times before which we stop sampling	
		autocorr_tol = autocorr_tol	
		stop_iter    = max_iter # stopping iteration; changes once convergence is reached
		converged  = False

	# Check auto_stop convergence type:
	if (auto_stop==True) and (isinstance(conv_type,tuple)==True) :
		if all(elem in param_names  for elem in conv_type)==True:
			print('\n Only considering convergence of following parameters: ')
			for c in conv_type:	
				print('          %s' % c)
			pass
		elif all(elem in param_names  for elem in conv_type)==False:
			conv_type_list = list(conv_type)
			if ('na_oiii5007_outflow_amp' in conv_type):
				conv_type_list.remove('na_oiii5007_outflow_amp')
			if ('na_oiii5007_outflow_fwhm' in conv_type):
				conv_type_list.remove('na_oiii5007_outflow_fwhm')
			if ('na_oiii5007_outflow_voff' in conv_type):
				conv_type_list.remove('na_oiii5007_outflow_voff')
			if all(elem in param_names  for elem in conv_type_list)==True:
				conv_type = tuple(conv_type_list)
				print('\n Only considering convergence of following parameters: ')
				for c in conv_type_list:	
					print('          %s' % c)
				pass
			else:
				raise ValueError('\n One of more paramters in conv_type is not a valid parameter.\n')
		write_log((min_samp,autocorr_tol,ncor_times,conv_type),41,run_dir)
	# Run emcee
	for k, result in enumerate(sampler.sample(pos, iterations=max_iter)):#,storechain=True)):
		if ((k+1) % write_iter == 0) and ((k+1)>=write_thresh) and ((k+1)<min_iter): 
			print('\nIteration = %d' % (k+1))
		if ((k+1) % write_iter == 0) and ((k+1)>=write_thresh) and ((k+1)>=min_iter) and (auto_stop==False):
			print('\nIteration = %d' % (k+1))
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
			f = open(run_dir+'log/MCMC_chain.csv','a')
			best_str = ', '.join(str(e) for e in best)
			f.write('\n'+str(k+1)+', '+best_str)
			f.close()
		# Checking autocorrelation times for convergence
		if ((k+1) % write_iter == 0) and ((k+1)>=min_iter) and ((k+1)>=write_thresh) and (auto_stop==True):
			# Autocorrelation analysis of chain to determine convergence; the minimum autocorrelation time is 1.0, which results when a time cannot be accurately calculated.
			tau = autocorr_convergence(sampler.chain,param_names,plot=False) # Calculate autocorrelation times for each parameter
			autocorr_chain.append(tau) # 
			# Calculate tolerances
			tol = (np.abs(tau-old_tau)/old_tau) * 100.0

			# If convergence for mean autocorrelation time 
			if (conv_type == 'mean'):
				par_conv = [] # converged parameter indices
				par_not_conv  = [] # non-converged parameter indices
				for x in range(0,len(param_names),1):
					if (round(tau[x],1)>1.0):# & (0.0<round(tol[x],1)<autocorr_tol):
						par_conv.append(x) # Append index of parameter for which an autocorrelation time can be calculated; we use these to calculate the mean
					else: par_not_conv.append(x)
				# Calculate mean of parameters for which an autocorrelation time could be calculated
				par_conv = np.array(par_conv) # Explicitly convert to array
				par_not_conv = np.array(par_not_conv) # Explicitly convert to array

				if (par_conv.size == 0) and (stop_iter == orig_max_iter):
					print('\nIteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Not enough iterations for any autocorrelation times!')
				elif ( (par_conv.size > 0) and (k+1)>(np.mean(tau[par_conv]) * ncor_times) and (np.mean(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.              | ' % (k+1))
						print(' | Performing %d iterations of sampling... | ' % min_samp )
						print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
						burn_in = (k+1)
						stop_iter = (k+1)+min_samp
						conv_tau = tau
						converged = True
				elif ((par_conv.size == 0) or ( (k+1)<(np.mean(tau[par_conv]) * ncor_times)) or (np.mean(tol[par_conv])>autocorr_tol)) and (stop_iter < orig_max_iter):
					print('\nIteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Jumped out of convergence! Resetting convergence criteria...')
					# Reset convergence criteria
					print('- Resetting burn_in = %d' % orig_burn_in)
					burn_in = orig_burn_in
					print('- Resetting max_iter = %d' % orig_max_iter)
					stop_iter = orig_max_iter
					converged = False

				if (par_conv.size>0):
					pnames_sorted = param_names[i_sort]
					tau_sorted    = tau[i_sort]
					tol_sorted    = tol[i_sort]
					print('{0:<30}{1:<40}{2:<30}'.format('\nIteration = %d' % (k+1),'%d x Mean Autocorr. Time = %0.2f' % (ncor_times,np.mean(tau[par_conv]) * ncor_times),'Mean Tolerance = %0.2f' % np.mean(tol[par_conv])))
					print('---------------------------------------------------------------------------------------------------')
					print('{0:<30}{1:<40}{2:<30}'.format('Parameter','Autocorrelation Time','Tolerance'))
					print('---------------------------------------------------------------------------------------------------')
					for i in range(0,len(pnames_sorted),1):
						if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
							print('{0:<30}{1:<40.5f}{2:<30.5f}'.format(pnames_sorted[i],tau_sorted[i],tol_sorted[i]))
						else: 
							print('{0:<30}{1:<40}{2:<30}'.format(pnames_sorted[i],' -------- ' ,' -------- '))
					print('---------------------------------------------------------------------------------------------------')

			# If convergence for median autocorrelation time 
			if (conv_type == 'median'):
				par_conv = [] # converged parameter indices
				par_not_conv  = [] # non-converged parameter indices
				for x in range(0,len(param_names),1):
					if (round(tau[x],1)>1.0):# & (tol[x]<autocorr_tol):
						par_conv.append(x) # Append index of parameter for which an autocorrelation time can be calculated; we use these to calculate the mean
					else: par_not_conv.append(x)
				# Calculate mean of parameters for which an autocorrelation time could be calculated
				par_conv = np.array(par_conv) # Explicitly convert to array
				par_not_conv = np.array(par_not_conv) # Explicitly convert to array

				if (par_conv.size == 0) and (stop_iter == orig_max_iter):
					print('\nIteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Not enough iterations for any autocorrelation times!')
				elif ( (par_conv.size > 0) and (k+1)>(np.median(tau[par_conv]) * ncor_times) and (np.median(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.              |' % (k+1))
						print(' | Performing %d iterations of sampling... |' % min_samp )
						print(' | Sampling will finish at %d iterations.  |' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
						burn_in = (k+1)
						stop_iter = (k+1)+min_samp
						conv_tau = tau
						converged = True
				elif ((par_conv.size == 0) or ( (k+1)<(np.median(tau[par_conv]) * ncor_times)) or (np.median(tol[par_conv])>autocorr_tol)) and (stop_iter < orig_max_iter):
					print('\nIteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Jumped out of convergence! Resetting convergence criteria...')
					# Reset convergence criteria
					print('- Resetting burn_in = %d' % orig_burn_in)
					burn_in = orig_burn_in
					print('- Resetting max_iter = %d' % orig_max_iter)
					stop_iter = orig_max_iter
					converged = False

				if (par_conv.size>0):
					pnames_sorted = param_names[i_sort]
					tau_sorted    = tau[i_sort]
					tol_sorted    = tol[i_sort]
					print('{0:<30}{1:<40}{2:<30}'.format('\nIteration = %d' % (k+1),'%d x Median Autocorr. Time = %0.2f' % (ncor_times,np.median(tau[par_conv]) * ncor_times),'Med. Tolerance = %0.2f' % np.median(tol[par_conv])))
					print('---------------------------------------------------------------------------------------------------')
					print('{0:<30}{1:<40}{2:<30}'.format('Parameter','Autocorrelation Time','Tolerance'))
					print('---------------------------------------------------------------------------------------------------')
					for i in range(0,len(pnames_sorted),1):
						if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):	
							print('{0:<30}{1:<40.5f}{2:<30.5f}'.format(pnames_sorted[i],tau_sorted[i],tol_sorted[i]))
						else: 
							print('{0:<30}{1:<40}{2:<30}'.format(pnames_sorted[i],' -------- ' ,' -------- '))
					print('---------------------------------------------------------------------------------------------------')	
				
			# If convergence for ALL autocorrelation times 
			if (conv_type == 'all'):
				if ( all( (x==1.0) for x in tau) ) and (stop_iter == orig_max_iter):
					print('\nIteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Not enough iterations for any autocorrelation times!')
				elif all( ((k+1)>(x * ncor_times)) for x in tau) and all( (x>1.0) for x in tau) and all(y<autocorr_tol for y in tol) and (stop_iter == max_iter):
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.              | ' % (k+1))
						print(' | Performing %d iterations of sampling... | ' % min_samp )
						print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
						burn_in = (k+1)
						stop_iter = (k+1)+min_samp
						conv_tau = tau
						converged = True
				elif (any( ((k+1)<(x * ncor_times)) for x in tau) or any( (x==1.0) for x in tau) or any(y>autocorr_tol for y in tol)) and (stop_iter < orig_max_iter):
					print('\n Iteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Jumped out of convergence! Resetting convergence criteria...')
					# Reset convergence criteria
					print('- Resetting burn_in = %d' % orig_burn_in)
					burn_in = orig_burn_in
					print('- Resetting max_iter = %d' % orig_max_iter)
					stop_iter = orig_max_iter
					converged = False
				if 1:
					pnames_sorted = param_names[i_sort]
					tau_sorted    = tau[i_sort]
					tol_sorted    = tol[i_sort]
					print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('\nIteration = %d' % (k+1),' ',' ',' ',' '))
					print('-----------------------------------------------------------------------------------------------------------------------------------')
					print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('Parameter','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
					print('-----------------------------------------------------------------------------------------------------------------------------------')
					for i in range(0,len(pnames_sorted),1):
						if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol)):
							conv_bool = True
						else: conv_bool = False
						if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
							print('{0:<30}{1:<30.5f}{2:<30.5f}{3:<30.5f}{4:<30}'.format(pnames_sorted[i],tau_sorted[i],tau_sorted[i]*ncor_times,tol_sorted[i],str(conv_bool)))
						else: 
							print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format(pnames_sorted[i],' -------- ' ,' -------- ',' -------- ',' -------- '))
					print('-----------------------------------------------------------------------------------------------------------------------------------')


			# If convergence for a specific set of parameters
			elif (isinstance(conv_type,tuple)==True):
				# Get indices of parameters for which we want to converge; these will be the only ones we care about
				par_ind = np.array([i for i, item in enumerate(param_names) if item in set(conv_type)])
				# Get list of parameters, autocorrelation times, and tolerances for the ones we care about
				param_interest   = param_names[par_ind]
				tau_interest = tau[par_ind]
				tol_interest = tol[par_ind]
				# New sort for selected parameters
				i_sort = np.argsort(param_interest) # this array gives the ordered indices of parameter names (alphabetical)
				if ( all( (x==1.0) for x in tau_interest) ) and (stop_iter == orig_max_iter):
					print('\nIteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Not enough iterations for any autocorrelation times!')
				elif all( ((k+1)>(x * ncor_times)) for x in tau_interest) and all( (x>1.0) for x in tau_interest) and all(y<autocorr_tol for y in tol_interest) and (stop_iter == max_iter):
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.              | ' % (k+1))
						print(' | Performing %d iterations of sampling... | ' % min_samp )
						print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
						burn_in = (k+1)
						stop_iter = (k+1)+min_samp
						conv_tau = tau
						converged = True
				elif (any( ((k+1)<(x * ncor_times)) for x in tau_interest) or any( (x==1.0) for x in tau_interest) or any(y>autocorr_tol for y in tol_interest)) and (stop_iter < orig_max_iter):
					print('\n Iteration = %d' % (k+1))
					print('-------------------------------------------------------------------------------')
					print('- Jumped out of convergence! Resetting convergence criteria...')
					# Reset convergence criteria
					print('- Resetting burn_in = %d' % orig_burn_in)
					burn_in = orig_burn_in
					print('- Resetting max_iter = %d' % orig_max_iter)
					stop_iter = orig_max_iter
					converged = False
				if 1:
					pnames_sorted = param_interest[i_sort]
					tau_sorted    = tau_interest[i_sort]
					tol_sorted    = tol_interest[i_sort]
					print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('\nIteration = %d' % (k+1),' ',' ',' ',' '))
					print('-----------------------------------------------------------------------------------------------------------------------------------')
					print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('Parameter','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
					print('-----------------------------------------------------------------------------------------------------------------------------------')
					for i in range(0,len(pnames_sorted),1):
						if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol)):
							conv_bool = True
						else: conv_bool = False
						if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
							print('{0:<30}{1:<30.5f}{2:<30.5f}{3:<30.5f}{4:<30}'.format(pnames_sorted[i],tau_sorted[i],tau_sorted[i]*ncor_times,tol_sorted[i],str(conv_bool)))
						else: 
							print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format(pnames_sorted[i],' -------- ' ,' -------- ',' -------- ',' -------- '))
					print('-----------------------------------------------------------------------------------------------------------------------------------')

				

			# Stop
			if ((k+1) == stop_iter):
				break

			old_tau = tau	

	elap_time = (time.time() - start_time)	   
	run_time = time_convert(elap_time)
	print("\n ecmee Runtime = %s. \n" % (run_time))
	if (auto_stop==True):
		# Write autocorrelation chain to log 
		np.save(run_dir+'/log/autocorrelation_chain',autocorr_chain)
	# Write to log file
	if (auto_stop==True):
		if (converged == True):
			write_log((burn_in,stop_iter,param_names,conv_tau,autocorr_tol,tol,ncor_times),42,run_dir)
		elif (converged == False):
			unconv_tol = (np.abs((old_tau) - (tau)) / (tau))
			write_log((burn_in,stop_iter,param_names,tau,autocorr_tol,unconv_tol,ncor_times),42,run_dir)
	write_log(run_time,43,run_dir) 

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

	# Collect garbage
	del sampler
	del lnprob_args
	gc.collect()

	return a, burn_in #, autocorr


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
	
	# Collect garbage
	del emcee_chain
	gc.collect()
	    
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
    # Get the max (most probable) value of the interpolated function
    pdfmax = xnew[np.where(np.isclose(pdfnew,np.max(pdfnew)))][0]
    # Get x-values to the left of the maximum 
    xh_left = xnew[np.where(xnew < pdfmax)]
    yh_left = pdfnew[np.where(xnew < pdfmax)]

    # Get x-values to the right of the maximum 
    xh_right = xnew[np.where(xnew > pdfmax)]
    yh_right = pdfnew[np.where(xnew > pdfmax)]
    try:
#     if 1:
        for l in range(0,len(yh_right),1):
            idx = find_nearest(yh_left,yh_right[l])
            xvec = xnew[[i for i,j in enumerate(xnew) if xh_left[idx]<=j<=xh_right[l]]] # x vector for simps
            yvec = pdfnew[[i for i,j in enumerate(xnew) if xh_left[idx]<=j<=xh_right[l]]] # y vector for simps
            integral = simps(y=yvec)#,x=xvec)
            if round(integral,2) == 0.68: #68.0/100.0:
                # 1 sigma = 68% confidence interval
                conf_interval_1 = [pdfmax - np.min(xvec),np.max(xvec) - pdfmax]
                
        return pdfmax,conf_interval_1[0],conf_interval_1[1],xvec,yvec*scale_factor

    except: 
        med = np.median(flat)
        std = np.std(flat)
        return med,std,std,x,np.zeros(len(prob))


def get_bin_centers(bins):
        bins = bins[:-1]
        bin_width = bins[1]-bins[0]
        new_bins =  bins + bin_width/2.0
        return new_bins


def param_plots(param_dict,burn_in,run_dir,save_plots=True):
	# Create a histograms sub-folder
	if (os.path.exists(run_dir + 'histogram_plots')==False):
		os.mkdir(run_dir + 'histogram_plots')
		os.mkdir(run_dir + 'histogram_plots/param_histograms')

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])

	for key in param_dict:
		ax1.clear()
		ax2.clear()
		ax3.clear()

		print('          %s' % key)
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
		# Calculate median and median absolute deviation of walkers at each iteration; we have depreciated
		# the average and standard deviation because they do not behave well for outlier walkers, which
		# also don't agree with histograms.
		c_med = np.median(chain,axis=0)
		c_madstd = mad_std(chain)
		ax3.plot(range(np.shape(chain)[1]),c_med,color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
		ax3.fill_between(range(np.shape(chain)[1]),c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in')
		ax3.set_xlim(0,np.shape(chain)[1])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'%s' % pname,fontsize=12)
		ax3.legend(loc='upper left')

		# Save the figure
		if (save_plots==True):
			figname = param_dict[key]['name']
			plt.savefig(run_dir+'histogram_plots/param_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=300,fmt='png')
			
	# Close plot window
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del ax3
	del flat
	gc.collect()

	return param_dict


def emline_flux_plots(burn_in,nwalkers,run_dir,save_plots=True):
	# Create a histograms sub-folder
	if (os.path.exists(run_dir + 'histogram_plots/flux_histograms')==False):
		os.mkdir(run_dir + 'histogram_plots/flux_histograms')

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
		ax1.clear()
		ax2.clear()
		ax3.clear()

		print('          %s' % key)
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
		c_med = np.median(chain,axis=0)
		c_madstd = mad_std(chain)
		ax3.plot(range(np.shape(chain)[0])[burn_in:],np.full(np.shape(chain[burn_in:])[0],c_med),color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
		ax3.fill_between(range(np.shape(chain)[0])[burn_in:],c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in')

		ax3.set_xlim(0,np.shape(chain)[0])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'%s' % key,fontsize=8)
		ax3.legend(loc='upper left')

		# Save the figure
		if (save_plots==True):
			figname = key
			plt.savefig(run_dir+'histogram_plots/flux_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=150,fmt='png')
		
	# Close plot
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del ax3
	del flat
	gc.collect()

	return flux_dict

def flux2lum(flux_dict,burn_in,nwalkers,z,run_dir,H0=71.0,Om0=0.27,save_plots=True):
	# Create a histograms sub-folder
	if (os.path.exists(run_dir + 'histogram_plots/lum_histograms')==False):
		os.mkdir(run_dir + 'histogram_plots/lum_histograms')
   
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
		ax1.clear()
		ax2.clear()
		ax3.clear()

		print('          %s' % key)
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
		c_med = np.median(chain,axis=0)
		c_madstd = mad_std(chain)
		ax3.plot(range(np.shape(chain)[0])[burn_in:],np.full(np.shape(chain[burn_in:])[0],c_med),color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
		ax3.fill_between(range(np.shape(chain)[0])[burn_in:],c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in')

		ax3.set_xlim(0,np.shape(chain)[0])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'%s' % key,fontsize=8)
		ax3.legend(loc='upper left')

		# Save the figure
		if (save_plots==True):
			figname = key
			plt.savefig(run_dir+'histogram_plots/lum_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=150,fmt='png')
		
	# Close plot	
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del ax3
	del flat
	del flux_dict
	gc.collect()

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
	if 0: 
	    for i in range(0,len(par_names),1):
	        print(par_names[i],par_best[i],sig_low[i],sig_upp[i])
	# Write best-fit paramters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit', format='E', array=par_best)
	col3 = fits.Column(name='sigma_low', format='E', array=sig_low)
	col4 = fits.Column(name='sigma_upp', format='E', array=sig_upp)
	cols = fits.ColDefs([col1,col2,col3,col4])
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'par_table.fits',overwrite=True)

	# Write full param dict to log file
	write_log((par_names,par_best,sig_low,sig_upp),50,run_dir)
	# Collect garbage
	del param_dict
	del flux_dict
	del lum_dict
	gc.collect()
	return None

def write_chain(param_dict,flux_dict,lum_dict,run_dir):
	# Save paramter dict as a npy file
	np.save(run_dir + '/log/param_dict.npy',param_dict)

	cols = []
	# Construct a column for each parameter and chain
	for key in param_dict:
		cols.append(fits.Column(name=key, format='E',array=param_dict[key]['chain'].flat))
	for key in flux_dict:
		cols.append(fits.Column(name=key, format='E', array=flux_dict[key]['chain']))
	for key in lum_dict:
		cols.append(fits.Column(name=key, format='E', array=lum_dict[key]['chain']))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'MCMC_chains.fits',overwrite=True)
	# Collect garbage
	del param_dict
	del flux_dict
	del lum_dict
	gc.collect()
	return None
    
def plot_best_model(param_dict,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
                           temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir):

	param_names  = [param_dict[key]['name'] for key in param_dict ]
	par_best       = [param_dict[key]['par_best'] for key in param_dict ]


	output_model = True
	fit_type     = 'final'
	comp_dict = fit_model(par_best,param_names,lam_gal,galaxy,noise,gal_temp,na_feii_temp,br_feii_temp,
			  temp_list,temp_fft,npad,velscale,npix,vsyst,run_dir,
			  fit_type,output_model)

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
		if (key not in ['residuals','noise','wave','data','model','na_feii_template','br_feii_template','host_galaxy','power']):
			ax1.axvline(lam_gal[np.where(comp_dict[key]['comp']==np.max(comp_dict[key]['comp']))[0][0]],color='xkcd:white',linestyle='--',linewidth=0.5)
			ax2.axvline(lam_gal[np.where(comp_dict[key]['comp']==np.max(comp_dict[key]['comp']))[0][0]],color='xkcd:white',linestyle='--',linewidth=0.5)

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

	# Close plot
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del param_dict
	gc.collect()

	return None


# Clean-up Routine
##################################################################################

def cleanup(run_dir):
	# Remove emission line fluxes csv 
	os.remove(run_dir + 'em_line_fluxes.csv')
	# Remove param_plots folder if empty
	if not os.listdir(run_dir + 'histogram_plots'):
		shutil.rmtree(run_dir + 'histogram_plots')
	return None

##################################################################################

def write_log(output_val,output_type,run_dir):
	"""
	This function writes values to a log file as the code runs.
	"""
	# Check if log folder has been created, if not, create it
	if os.path.exists(run_dir+'/log/')==False:
		os.mkdir(run_dir+'/log/')
		# Create log file 
		logfile = open(run_dir+'log/log_file.txt','a')
		s = """
		\n############################### BADASS v.6.0 LOGFILE ###############################
		"""
		logfile.write(s)
		logfile.close()


	# sdss_prepare
	# output_val=(file,ra,dec,z,fit_min,fit_max,velscale,ebv), output_type=0
	if (output_type==0):
		file,ra,dec,z,fit_min,fit_max,velscale,ebv = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		s = """
\n 
file: '%s'
(RA, DEC): (%s, %s)
redshift: %0.5f
fitting region: (%0.1f,%0.1f)
velocity scale: %0.3f   (km/s/pixel)
galactic E(B-V): %0.3f
		""" % (file,ra,dec,z,fit_min,fit_max,velscale,ebv)
		logfile.write(s)
		logfile.close()

	if (output_type=='outflow_test'):
		rdict,sigma = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Outflow Fitting Results ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('Parameter','Best-fit Value','+/-sigma'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		for key in rdict:
			logfile.write('\n{0:<30}{1:<30}{2:<30}'.format(key,rdict[key]['med'],rdict[key]['std']))
		logfile.write('\n{0:<30}{1:<30}'.format('Residual Std. Dev. = ',sigma))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()

	# outflow_test
	# write_log((cond1,cond2,cond3),20)
	if (output_type==20):
		cond1,cond2,cond3 = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		s = """
\n
### Tests for Outflows ###
------------------------------------
Outflow FWHM condition: %s
Outflow VOFF condition: %s
Outflow amplitude condition: %s
------------------------------------
 	Fitting for outflows...
 		""" % (cond1,cond2,cond3)
		logfile.write(s)
		logfile.close()
	elif (output_type==21):
		cond1,cond2,cond3 = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		s = """
\n
### Tests for Outflows ###
------------------------------------
Outflow FWHM condition: %s
Outflow VOFF condition: %s
Outflow amplitude condition: %s
------------------------------------
 	Not fitting for outflows...
		""" % (cond1,cond2,cond3)
		logfile.write(s)
		logfile.close()

	# Final Initialize MCMC
	if (output_type==3):
		df = pd.DataFrame(output_val).T
		df.fillna(0, inplace=True)
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Final Fitting Parameters ###')
		logfile.write('\n----------------------------------------------------')
		logfile.write('\n{0:<25}{1:>25}'.format('Parameter','Initial Value') )
		logfile.write('\n----------------------------------------------------')
		for key in output_val:
			logfile.write('\n{0:<25}{1:>25}'.format(key, output_val[key]['init']))
		logfile.write('\n----------------------------------------------------')
		logfile.close()

	# run_emcee
	if (output_type==40): # write user input emcee options
		ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter,threads = output_val
		# write_log((ndim,nwalkers,auto_stop,burn_in,write_iter,write_thresh,min_iter,max_iter,threads),40)
		a = str(datetime.datetime.now())
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Emcee Options ###')
		logfile.write('\n----------------------------------------------------')
		logfile.write('\n{0:<30}{1:<25}'.format('ndim'        , ndim ))
		logfile.write('\n{0:<30}{1:<25}'.format('nwalkers'    , nwalkers ))
		logfile.write('\n{0:<30}{1:<25}'.format('auto_stop'   , str(auto_stop) ))
		# logfile.write('\n{0:<30}{1:<25}'.format('conv_type'   , conv_type ))
		logfile.write('\n{0:<30}{1:<25}'.format('user burn_in', burn_in ))
		logfile.write('\n{0:<30}{1:<25}'.format('write_iter'  , write_iter ))
		logfile.write('\n{0:<30}{1:<25}'.format('write_thresh', write_thresh ))
		logfile.write('\n{0:<30}{1:<25}'.format('min_iter'    , min_iter ))
		logfile.write('\n{0:<30}{1:<25}'.format('max_iter'    , max_iter ))
		logfile.write('\n{0:<30}{1:<25}'.format('threads'     , threads ))
		logfile.write('\n{0:<30}{1:<25}'.format('start_time'  , a ))
		logfile.write('\n----------------------------------------------------')
		logfile.close()
	if (output_type==41): # write user input auto_stop options
		min_samp,autocorr_tol,ncor_times,conv_type = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		logfile.write('\n')
		logfile.write('\n### Autocorrelation Options ###')
		logfile.write('\n----------------------------------------------------')
		logfile.write('\n{0:<30}{1:<25}'.format('min_samp'  , min_samp     ))
		logfile.write('\n{0:<30}{1:<25}'.format('tolerance%', autocorr_tol ))
		logfile.write('\n{0:<30}{1:<25}'.format('ncor_times', ncor_times   ))
		logfile.write('\n{0:<30}{1:<25}'.format('conv_type' , conv_type    ))
		logfile.write('\n----------------------------------------------------')
		logfile.close()
	if (output_type==42): # write autocorrelation results to log
		# write_log((k+1,burn_in,stop_iter,param_names,tau),42,run_dir)
		burn_in,stop_iter,param_names,tau,autocorr_tol,tol,ncor_times = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		logfile.write('\n')
		logfile.write('\n### Autocorrelation Results ###')
		logfile.write('\n----------------------------------------------------')
		logfile.write('\n{0:<30}{1:<25}'.format('conv iteration', burn_in   ))
		logfile.write('\n{0:<30}{1:<25}'.format('stop iteration', stop_iter ))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('Parameter','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		for i in range(0,len(param_names),1):
			if (burn_in > (tau[i]*ncor_times)) and (0 < tol[i] < autocorr_tol):
				c = 'True'
			elif (burn_in < (tau[i]*ncor_times)) or (tol[i]>= 0.0):
				c = 'False'
			logfile.write('\n{0:<30}{1:<30.5f}{2:<30.5f}{3:<30.5f}{4:<30}'.format(param_names[i],tau[i],(tau[i]*ncor_times),tol[i],c))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		logfile.close()
	if (output_type==43): # write autocorrelation results to log
		# write_log(run_time,43,run_dir)
		run_time = output_val
		a = str(datetime.datetime.now())
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		logfile.write('\n{0:<30}{1:<25}'.format('emcee_runtime',run_time ))
		logfile.write('\n{0:<30}{1:<25}'.format('end_time',  a ))
		logfile.write('\n----------------------------------------------------------------------------')
		logfile.close()
	if (output_type==50): # write best fit parameters results to log
		par_names,par_best,sig_low,sig_upp = output_val 
		# write_log((par_names,par_best,sig_low,sig_upp),50,run_dir)
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Best-fit Parameters & Uncertainties ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Best-fit Value','-sigma','+sigma'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		for par in range(0,len(par_names),1):
			logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(par_names[par],par_best[par],sig_low[par],sig_upp[par]))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()

	# BH mass estimates 
	if (output_type==60):
		L5100_Hb, MBH_Hb = output_val 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### H-beta AGN Luminosity & Black Hole Estimate ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(' ','Best estimate','-sigma (dex)','+sigma (dex)'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(L5100)',L5100_Hb[0],L5100_Hb[1],L5100_Hb[2]))
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(M_BH)',MBH_Hb[0],MBH_Hb[1],MBH_Hb[2]))
		logfile.close()
	if (output_type==61):
		L5100_Ha, MBH_Ha = output_val 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### H-alpha AGN Luminosity & Black Hole Estimate ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(' ','Best estimate','-sigma (dex)','+sigma (dex)'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(L5100)',L5100_Ha[0],L5100_Ha[1],L5100_Ha[2]))
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(M_BH)',MBH_Ha[0],MBH_Ha[1],MBH_Ha[2]))
		logfile.close()
	# Systemic Redshift
	if (output_type==70):
		z_best = output_val 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Best-fitting Systemic Redshift ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(' ','Best estimate','-sigma (dex)','+sigma (dex)'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30.8f}{2:<30.8f}{3:<30.8f}'.format('z_systemic',z_best[0],z_best[1],z_best[2]))
		logfile.close()

	return None

##################################################################################
