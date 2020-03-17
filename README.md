![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_logo.gif)

BADASS is an open-source spectral analysis tool designed for detailed decomposition of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical.  The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte Carlo sampler [emcee](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract) for robust parameter and uncertainty estimation, as well as autocorrelation analysis to access parameter chain convergence.  BADASS can fit the following spectral features:
- Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.
- Broad and Narrow FeII emission features using the FeII templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract).
- Broad permitted and narrow forbidden emission line features. 
- AGN power-law continuum. 
- "Blue-wing" outflow emission components found in narrow-line emission. 

All spectral components can be turned off and on via the [Jupyter Notebook](https://jupyter.org/) interface, from which all fitting options can be easily changed to fit non-AGN-host galaxies (or even stars!).  The code was originally written in Python 2, but is now fully compatible with Python 3.  BADASS was originally written to fit Keck Low-Resolution Imaging Spectrometer (LRIS) data ([Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)), but because BADASS is open-source and *not* written in an expensive proprietary language, one can easily contribute to or modify the code to fit data from other instruments.  

<b>  
If you use BADASS for any of your fits, I'd be interested to know what you're doing and what version of Python you are using, please let me know via email at remington.sexton-at-email.ucr.edu.
</b>


* TOC 
{:toc}

# Installation

The easiest way to get started is to simply clone the repository. 

As of BADASS v6.0.1, the following packages are required (Python 2.7):
- `numpy 1.11.3`
- `scipy 1.1.0`
- `pandas 0.23.4`
- `matplotlib 2.2.3`
- `astropy 2.0.9`
- `astroquery 0.3.9`
- `emcee 2.2.1`
- `natsort 5.5.0`

The code is run entirely through the Jupyter Notebook interface, and is set up to run on the four included SDSS spectra in the ".../example_spectra/" folder, one at a time.  If one chooses to fit numerous spectra consecutively, this is the recommended directory structure:

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_example_spectra.png)

Simply create a folder containing the SDSS FITS format spectrum, and BADASS will generate a working folder for each fit titled "MCMC_output_#".  BADASS automatically generates a new output folder if the same object is fit again (it does not delete previous fits).

In the Notebook, one need only specify the location of the spectra, the location of the BADASS support files, and the location of the templates for pPXF, as shown below:

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_directory_structure.png)


