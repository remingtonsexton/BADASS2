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

# Usage

## Fitting Options
 
![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_usage_1.png)

**`fit_reg`**: *tuple/list of length (2,)*; *Default: (4400,5800) # Hb/[OIII]/Mg1b/FeII region* ; the minimum and maximum desired fitting wavelength in angstroms, for example (4400,7000).  This is passed to the `determine_fit_reg()` function to check if this region is valid and and which emission lines to fit.

**`good_thresh`**: *float [0.0,1.0]*; *Default: 0.5*; the cutoff for minimum fraction of "good" pixels (determined by SDSS) within the fitting range to allow for fitting of a given spectrum.  If the spectrum has fewer good pixels than this value, BADASS skips over it and moves onto the next spectrum.

**`test_outflows`**: *bool*; *Default: True*; if *False*, BADASS does not test for outflows and instead does whatever you tell it to. if *True*, BADASS performs maximum likelihood fitting of outflows, using monte carlo bootstrap fitting to determine uncertainties, and uses the BADASS prescription for determining the presence of outflows.  If all of the BADASS outflow criteria are satisfied, the final model includes outflow components.  The BADASS outflow criteria to justify the inclusion of outflow components are the following:

1. <img src="https://render.githubusercontent.com/render/math?math=(%5Crm%7BFWHM%7D_%7B%5Crm%7B%5BOIII%5D%2Ccore%7D%7D%2B%5Cdelta%20%5Crm%7BFWHM%7D_%7B%5Crm%7B%5BOIII%5D%2Ccore%7D%7D)%20%3C%20(%5Crm%7BFWHM%7D_%7B%5Crm%7B%5BOIII%5D%2Coutflow%7D%7D-%5Cdelta%20%5Crm%7BFWHM%7D_%7B%5Crm%7B%5BOIII%5D%2Coutflow%7D%7D)">
2. <img src="https://render.githubusercontent.com/render/math?math=(v_%7B%5Crm%7B%5BOIII%5D%2Ccore%7D%7D-%5Cdelta%20v_%7B%5Crm%7B%5BOIII%5D%2Ccore%7D%7D)%20%3E%20(v_%7B%5Crm%7B%5BOIII%5D%2Coutflow%7D%7D%2B%5Cdelta%20v_%7B%5Crm%7B%5BOIII%5D%2Coutflow%7D%7D)">
3. <img src="https://render.githubusercontent.com/render/math?math=(A_%7B%5Crm%7B%5BOIII%5D%2Ccore%7D%7D-%5Cdelta%20A_%7B%5Crm%7B%5BOIII%5D%2Ccore%7D%7D)%20%3E%203%5Csigma_%7B%5Crm%7Bnoise%7D%7D"> 

**`mcbs_niter`**: *int*; *Default: 10*; the number of monte carlo bootstrap simulations for outflow testing.  If set to 0, outflows will not be tested for.

**`min_sn_losvd`**: *int*; *Default: 2*; minimum S/N threshold for fitting the LOSVD.  Below this threshold, BADASS does not perform template fitting with pPXF and instead uses a 5.0 Gyr SSP galaxy template as a stand-in for the stellar continuum.

## Autocorrelation/Convergence Options

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_usage_2.png)

**`nwalkers`**: *int*; *Default: 100*; number of "walkers" per parameter used by emcee to explore each parameter space.  The minimum number of walkers is 2 x ( # of parameters), set by emcee.

**`auto_stop`**: *bool*; *Default: True*; if set to *True*, autocorrelation is used to automatically stop the fitting process when a convergence criteria (`conv_type`) is achieved. 

**`conv_type`**: *str*; *Default: "all"*; *options: "all", "median", "mean", list of parameters*; mode of convergence.  Convergence of 'all' ensures all fitting parameters have achieved the desired `ncor_times` and `autocorr_tol` criteria, while "median" and "mean" only ensure that `ncor_times` and `autocorr_tol` criteria have been met for the median or mean of all parameters, respectively.  A list of valid parameters is also acceptable to ensure specific parameters have achieved convergence even if others have not.  In general "median" requires the fewest number of iterations and is not sensitive to poorly-constrained parameters, and "all" and "mean" require the most number of iterations and are much more sensitive to fluctuations in calculated autocorrelation times and tolerances.  A list of parameters is suitable in cases where one is only interested in certain spectral features.

**`min_samp`**: *int*; *Default: 2500*; if `auto_stop=True`, then the `burn_in` is the iteration at which convergence is achieved, and `min_samp` is the number of iterations *past convergence* used for posterior sampling (the samples used for histograms and estimating best-fit parameters and uncertainties.  If for some reason the parameters "jump out" of convergence, the `burn_in` will reset and BADASS will continue to sample until convergence is met again.  If emcee completes `min_samp` iterations after convergence is achieved without jumping out of convergence, this concludes the MCMC sampling.

**`ncor_times`**: *int* or *float*; *Default=2.0*; The number of integrated autocorrelation times (iterations) needed for convergence.  We recommend a minimum of `ncor_times=2.0`.  In general, it will require more than 2.0 autocorrelation times to calculate the autocorrelation time for a parameter chain.  Increasing `ncor_times` ensures that the parameter chain has stopped exploring the parameter space and is ready to begin sampling for the posterior distribution. 

**`autocorr_tol`**: *int* or *float*; *Default=5*; the percent change in the current integrated autocorrelation time and the previously calculated integrated autocorrelation time.  The `write_iter` determines how often BADASS checks a parameter's integrated autocorrelation time.  In general, we find that `autocorr_tol=5` (a 5% change) is acceptable for a converged parameter chain.  A parameter chain that diverges more than 10% in 100 iterations could still be exploring the parameter space for a stable solution.  A `autocorr_tol=1` (a 1% change) typically requires many more iterations than necessary for convergence. 

**`write_iter`**: *int*; *Default=100*; the frequency at which BADASS writes the current parameter values (median walker positions).  If `auto_stop=True`, then BADASS checks for convergence every `write_iter` iteration for convergence.

**`write_thresh`**: *int*; *Default=100*; the iteration at which writing (and checking for convergence if `auto_stop=True`) begins.  BADASS does not check for convergence before this value.

**`burn_in`**: *int*; *Default=47500*; if `auto_stop=False` then this serves as the burn-in for a maximum number of iterations.  If `auto_stop=True`, this value is ignored.

**`min_iter`**: *int*; *Default=100*; the minimum number of iterations BADASS performs before it is allowed to stop.  This is true regardless of the value of `auto_stop`.

**`max_iter`**: *int*; *Default=50000*; the maximum number of iterations BADASS performs before stopping.  This value is adhered to regardless of the value of `auto_stop` to set a limit on the number of iterations before BADASS should "give up."

## Model Options

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_usage_3.png)

These options are more-or-less self explanatory.  One can fit components appropriate (or not appropriate) for the types of objects they are fitting.  We summarize each component below:

**`fit_feii`**: *Default=True*; Broad and narrow FeII templates are taken from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) with each line modeled using a Gaussian.  FeII emission can be very strong in some Type 1 (broad line) AGN, but is almost undetectable in Type 2 (narrow line) AGN.

**`fit_losvd`**: *Default=True*; Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.  This is used to obtain stellar kinematics in spectra with resolvable absorption features, such as stellar velocity and dispersion.  If the S/N of the continuum (determined by the initial maximum likelihood fit) is less than 2.0, then `fit_losvd` is set to `False` by BADASS, since most of the time, trying to fit stellar features with S/N<5.0 produces non-sensical uncertainties.

**`fit_host`**: *Default=True*; this fits a 5.0 Gyr SSP galaxy template for the maximum likelihood fit.  If it is determined that the S/N of the spectra is too low to fit the LOSVD (S/N<2), then this simple host galaxy template takes the place of the spectral template fitting.

**`fit_power`**: *Default=True*; this fits a power-law component (steepness decreasing with increasing wavelength) to simulate the effect of the AGN continuum. 

**`fit_broad`**: *Default=True*; broad permitted emission lines commonly seen in Type 1 AGN.  

**``*fit_narrow*: *Default=True*; narrow forbidden emission lines seen in both Type 1 and Type 2 AGN, and some starforming galaxies. 

**`fit_outflows`**: *Default=True*; fitting of blueshifted (or "blue wing") emission in narrow-line features, indicative of outflowing NLR gas.  If `test_outflows=True`, and if BADASS determines that the inclusion of an "outflow" component does not satisfy the outflow criterion (read above), then `fit_outflows` is overridden to `False`.

**`tie_narrow`**: *Default=False*; tying all narrow-line widths (FWHM) together across the entire wavelength range is a common option in many fitting pipelines.  This is not recommended since different atomic families of lines can have different kinematics (and this is measurable!), however it is included as an option.

Examples of these spectral components can be seen in the example fit below:

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_model_options.png)

## The Main Function

All of the above options are fed into the `run_BADASS()` function as such:

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_usage_4.png)

# Output

BADASS produces a number of different outputs for the user at the end of the fitting process.  We summarize these below.

## Best-fit Model

This is simply a figure that shows the data, model, residuals, and best-fit components, to visually ascertain the quality of the fit. 

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_output_bestfit.png)

## Parameter Chains, Histograms, Best-fit Values & Uncertainties

For every parameter that is fit, BADASS outputs a figure to visualize the full parameter chain and all walkers, the burn-in, and the final posterior histogram with the best-fit values and uncertainties.  The purpose of these is for visual inspection of the parameter chain to observe its behavior during the fitting process, to get a sense of how well initial parameters were fit, and how walkers behave as the fit progresses.

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_output_chain.png)

## Log File

The entire fitting process, including selected options, is recorded in a text file for reference purposes.  It provides:
- File location
- FITS Header information, such as (RA,DEC) coordinates, SDSS redshift, velocity scale (km/s/pixel), and *E(B-V)*, which is retrieved from NED online during the fitting process and used to correct for Galactic extinction.
- Outflow fitting results (if `test_outflows=True`)
- Initial fitting parameters
- Autocorrelation times and tolerances for each parameter (if `auto_stop=True`)
- Final MCMC parameter best-fit values and uncertainties 
- Systemic redshift measured from stellar velocity (if `fit_losvd=True`)
- AGN luminosities at 5100 Å inferred from broad-line luminosity relation ([Greene et al. 2005](https://ui.adsabs.harvard.edu/abs/2005ApJ...630..122G/abstract))
- Black hole mass estimates using broad H-beta ([Sexton et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)) and broad H-alpha ([Woo et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract)) measurements.

![](https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_output_logfile.png)

## Best-fit Model Components

Best-fit model components are stored as arrays in `best_model_components.fits` files to reproduce the best-fit model figures as shown above.  This can be accessed using the [`astropy.io.fits`](https://docs.astropy.org/en/stable/io/fits/) module, for example:
```
from astropy.io import fits

hdu = fits.open(best_model_components.fits)
tbdata = hdu[1].data     # FITS table data is stored on FITS extension 1
data   = tbdata['data']  # the SDSS spectrum 
wave   = tbdata['wave']  # the rest-frame wavelength vector
model  = tbdata['model'] # the best-fit model
hdu.close()
```

Below we show an example data model for the FITS-HDU of `best_model_components.fits`.  You can print out the columns using 
```
print(tbdata.columns)
```
which shows

<p align="center">
<img src="https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_output_bestfit_comp.png" width="400" />
</p>

## Best-fit Parameters & Uncertainties 

All best-fit parameter values and their upper and lower 1-sigma uncertainties are stored in `par_table.fits` files so they can be more quickly accessed than from a text file.  These are most easily accessed using the (`astropy.table`](https://docs.astropy.org/en/stable/table/pandas.html) module, which can convert a FITS table into a Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html):

```
from astropy.table import Table  

table = Table.read('par_table.fits')
pandas_df = table.to_pandas()
print pandas_df
```
which shows

<p align="center">
<img src="https://github.com/remingtonsexton/BADASS2/blob/master/figures/BADASS_output_partable.png" width="400" />
</p>

# Contributing

Please let us know if you wish to contribute to this project by reporting any bugs, issuing pull requests, or requesting any additional features to help make BADASS the most detailed spectroscopic analysis tool in astronomy.

# Credits

#### [Remington Oliver Sexton](https://astro.ucr.edu/members/graduate-students/#Remington) (UC Riverside, Physics & Astronomy)

#### William Matzko (George Mason University, Physics and Astronomy)

#### Nicholas Darden (UC Riverside, Physics & Astronomy)

# License

MIT License

Copyright (c) 2019 Remington Oliver Sexton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
