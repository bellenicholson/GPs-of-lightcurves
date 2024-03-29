#!/usr/bin/env python3


import numpy as np
# import pandas as pd
from matplotlib import pyplot as p
import george, emcee, corner
from astropy.io.ascii import read
from scipy.optimize import minimize
import sys

def get_emcee_results(samples,ndim,lam=True):
    
    results = []
    # uperr=[]
    # loerr = []
    for i in range(ndim):
        samp = samples[:,i]
        if i == 2: samp = np.exp(samp)
        # if i == 4: samp = np.exp(samp)
        if i == 5: 
            if lam:
                samp =  np.sqrt(np.exp(samp)/2.)
            else:
                samp =  np.exp(samp)
                
        v = np.percentile(samp, [16, 50, 84], axis=0)
        results.append(v[1])
        results.append(v[2]-v[1])
        results.append(v[1]-v[0])
    
    return results

def walker_diagnostic_plot(samples,lnP,ndim):

    # samples = sampler.chain
    # lnP = sampler.lnprobability
    fig,axes=p.subplots(ndim+1,1,sharex=True,figsize=(12,2*(ndim+1)))
    axes[0].plot(lnP,alpha=0.5,lw=0.5)
    axes[0].set_ylabel(r'$\log Prob$')
    labels=['Mean','Jitter',r'$\log A$',r'$\Gamma_1$',r'$\log P$',r'$\log m$']
    for i in range(ndim):
        axes[i+1].plot(samples[:,:,i],alpha=0.5,lw=0.5)
        axes[i+1].set_ylabel(labels[i])
    axes[-1].set_xlabel('steps');

def uniform_lnprior(parvec,time):
    men,lnjit, logA,gamma, logP, logm = parvec
    dt = time[1]-time[0]
    T = time[-1]-time[0]
    if abs(lnjit) > 20:
        return -np.inf
    elif abs(men) > 15:
        return -np.inf
    elif abs(logA) > 30:
        return -np.inf
    elif gamma < 0:
        return -np.inf
    elif gamma > 15:
        return -np.inf
    elif logP > np.log(T):
        return -np.inf
    elif logP < np.log(2*dt):
        return -np.inf    
    elif abs(logm) > 30:
        return -np.inf
    
    return 0
    

def log_like_prior(parvec,gp,time,flux,fluxerr):
    lpr = uniform_lnprior(parvec,time)
    if not np.isfinite(lpr):
        # print('here prior')
        return -np.inf

    gp.set_parameter_vector(parvec)
    try:
        gp.compute(time,fluxerr)

    except:
        # print('here compute')
        return -np.inf
    
    try:    
        loglike = gp.log_likelihood(flux)
    except:
        # print('here log like')
        return -np.inf    

    return loglike + lpr

def neg_log_like_prior(parvec,gp,time,flux,fluxerr):
    return -log_like_prior(parvec,gp,time,flux,fluxerr)


if __name__ == '__main__':

    _,lightdir,lightname,parfile,outdir = sys.argv

    # time, flux, fluxerr = np.loadtxt('./LocalTests/BasicGP_lightcurve.txt').T
    
    lightcurve = np.loadtxt(lightdir+lightname).T
    
    # if min(lightcurve.shape)<3:
    #     time, flux = lightcurve[0], lightcurve[1]
    #     fluxerr = np.ones(len(time))*1e-4
    # else:
    #     time, flux, fluxerr = lightcurve[0][::], lightcurve[1][::], lightcurve[2][::]


    time, flux, rv = lightcurve[0][::10], lightcurve[1][::10], lightcurve[2][::10]
    fluxerr = np.ones(len(time))*1e-4


    filename = outdir+lightname[:-4]

    # input_pars = np.loadtxt(lightdir+parfile)
    # lnjit = np.log(abs(np.mean(fluxerr)))
    num,mean,lnjit,amp,gamma,logPeriod,metric = np.loadtxt(lightdir+parfile)
    input_gppars = [mean,lnjit,amp,gamma,logPeriod,metric]

    k = input_gppars[2] * george.kernels.ExpSine2Kernel(gamma=input_gppars[3], log_period=input_gppars[4]) * george.kernels.ExpSquaredKernel(input_gppars[5])
    gp = george.GP(k,mean = input_gppars[0], fit_mean =True, white_noise=input_gppars[1], fit_white_noise=True)

    gpinput_pars = gp.get_parameter_vector()

    gp.compute(time, fluxerr)

    # print(input_gppars)
    # print(gpinput_pars)


    # input_pars = [lnjit, *gpinput_pars]

    #uncomment for light curves:
    # min_pars = minimize(neg_log_like_prior,gpinput_pars,args=(gp,time,flux,fluxerr))
    #uncomment for RVs:
    min_pars = minimize(neg_log_like_prior,gpinput_pars,args=(gp,time,rv,fluxerr))
    # print('minimize pars', min_pars.x)


        #
    ndim, nwalkers = len(min_pars.x), 80
    pos =  min_pars.x + np.random.randn(nwalkers, ndim)*1e-4

    # backend = emcee.backends.HDFBackend(filename+'_run19.h5')
    # backend.reset(nwalkers, ndim)
    
    # uncomment for light curves:
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_like_prior, args=[gp,time,flux,fluxerr])#,backend=backend)
    # uncomment for rv:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_like_prior, args=[gp,time,rv,fluxerr])#,backend=backend)

    # sampler.run_mcmc(pos,800)

    burn_steps = 300
    # print("Running burn-in 1...")
    sampler.run_mcmc(pos, burn_steps)
    state = sampler.get_last_sample()

    # print("Running burn-in 2...")
    sampler.reset()
    sampler.run_mcmc(state, burn_steps)
    state = sampler.get_last_sample()
    bestvals = state.coords[state.log_prob==max(state.log_prob)]
    pos2 = bestvals + np.random.randn(nwalkers, ndim)*1e-4

    # print("Running burn-in 3...")
    sampler.reset()
    sampler.run_mcmc(pos2, burn_steps)
    state = sampler.get_last_sample()
    bestvals2 = state.coords[state.log_prob==max(state.log_prob)]
    pos3 = bestvals2 + np.random.randn(nwalkers, ndim)*1e-4


    max_steps = 8000
    index = 0
    autocorr = np.empty(max_steps)
    flag=1
    old_tau = np.inf

    for sample in sampler.sample(pos3, iterations=max_steps, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 60 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            flag=0
            break
        old_tau = tau

    # last_state = sampler.get_last_sample()
    # print(last_state)
    # np.savetxt(filename+'_finalstate.txt',final_state)
    # flag=1
    # sampler = emcee.backends.HDFBackend('./LocalTestsLightcurveFromGP_1e0_360_run11.h5')



    if flag==0:
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        
    else:
        big_burn = 2000
        try:
            tau = sampler.get_autocorr_time(discard=big_burn)
            burnin = int(2 * np.max(tau)) + big_burn
            thin = int(0.5 * np.min(tau))
        except:
            flag=2
            burnin  = big_burn
            thin = 1 

        
        
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    samples = sampler.get_chain(discard=burnin,thin=thin)
    lnp = sampler.get_log_prob(discard=burnin,thin=thin)


    corner.corner(flat_samples, labels=['Mean','Jitter',r'$\log A$',r'$\Gamma_1$',r'$\log P$',r'$\log m$']);
    p.savefig(filename+'_corner.png')
    p.close()

    # print(np.shape(samples))
    walker_diagnostic_plot(samples,lnp,ndim)
    p.savefig(filename+'_walkers.png')
    p.close()




    results = get_emcee_results(flat_samples,ndim,lam=False)

    np.savetxt(filename+'_results.txt', np.array([num,flag] + results), fmt='%7.4e',
               header = 'num flag mean mean_uper mean_loer lnjit lnjit_uper lnjit_loer amp amp_uper amp_loer gamma1 gamma1_uper gamma1_loer logP logP_uper logP_loer metric metirc_uper metric_loer',comments='')
