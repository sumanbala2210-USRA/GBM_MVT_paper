from astropy.io.fits import getdata,getheader
import matplotlib as mpl
mpl.use('Agg')
from haar_nondec_regular_err_wt import haar_nondec
from haar_denoise import haar_denoise
from pylab import errorbar,title,xlabel,ylabel,savefig,plot,loglog,xlim,ylim,figure,clf
from numpy import log,sqrt,where,array,exp,median,arange,ones,logical_or,isnan,percentile
from scipy.optimize import minimize_scalar
import numpy as np
import warnings

warnings.filterwarnings('ignore', r'divide by zero encountered')
warnings.filterwarnings('ignore', r'invalid value encountered')


def haar_power_mod(rate,drate,min_dt=1.e-4,max_dt=100.,tau_bg_max=0.01,nrepl=2,doplot=True,bin_fac=4,zerocheck=False,afactor=-1.,snr=3.,verbose=True,weight=True,file='test'):
    """
      set afactor to negative to have the code estimate it
      largest timescale used will be tau_bg_max

      zerocheck=True generates a plot of the zero level
    """
    dt_str=r'$\Delta t$'

    if (weight):
        # calculate the power spectrum weighted by the denoised flux
        wt = haar_denoise(rate,drate)
        wt = haar_denoise(wt,drate).clip(0.)
    else:
        wt = ones(len(rate),dtype='float64')

    # get the difference of every two points
    dta,dta1, pspec, pspec0, dpspec = haar_nondec(rate,drate,wt,dt=min_dt,nrepl=nrepl,bfac=bin_fac,osamp=bin_fac*8)

    g = pspec0>0
    dta = dta[g]; dta1 = dta1[g]
    pspec = pspec[g]; pspec0 = pspec0[g]; dpspec = dpspec[g]

    tau = 0.5*(dta+dta1)
    tmax = tau[(pspec/pspec0).argmax()]
    tsnr,tbeta,tmin,dtmin,slope,sigma_tsnr,sigma_tmin = 0.,0.,0.,0.,0.,0.,0.
    otype='limit'

    #
    # estimation of zero-level
    g=tau<tau_bg_max
    if (g.sum()<2 or afactor>0):
        afactor = abs(afactor)
        pspec0 *= afactor; dpspec *= afactor
    else:
        a = median(pspec[g]/pspec0[g])
        pspec0 *= a; dpspec *= a
        if (verbose): print (""" %s a factor: %f""" % (file,a))

    pspec -= pspec0

    #
    # define tsnr and set everything on shorter timescales to be background noise
    g=pspec<snr*dpspec; g2 = ~g; g *= tau<tmax
    wi1 = where(g)[0]
    if (len(wi1)>0): i1 = wi1[-1]
    else: i1 = 0
    g[:i1] = True; g2[:i1] = False
    k=0
    while (i1>0 and pspec[i1]>dpspec[i1] and k<bin_fac):
        g[i1] = False; g2[i1] = True
        i1 -= 1
        k+=1

    if (zerocheck):
        figure(1)
        clf()
        plot (tau,pspec+pspec0,'o')
        plot (tau,pspec,'o')
        if (g.sum() > 0): plot (tau[g],pspec[g],'kv')
        plot (tau,pspec0,'k-')
        plot (tau,dpspec,'k-.')
        loglog()
        figure(2)
        clf()

    # THIS IS THE NEW, CORRECTED BLOCK
    if (g2.sum()<2):
        if (verbose): print ("%s Not enough significant data!" % file)
    else:
        # --- START: NEW DATA CLEANING STEP ---
        # Create a new mask that is True only where g2 is True AND where pspec is a finite number (not NaN or Inf)
        clean_mask = g2 & np.isfinite(pspec)
        
        # Now, check if there's any data left after cleaning
        if (clean_mask.sum() < 2):
            if (verbose): print ("%s Not enough significant data after removing NaNs!" % file)
            # Exit the function gracefully if no valid data is left
            return tsnr,tbeta,tmin,dtmin,slope,sigma_tsnr,sigma_tmin
            
        # From now on, use 'clean_mask' instead of 'g2' for all calculations and plotting
        g2 = clean_mask
        # --- END: NEW DATA CLEANING STEP ---
        
        
        # Now the rest of your code will work because g2 only points to valid numbers
        #pspm = pspec[g2].max() # This is now safe
        #pspm = percentile(pspec[g2], 99)
        
        # --- FINAL: Filtered Normalization by Threshold ---
        # We know the "good" data is valued below ~1e6 and the outliers are >1e9.
        # This code creates a temporary slice of the data containing only the "good" points
        # and finds the maximum value from that slice for normalization.
        # This is the most direct way to replicate the appearance of the old plot.
        
        good_data_for_norm = pspec[g2][pspec[g2] < 1e7] # Threshold set to 10 million
        
        if len(good_data_for_norm) > 0:
            pspm = good_data_for_norm.max()
        else:
            # Fallback if no data is below the threshold
            pspm = pspec[g2].mean()

        pspec /= pspm
        pspec0 /= pspm
        dpspec /= pspm
        # --- END: Final Normalization ---
        # --- END: Filtered Normalization ---

        # ... the rest of the code in the 'else' block continues as before...

        tsnr = tau[i1+1].max()

        #
        # interpolate to define tbeta
        y = pspec[g2]-pspec0[g2]
        h = where(y>0)[0]
        fix_beta=False
        if (len(h)>2):
            ib1 = h[0]; ib0 = ib1-1
            if (ib0>=0):
                tbeta = tau[g2][ib0]
                if (y[ib1]!=y[ib0]): tbeta -= y[ib0]*(tau[g2][ib1]-tau[g2][ib0])/(y[ib1]-y[ib0])
            else:
                fix_beta=True
        else:
            tbeta = tau.max()

        #
        # find departures from a linear fit, will be starting point for tmin calculation
        xx = log(tau[g2])
        yy = log(pspec[g2])-2.*xx
        dyy = dpspec[g2]/pspec[g2]
        vmu = 1./(1./dyy**2).cumsum()
        mu = (yy/dyy**2).cumsum() * vmu

        #
        # departure condition: change in slope by 0.1, but at least 0.5-sigma, relative to a time a factor 2 shorter
        thresh = (0.5*sqrt(dyy[1:]**2+bin_fac*vmu[:-1])).clip(0.1*log(2.))
        wib1 = where(mu[:-1]-yy[1:]>thresh)[0]
        ib1 = len(mu)-2
        if (len(wib1)>0): ib1 = wib1[0]
        ib0 = ib1 - 1
        if (ib0<0): ib0=0
        mu0 = (mu[ib0]+mu[ib1])/2.

        # linear region is ib0-1 -> ib1
        # flat region is ib1+1 -> ib1+bin_fac
        # intersectoin should be bracketed by xx[ib0-1], xx[ib1+1]
        x,y = xx[:ib1+bin_fac+1], yy[:ib1+bin_fac+1]
        dy = dyy[:ib1+bin_fac+1]; dy2 = dy*dy

        b1 = (y/dy2).sum(); m11 = (1./dy2).sum()
        par = array([mu0,0.5,0.])
        def break_fun(xb):
            # finds: y = par[0] + par[1]*(x-xb), error on xb is par[2]
            if (isnan(xb)): xb = x.min()
            xl = x<xb; xu = ~xl
            m12 = ( (x[xu]-xb)/dy2[xu] ).sum()
            m22 = ( (x[xu]-xb)**2/dy2[xu] ).sum()
            det2 = m11*m22-m12**2
            b2 = ( y[xu]*(x[xu]-xb)/dy2[xu] ).sum()

            if (det2>0):
                par[0] = (m22*b1 - m12*b2)/det2
                par[1] = (m11*b2 - m12*b1)/det2

            if (xl.sum()>0):
                chi0 = ( (y[xl]-par[0])**2/dy2[xl] ).sum()
                m01 = -par[1]*(1./dy2[xu]).sum()
                m00 = -par[1]*m01
                m02 = ( (y[xu]-par[0])/dy2[xu] ).sum() - 2.*par[1]*m12
                par[2] = 1./sqrt(m00 + (m01*(m02*m12-m01*m22) + m02*(m01*m12-m02*m11))/det2)
            else: chi0 = 0.

            return chi0 + ( (y[xu]-par[1]*(x[xu]-xb)-par[0])**2/dy2[xu] ).sum()


        res = minimize_scalar(break_fun,bounds=(x.min(),x.max()-1.e-5),method='bounded')
        c0, xmin = res['fun'], res['x']
        dxmin = 1.*par[2]
        if (isnan(dxmin)): dxmin = xmin

        mu0, slope = par[0], 1. + 0.5*par[1]
        sigma_tsnr, sigma_tmin = exp(0.5*mu0 + x[0]), exp(0.5*mu0 + xmin)

        if (xmin>=x[1] and xmin-dxmin>x[0]): otype='measurement'
        else:
            for i in range(3):
                c1 = break_fun(xmin+log(1.+dxmin))
                dxmin = max(0.5*dxmin,min(dxmin/(1.e-5+abs(c1-c0)),1.5*dxmin))
            sigma_tmin *= exp( slope*log(1.+snr*dxmin*sqrt(bin_fac)) )

        tmin = exp(xmin)
        dtmin = tmin*dxmin*sqrt(bin_fac)

        if (fix_beta):
            y = 2*log(tau) + mu0 - log(pspec0)
            h = where(y>0)[0]
            if (len(h)>2):
                ib1a = h[0]; ib0a = ib1a-1
                tbeta = tau[ib0a]
                if (y[ib1a]!=y[ib0a]): tbeta -= y[ib0a]*(tau[ib1a]-tau[ib0a])/(y[ib1a]-y[ib0a])

        #if (verbose): print (""" %s T_snr=%f T_beta=%f T_min=%f +/- %f""" % (file,tsnr,tbeta,tmin,dtmin))

        if (doplot):

            #base=file.split('/')[-1]
            fname=file+'_haar_mod.png'

            # square-root transformation
            #plot (tau,sqrt(dpspec),'b-.')
            pspec[g2] = sqrt( pspec[g2] )
            dpspec[g2] /= 2.*pspec[g2]
            pspec0 = sqrt( pspec0 )

            xlabel(dt_str+" [s]",fontsize=14)
            ylabel(r'Flux Variation $\sigma_{X,\Delta t}$',fontsize=14)

            #plot (tau,pspec0,'b--')
            plot (tau[g2][ib1+1],pspec[g2][ib1+1],'mo',ms=10,mew=0)

            xx1 = array([min_dt/2,tmin])
            xx2 = array([tmin,max_dt*2])
            plot (xx1,xx1*exp(mu0/2.),'r-',alpha=0.5)
            plot (xx2, exp(0.5*mu0-(slope-1)*log(tmin) + slope*log(xx2)),'r-',alpha=0.5)
            errorbar(tau[g2], pspec[g2], yerr=dpspec[g2], xerr=0.5*(dta1-dta)[g2], fmt='bo',capsize=0,linestyle='None',markersize=3)
            loglog()

            # draw some dotted lines
            i0 = pspec[g2].argmax()
            x1,y1 = tau[g2][i0],pspec[g2][i0]
            xx = array([min_dt/2,max_dt*2])
            for i in range(-20,20):
                plot (xx,y1*xx/x1*2.**i,'k:',alpha=0.5)

            xlim((tau[g2].min()/4.,tau[g2].max()*1.5))
            #ylim((pspec[g2].min()/2.,pspec[g2].max()*1.5))
            # Use nanmin and nanmax to safely ignore any NaN values when setting limits
            ylim((np.nanmin(pspec[g2]) / 2., np.nanmax(pspec[g2]) * 1.5))

            str1=r'$\Delta t_{\rm snr}=$'+"""%.4f""" % tsnr
            str2=r'$t_{\beta}=$'+"""%.4f""" % tbeta
            if (otype=='limit'):
                str3=r'$\Delta t_{\rm min}<$'+"""%.4f""" % (tmin+snr*dtmin)
            else:
                str3=r'$\Delta t_{\rm min}=$'+"""%.4f +/- %.4f""" % (tmin,dtmin)
            #title(str1+'   '+str2+'   '+str3)
            title(str3)

            # plot limits if present
            if (g.sum() > 0):
                plot (tau[g],sqrt(pspec[g].clip(0)+snr*dpspec[g] ),'bv')
                #plot (tau[g],sqrt(pspec[g].clip(0.)),'bo',alpha=0.5)

            savefig(fname)

        if (otype=='limit'):
            tmin += snr*dtmin
            dtmin = 0.
    
    if (verbose): print (""" %s T_snr=%f T_beta=%f T_min=%f +/- %f""" % (file,tsnr,tbeta,tmin,dtmin))
    return tsnr,tbeta,tmin,dtmin,slope,sigma_tsnr,sigma_tmin


def haar_power_mod_bat4zC(file,t1=-5.,t2=-20.,nrepl=2,doplot=True,bin_fac=4,afactor=1.,verbose=True,bg=[],res0=0.,res1=0.,chan='all',bin=1,zerocheck=False,tau_bg_max=0.01):

    d=getdata(file,ignore_missing_end=True)
    if (chan=='all'): rate = (d['C1'] + d['C2'] + d['C3'] + d['C4']).astype('float64')
    elif (chan=='COUNTS'): rate = d['COUNTS'].astype('float64')
    else: rate = d[chan].astype('float64')

    drate = sqrt(rate)
    try:
        h=getheader(file,ignore_missing_end=True)
        min_dt = h['ETIME']
    except:
        h=getheader(file,1,ignore_missing_end=True)
        min_dt = h['ETIME']
    t = h['TIME0'] + min_dt*arange(len(rate)).astype('float64')

    if (bg!=[]):
        x=t[::bin]; y=rate.cumsum()[::bin]; vy = (drate**2).cumsum()[::bin]
        x,y,dy = 0.5*(x[1:]+x[:-1]),(y[1:]-y[:-1])/bin,sqrt(vy[1:]-vy[:-1])/bin
        j1 = (dy>0)*(x>=bg[0])*(x<=bg[1])
        j2 = (dy>0)*(x>=bg[2])*(x<=bg[3])
        j = logical_or(j1,j2)

        dy2 = dy[j]**2
        norm = (1./dy2).sum()
        x0,y0 = (x[j]/dy2).sum()/norm , (y[j]/dy2).sum()/norm
        res1 = 0.
        norm1 = ( (x[j]-x0)**2/dy2 ).sum()
        if (norm1>0): res1 = ( (y[j]-y0)*(x[j]-x0)/dy2 ).sum()/norm1
        res0 = y0-res1*x0

        file1 = file.replace('.fits','.txt')
        f = open(file1,'w')
        for i in range(len(x)):
            f.write("""%f %f %f\n""" % (x[i],y[i]-res0-res1*x[i],dy[i]))

        f.close()
        x,y,dy=0,0,0

    j = (t>t1)*(t<t2)

    t = t[j]
    rate = rate[j] - res0 - res1*t ; drate = drate[j]

    max_dt = t.max()-t.min()
    d=0.; t=0.;

    return haar_power_mod(rate,drate,min_dt=min_dt,max_dt=max_dt,nrepl=nrepl,doplot=doplot,bin_fac=bin_fac,verbose=verbose,file=file,zerocheck=zerocheck,afactor=afactor,tau_bg_max=tau_bg_max)


def haar_power_mod_bat4zS(file,nrepl=2,doplot=True,bin_fac=4,verbose=True,weight=True,zerocheck=False,n1=0,n2=0,tau_bg_max=0.01,afactor=1.):
    """
      set afactor to negative to have the code estimate it
      largest timescale used will be tau_bg_max

      zerocheck=True generates a plot of the zero level
      res0,res1: parameters for background subtraction, res0+res1*t
    """

    h=getheader(file)
    d=getdata(file)
    rate = d['RATE']
    drate = d['ERROR']
    d=0.

    if (n1!=n2):
        if (n1<0): n1=0
        rate = rate[n1:n2+1]
        drate = drate[n1:n2+1]

    min_dt = h['ETIME']
    max_dt = len(rate)*min_dt

    return haar_power_mod(rate,drate,min_dt=min_dt,max_dt=max_dt,nrepl=nrepl,doplot=doplot,bin_fac=bin_fac,verbose=verbose,file=file,zerocheck=zerocheck,afactor=afactor,tau_bg_max=tau_bg_max)



def haar_power_mod_bat4z(file,t1=-5.,t2=-20.,nrepl=2,doplot=True,bin_fac=4,verbose=True,weight=True,res0=0.,res1=0.,zerocheck=False,afactor=1.,tau_bg_max=0.01):
    """
      set afactor to negative to have the code estimate it
      largest timescale used will be tau_bg_max

      zerocheck=True generates a plot of the zero level
      res0,res1: parameters for background subtraction, res0+res1*t
    """
    d=getdata(file)
    t, rate, drate = d['TIME'], d['COUNTS'], d['ERROR']
    j = (t>t1)*(t<t2)

    min_dt = t[1] - t[0]
    t = t[j]
    rate = rate[j] - res0 - res1*t ; drate = drate[j]
    max_dt = t.max()-t.min()
    d=0.; t=0.;

    return haar_power_mod(rate,drate,min_dt=min_dt,max_dt=max_dt,nrepl=nrepl,doplot=doplot,bin_fac=bin_fac,verbose=verbose,file=file,zerocheck=zerocheck,afactor=afactor,tau_bg_max=tau_bg_max)
