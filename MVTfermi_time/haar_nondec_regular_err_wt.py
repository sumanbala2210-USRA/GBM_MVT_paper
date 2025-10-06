from numpy import log2,hstack,arange,round,unique,zeros,sqrt,cumsum,ceil,log2,exp
from scipy.special import erf

def haar_nondec(data,error,weight,dt=1.0,osamp=32.,nrepl=1,bfac=4.):
    """
       data is in the input (Poisson) time series

       dt: the binsize in seconds of the regularly spaced time series

       wt: weighting factor

       bfac: is the bin_factor, >=1 defining the sampling of the
         output data.  Larger bfac, more points.

       osamp: over-sampling of the time series, later binned up.  must
         have osamp>=bfac.

       nrepl: repeat the time series end on end?  default no (nrepl=1).
    """
    cx=hstack( (0,cumsum( data.astype('float64') )) )
    wt=hstack( (0,cumsum( weight.astype('float64') )) )
    vx=hstack( (0,cumsum( error.astype('float64')**2 )) )

    for i in range(nrepl-1):
        cx = hstack((cx,cx[1:-1]+cx[-1]))
        wt = hstack((wt,wt[1:-1]+wt[-1]))
        vx = hstack((vx,vx[1:-1]+vx[-1]))

    nmax = len(cx)-1
    lscl_max=int(ceil(log2(nmax)))

    if (bfac<=0): bfac=1.
    if (osamp<bfac): osamp=bfac

    if (bfac<osamp):
        scl_out=2**( arange(lscl_max,dtype='int32') )
        scl2=2.*round( 2**( arange(lscl_max*bfac,dtype='float64')/bfac)/2. )
        scl_out=hstack((scl_out,scl2)).astype('int32')
        scl_out.sort(); scl_out = unique( scl_out )
        scl_out=scl_out[(scl_out>0)*(2*scl_out<=nmax)]

    scales=2**( arange(lscl_max,dtype='int32') )
    scl2=2.*round( 2**( arange(lscl_max*osamp,dtype='float64')/osamp)/2. )
    scales=hstack((scales,scl2)).astype('int32')
    scales.sort(); scales = unique( scales )
    scales=scales[(scales>0)*(2*scales<=nmax)]
    if (bfac>=osamp): scl_out = 1*scales

    nscales=len(scales)
    pspec = zeros(nscales,dtype='float64')
    pspec0 = zeros(nscales,dtype='float64')
    vpspec = zeros(nscales,dtype='float64')

    for k in range(nscales):
        scl=scales[k]; scl2 = scl**2
        cfac = nmax/(nmax-2.*scl+1)

        wav2 = (cx[2*scl:nmax+1] - 2*cx[scl:nmax-scl+1] + cx[:nmax-2*scl+1])**2
        vwav = vx[2*scl:nmax+1] - vx[:nmax-2*scl+1]

        wt0 = (wt[2*scl:nmax+1] - wt[:nmax-2*scl+1])/(2.*scl)
        wts = wt0.mean()

        pspec[k] = (wav2*wt0).sum()*cfac/scl2 / wts
        pspec0[k] = (vwav*wt0).sum()*cfac/scl2 / wts
        # signal=0 error only
        vpspec[k] = ( ((vwav*wt0)**2).sum()/scl2/wts**2 + 0.5*pspec0[k] )*cfac**2/scl


    scl1,scl2 = scl_out[:-1], scl_out[1:]
    nscales1 = len(scl1)
    psp = zeros(nscales1,dtype='float64')
    psp0 = zeros(nscales1,dtype='float64')
    dpsp = zeros(nscales1,dtype='float64')

    for i in range(nscales1):
        h = (scales>=scl1[i])*(scales<scl2[i])
        nh = h.sum()
        if (nh>0):
            scl1[i],scl2[i] = scales[h].min(),scales[h].max()
            psp[i] = pspec[h].sum()/nh
            psp0[i] = pspec0[h].sum()/nh
            dpsp[i] = sqrt( vpspec[h].sum()*(nrepl+1.)/nh )

    return dt*scl1,dt*scl2,psp,psp0,dpsp
