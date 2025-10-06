from numpy import zeros,empty,ceil,log2,hstack,sqrt,abs,median

def haar_denoise(data,err=[],thresh_fac=1.,estimate_noise=False,soft=False):
    """
        Input:
            data: data vector
            err: option error vector, assumed =1 if absent
            thresh_fac: require error > thresh_fac during denoising

            (larger thresh_fac, more denoising)
        Output:
            denoised data vector

        Notes: err assumed to be a vector if present.  To use a scalar
          error for all data points, set thresh_fac to that value.
    """
    cx=1.*data.astype('float64')
    ln0=len(data)
    use_err=True
    '''
    if (err==[]): use_err=False
    else: vx = err.astype('float64')**2
    '''

    if err is None or err.size == 0:
        use_err = False
    else:
        vx = err.astype('float64')**2

    n=int(ceil( log2(ln0) ))
    ln=2**n

    # extend to get 2^n ln
    l1=0
    if (ln>ln0):
        l1 = int( 0.5*(ln-ln0) )
        l2 = ln-ln0-l1
        cx = hstack((cx[:l1][::-1],cx,cx[-l2:][::-1]))
        if (use_err): vx = hstack((vx[:l1][::-1],vx,vx[-l2:][::-1]))


    noise=1.
    if (estimate_noise):
        if (use_err):
            err2 = 1./sqrt(2.)*sqrt(err[1:]**2+err[:-1]**2)
            noise = 1.05*median(abs(data[1:]-data[:-1])/err2)
        else: noise = 1.05*median(abs(data[1:]-data[:-1]))
        #print ("Estimated Noise level = %.2e") % noise

    x0 = cx.mean()
    cx -= x0
    xm = empty(ln,dtype='float64')
    x_recon = zeros(ln,dtype='float64') + x0
    cx[:] = cx.cumsum()
    if (use_err):
        vx[:] = vx.cumsum()
        vxm = empty(ln,dtype='float64')


    tlt = 1.386*(thresh_fac*noise)**2
    for m in range(n):
        scl=2**(n-m-1)

        # apply the forward tranform
        xm[:-2*scl]     = 2*cx[scl:-scl]-cx[:-2*scl]-cx[2*scl:]
        xm[-2*scl:-scl] = 2*cx[-scl:]-cx[-2*scl:-scl]-cx[:scl]-cx[-1]
        xm[-scl:]       = 2*cx[:scl]-cx[-scl:]-cx[scl:2*scl]+cx[-1]
        # variance
        if (use_err):
            vxm[:-2*scl]= vx[2*scl:]-vx[:-2*scl]
            vxm[-2*scl:]= vx[:2*scl]+vx[-1]-vx[-2*scl:]
        else: vxm=2*scl

        # threshold
        h = xm*xm <= tlt*m*vxm
        xm[h] = 0
        if (soft):
            mh = ~h
            if (use_err): xm[mh] *= sqrt(1. - tlt*m*vxm[mh]/xm[mh]**2)
            else: xm[mh] *= sqrt(1. - tlt*m*vxm/xm[mh]**2)

        # reconstruction via inverse transformation
        xm[:] = xm[::-1].cumsum()
        x_recon[2*scl:]    += (2*xm[scl:-scl]-xm[:-2*scl]-xm[2*scl:])[::-1]/(2*scl)**2
        x_recon[scl:2*scl] += (2*xm[-scl:]-xm[-2*scl:-scl]-xm[:scl]-xm[-1])[::-1]/(2*scl)**2
        x_recon[:scl]      += (2*xm[:scl]-xm[-scl:]-xm[scl:2*scl]+xm[-1])[::-1]/(2*scl)**2

    return x_recon[l1:l1+ln0]
