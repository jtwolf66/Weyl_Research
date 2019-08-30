# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:58:10 2016

@author: Joseph
"""
import scipy as sp
from scipy import exp, pi, log, cos, sin
#from scipy.interpolate import RegularGridInterpolator as rgi
#import sklearn as skl
import numpy as np
import time
from mayavi import mlab
import peakutils
import itertools as it
#import matplotlib as mp
#from joblib import Parallel as par
#from joblib import delayed
import ipyparallel as ipp
#   from functools import partial

def F(x, y, z, n, d, cont=8, plot=1):   
    """[summary]
    
    [description]
    
    Arguments:
        x {[numpy.array]} -- [description]
        y {[numpy.array]} -- [description]
        z {[numpy.array]} -- [description]
        n {[integer]} -- [description]
        d {[float]} -- [description]
    
    Keyword Arguments:
        cont {number} -- [Number of contours to plot] (default: {8})
        plot {number} -- [If 0: don't plot, If 1: plot] (default: {1})
    
    Returns:
        [type] -- [description]
    """
   # p = multiprocessing.Pool(multiprocessing.cpu_count())

#    try:
#        plot == 0 or plot == 1
#    except ValueError:
#        print('Invalid value for plot. Please input 0 or 1.')
#
            
    def fun(x, n, d):
        return 1j/2 * (exp(-pi*d*1j*(x-n))/(x-n) *(-1+exp(2*pi*1j*(d-1)*(x-n))) + (exp(-pi*d*1j*(x+n))/(x+n) *(-1+exp(2*pi*1j*(d-1)*(x+n)))))
        
    X, Y, Z = np.mgrid[x[0]:x[-1]:len(x)*1j, y[0]:y[-1]:len(y)*1j, z[0]:z[-1]:len(z)*1j]

   # temp1, temp2, temp3 = np.zeros_like(x, dtype=np.complex_), np.zeros_like(y, dtype=np.complex_), np.zeros_like(z, dtype=np.complex_)
    
    '''for ns in xrange(-n, n):
        temp1 = np.add(fun(x, ns, d), temp1)
        temp2 = np.add(fun(y, ns, d), temp2)
        temp3 = np.add(fun(z, ns, d), temp3)
       '''
#mult
    V = np.zeros((len(x),len(y),len(z)), dtype=np.complex_)
    
    def Vgive(m):
        
        return np.dot(np.dot(fun(x,m[0],d)[:,None], fun(y,m[1],d)[None,:])[:,:,None], fun(z,m[2],d)[None,:])
    V=sum(p.map(map(Vgive,it.product(range(-n,n+1),repeat=3))))
    #Vtemp = par(n_jobs=-1,verbose=5)(delayed(Vgive)(m) for (m =! (0,0,0)) in it.prod-uct(range(-n,n+1),repeat=3))ma
    
    #V = sum(Vtemp)
    '''
    for m in it.product(range(-n,n+1),repeat=3):
        if (m[0] == m[1] == m[2] == 0) == False:
            Vtemp[:,:,:][] += np.dot(np.dot(fun(x,m[0],d)[:,None], fun(y,m[1],d)[None,:])[:,:,None], fun(z,m[2],d)[None,:])
'''
    #V = np.dot(np.dot(temp1[:,None], temp2[None,:])[:,:,None], temp3[None,:])
    if plot == 1:
        return mlab.contour3d(X, Y, Z, np.absolute(V), opacity=0.5, contours=cont)
    elif plot == 0:
        return X, Y, Z, V


def Fn(x, y, z, nl, d, nu='test', cont=8, plot=1):   
    '''Man page to be added'''
    if nu == 'test':
        nu = abs(nl)
#    try:
#        plot == 0 or plot == 1
#    except ValueError:
#        print('Invalid value for plot. Please input 0 or 1.')
        
    
    def fun(x,n,d):
        return 1j/2 * (exp(-pi*d*1j*(x-n))/(x-n) *(-1+exp(2*pi*1j*(d-1)*(x-n))) + (exp(-pi*d*1j*(x+n))/(x+n) *(-1+exp(2*pi*1j*(d-1)*(x+n)))))
    
    
    def Vgive(m):
        if (m[0] == m[1] == m[2] == 0) == False:
            return np.random.normal(scale=(1.0/((m[0]**2.0+m[1]**2.0+m[2]**2.0)**(3.0/4.0))))*np.dot(np.dot(fun(x,m[0],d)[:,None], fun(y,m[1],d)[None,:])[:,:,None], fun(z,m[2],d)[None,:])
        else:
            return 0

    X,Y,Z = np.mgrid[x[0]:x[-1]:len(x)*1j,y[0]:y[-1]:len(y)*1j,z[0]:z[-1]:len(z)*1j]

    #temp1, temp2, temp3 = np.zeros_like(x, dtype=np.complex_), np.zeros_like(y, dtype=np.complex_), np.zeros_like(z, dtype=np.complex_)
    
    V = np.zeros((len(x),len(y),len(z)))
    amr = v.map(Vgive, [m for m in it.product(range(nl,nu+1),repeat=3)])
    
    for i in xrange(0,len(amr)):
        V += amr[i]
    #V = sum(amr)
    #par(n_jobs=-1)(delayed(Vgive)(m) for m in it.product(range(n+1),repeat=3):
    #    if (m[0] == m[1] == m[2] == 0) == False:
    #        V = np.random.normal(scale=(1.0/((m[0]**2.0+m[1]**2.0+m[2]**2.0)**(3.0/4.0))))*np.dot(np.dot(temp1[:,None], temp2[None,:])[:,:,None], temp3[None,:]) + V

    if plot == 1:
        return mlab.contour3d(X, Y, Z, np.absolute(V), opacity=0.5, contours=cont)
    elif plot == 0:
        return X, Y, Z, V
        
        
if __name__ == '__main__':
    
    print 'Starting now'

    rc = ipp.Client()
    dv = rc[:]
    v = rc.load_balanced_view()

    # scatter 'id', so id=0,1,2 on engines 0,1,2
    dv.scatter('id', rc.ids, flatten=True)
    print("Engine IDs: ", dv['id'])

    xx = sp.linspace(0.0,2.0,20)
    yy = xx
    zz = xx

    lt = np.array([0.0])
    
    #Decide which mode you're interested in
    # mode == 0 -> Get data for peak finding
    # mode == 1 -> Plot
    
    mode = 0
    
    start = time.time()    
    if mode == 0:
        X, Y, Z, V = Fn(xx, yy, zz, 2, lt, plot=0)
        
        peaknum = 0
        V = np.absolute(V)
        V = V/np.nanmax(V)
        
        xind = peakutils.indexes(V[1:-1,1,1],thres=0.1)
        
        yind = [np.zeros_like(xind) for i in xrange(0,len(xind))]
        zind = []
        for i in xrange(0,len(xind)):
            yind[i] = peakutils.indexes(V[xind[i],1:-1,1],thres=0.1)
            zind.append([])
    
         
        for i in xrange(0,len(xind)):
            for j in xrange(0,len(yind[i])):
                zind[i].append(peakutils.indexes(V[xind[i],yind[i][j],1:-1],thres=0.1))
                peaknum = peaknum + len(zind[i][j])
        print 'Peaknum =', peaknum
        '''
        #xind = signal.find_peaks_cwt(np.absolute(V[:,1,1]), np.arange(.5,5,.1)*len(V[:,1,1]))
        xind = np.array([], dtype=int)
        for i in xrange(1,len(V[:,1,1])):
            if i == 0:
                if ((( V[i,1,1] > V[i,2,1]) and (V[i,1,1] > V[i,1,2])) and (V[i,1,1]) > (V[i+1,1,1])) == True:
                    xind = np.append(xind,i)
                
            elif i == len(V[:,1,1]) - 1:
                 if ((( V[i,1,1] > V[i,2,1]) and (V[i,1,1] > V[i,1,2])) and (V[i,1,1]) > (V[i-1,1,1])) == True:
                    xind = np.append(xind,i)
            else:
                if V[i,1,1] >= max(V[i-1:i+2,1,1]):
                    xind = np.append(xind,i)
        
        yind = [np.zeros_like(xind) for i in xrange(0,len(xind))]
            
        for i in xrange(1,len(xind)):
            for k in xrange(1,len(V[xind[i],:,1]))
                if xind[i] == 1:
                    if k == 1:
                        if V[xind[i],k,1] >= max(V[xind[i]:xind[i]+2,k:k+2,1:3]):
                            yind[i] = np.append(yind[i], k)
                    elif k == len(V[xind[1],:,1]) - 1:
                        if V[xind[i],:,1] >= max(V[xind[i]:xind[i]+2,k-1:k+1,1:3]):
                            yind[i] = np.append(yind[i], k)
                    else:
                        if V[xind[i],k,1] >= max(V[xind[i]:xind[i]+2,k-1:k+2,1:3]):
                            yind[i] = np.append(yind[i], k)
                
                elif xind[i] == len(V[:,1,1]):
                    if k == 1:
                        if V[xind[i],k,1] >= max(V[xind[i]-1:xind[i]+1,k:k+2,1:3]):
                            yind[i] = np.append(yind[i], k)
                    elif k == len(V[i,:,1]) - 1:
                        if V[xind[i],:,1] >= max(V[xind[i]-1:xind[i]+1,k-1:k+1,1:3]):
                            yind[i] = np.append(yind[i], k)
                    else:
                        if V[xind[i],k,1] >= max(V[xind[i]-1:xind[i]+1,k-1:k+2,1:3]):
                            yind[i] = np.append(yind[i], k)
            
                else:
                    if V[xind[i],k,1] >= max(V[xind[i]-1:xind[i]+2,k:k+2,1:3]):
                            yind[i] = np.append(yind[i], k)
                    elif k == len(V[i,:,1]) - 1:
                        if V[xind[i],:,1] >= max(V[xind[i]-1:xind[i]+2,k-1:k+1,1:3]):
                            yind[i] = np.append(yind[i], k)
                    else:
                        if V[xind[i],k,1] >= max(V[xind[i]-1:xind[i]+2,k-1:k+2,1:3]):
                            yind[i] = np.append(yind[i], k)
    
            zind = [np.zeros_like(yind) for i in xrange(0,len(yind))]
            
            for i in xrange(1,len(xind)):
                for k in xrange(1,len(yind[i])):
                    for m in xrange(1,len(V[xind[i],yind[i][k],:])):
                        if xind[i] == 1 and yind[i][k] == 1 and m == 1:
                            if V[xind[i],yind[i,k],m] >= max(V[1:3,1:3,1:3]):
                                zind[i][k] = np.append(zind[i][k], m)
                        elif xind[i] == len(V[:,1,1]) - 1 and yind[i][k] == len(V[1,:,1]) -1 and m == len(V[1,1,:]) - 1:
                            if V[xind[i],yind[i,k],m] >= max(V[xind[i]-1:xind[i]+1, yind[i][k]-1:yind[i][k]+1,m-1:m+1]):
                                zind[i][k] = np.append(zind[i][k], m)
                        elif xind[i] == 1:
                            if yind[i][k] == 1:
                                if m == len(V[1,1,:]) - 1:
                                    if V[xind[i],yind[i,k],m] > max(V[1:3,1:3,m-1:m+1]):
                                        zind[i][k] = np.append(zind[i][k], m)
                                elif V[xind[i],yind[i,k],m] > V[1:3,1:3,m-1:m+2]:
                                    zind[i][k] = np.append(zind[i][k], m)
                            elif yind[i][k] == len(V[1,:,1]) - 1:
                                if m == len(V[1,1,:]) - 1:
                                    if V[xind[i],yind[i,k],m] > max(V[1:3,yind[i][k]-1:yind[i][k]+1,m-1:m+1]):
                                        zind[i][k] = np.append(zind[i][k], m)
       ##EDIT FROM HERE                         elif V[xind[i],yind[i,k],m] > V[1:3,yind[i][k]-1:yind[i][k]+1,m-1:m+2]:
                                    zind[i][k] = np.append(zind[i][k], m)
                '''                
                    
    elif mode == 1:
        Fn(xx, xx, xx, 10, lt, plot = 1)
        mlab.show()
    
    else:
        print 'Choose valid mode, ',mode, 'is not valid.' 
    
    end = time.time()
    
    print end - start
    
    #mlab.show()
    
    print 'Now we\'re here'
    
    #Z2= np.dot(funcl[2][:,None],funcl2[1][None,:])
        
    #mlab.figure(bgcolor=(1,1,1))
    #mlab.mesh(X,Y,abs(Z1))
    #mlab.show()
    
    #mlab.figure(bgcolor=(1,1,1))
    #mlab.mesh(X,Y,abs(Z2))
    #mlab.show()
    
