# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:47:54 2016

@author: Joseph
"""

import scipy as sp
from scipy import exp,pi,log,cos,sin
import sklearn as skl
import numpy as np
import time
from mayavi import mlab 

print('Starting now')
xx = sp.linspace(0,4*pi,1000);
ll=sp.linspace(1,0,2);
nn=sp.arange(-10,10,1)
an=np.random.randn(len(nn),2)

X,Y = np.meshgrid(xx,xx)
func=np.zeros(shape=[len(xx)],dtype=np.complex_)
funcl= [func for i in xrange(0,len(ll))]
funcl2= [func for i in xrange(0,len(ll))]
#funcg1=np.zeros_like(funcl,dtype=np.complex_)
#funcg2=np.zeros_like(funcl,dtype=np.complex_)


start=time.time()
for d in xrange(0,len(ll)):
    for xt in xrange(0,len(xx)):
        for n in xrange(0,len(nn)):
            L=1.0
            fun= 1j/2 * (exp(-pi*d*1j*(xx[xt]-nn[n]))/(xx[xt]-nn[n]) *(-1+exp(2*pi*1j*(ll[d]-1)*(xx[xt]-nn[n]))) + (exp(-pi*d*1j*(xx[xt]+nn[n]))/(xx[xt]+nn[n]) *(-1+exp(2*pi*1j*(ll[d]-1)*(xx[xt]+nn[n])))))
            #fun=  ((2*xx[xt]*cos(ll[d]*nn[n]/L*pi)*sin(ll[d]*xx[xt]*pi)-2*nn[n]/L*cos(ll[d]*xx[xt]*pi)*sin(ll[d]*nn[n]/L*pi))/(xx[xt]**2-(nn[n]/L)**2))
            funcl[d][xt]= funcl[d][xt] +  fun
            #L=5.5
            #fun=  ((2*xx[xt]*cos(ll[d]*nn[n]/L*pi)*sin(ll[d]*xx[xt]*pi)-2*nn[n]/L*cos(ll[d]*xx[xt]*pi)*sin(ll[d]*nn[n]/L*pi))/(xx[xt]**2-(nn[n]/L)**2))
            funcl2[d][xt]= funcl2[d][xt] +  fun

            #funcg1[d][xt]= funcg1[d][xt] + an[n,0] * fun
            #funcg2[d][xt]= funcg2[d][xt] + an[n,1] * fun
end=time.time()


print(end-start)

Z1= np.dot(funcl[0][:,None],funcl2[0][None,:])
#for i in xrange(0,len(xx)):
    
#Z2= np.dot(funcl[2][:,None],funcl2[1][None,:])
    
mlab.figure(bgcolor=(1,1,1))
mlab.mesh(X,Y,abs(Z1))
mlab.show()

#mlab.figure(bgcolor=(1,1,1))
#mlab.mesh(X,Y,abs(Z2))
#mlab.show()

end2=time.time()
print(end2-end)


'''clear

xx=linspace(0,2*pi,1000);
ll=linspace(0,1,100);
nn=-10:1:10;
an=randn(length(nn),1);
an2=randn(length(nn),1);

[X,Y]=meshgrid(xx,nn);
[Tx,Ty]=meshgrid(xx,xx);

figure(1);
for l=5:5:length(ll)
    for xt=1:1:length(xx)
        for n=1:1:length(nn)  
            %func{l}(n,xt)= (1/(xx(xt)^2-nn(n)^2)).*(exp(-i*ll(l)*xx(xt))*(nn(n)*sin(ll(l)*nn(n))-i*xx(xt)*cos(ll(l)*nn(n)))+exp(i*(ll(l)-2*pi)*xx(xt))*(nn(n)*sin(nn(n)*(ll(l)-2*pi)) + i*xx(xt)*cos(nn(n)*(ll(l)-2*pi))));
            %func2{l}(n,xt)= an(n)*func{l}(n,xt);
            if n==1
%                 func{l}(xt) = (-i)/(n-xx(xt)) * (exp(i*(2*pi-ll(l)*pi))-exp(i*ll(l)*pi));
                func{l}(xt)= (1/(xx(xt)^2-nn(n)^2)).*(exp(-i*ll(l)*xx(xt))*(nn(n)*sin(ll(l)*nn(n))-i*xx(xt)*cos(ll(l)*nn(n)))+exp(i*(ll(l)-2*pi)*xx(xt))*(nn(n)*sin(nn(n)*(ll(l)-2*pi)) + i*xx(xt)*cos(nn(n)*(ll(l)-2*pi))));
                func2{l}(xt) = an(n)*func{l}(xt);
                func3{l}(xt) = an2(n)*func{l}(xt);
            else
%                 func{l}(xt) = func{l}(xt) + (-i)/(n-xx(xt)) * (exp(i*(2*pi-ll(l)*pi))-exp(i*ll(l)*pi));
                func{l}(xt)= func{l}(xt) + (1/(xx(xt)^2-nn(n)^2)).*(exp(-i*ll(l)*xx(xt))*(nn(n)*sin(ll(l)*nn(n))-i*xx(xt)*cos(ll(l)*nn(n)))+exp(i*(ll(l)-2*pi)*xx(xt))*(nn(n)*sin(nn(n)*(ll(l)-2*pi)) + i*xx(xt)*cos(nn(n)*(ll(l)-2*pi))));
                func2{l}(xt) = func2{l}(xt) + an(n)*func{l}(xt);
                func3{l}(xt) = func3{l}(xt) + an2(n)*func{l}(xt);
            end
        end
    end
end
fprintf('test')
num=25;
for i=1:1:length(func{num})
    for le=1:1:length(func{num})
        T(i,le)= abs(func{num}(i) + func{num}(le));
        Ta(i,le)=abs(func2{num}(i) + func3{num}(le));
    end
end
fprintf('\n loop finished')

mesh(Tx,Ty,T)
figure(2)
mesh(Tx,Ty,Ta)
% contour(X,Y,abs(func{100}),30)
% ylabel('N value') 
% xlabel('k value')
% title('Contour plot of Fourier Transform of cos(nx) on [0,2*pi]')
% figure(2);    
% contour(X,Y,abs(func2{100}),30)Y
% ylabel('N value')
% xlabel('k value')   
% title('Contour plot of Fourier Transform of an*cos(nx) on [0,2*pi]')
%     '''
