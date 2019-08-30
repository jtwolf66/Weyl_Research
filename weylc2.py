#import matplotlib as mp
import numpy as np
import time
import scipy as sp
from scipy import pi
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from mayavi import mlab


def weyl(dom, nv, size, angles,supr=True):
    """
    weyl(dom, nv, size, angles) computes the Eigenvalues for 3
        domains, viz. 3-Torus, Half-Turn 3-Torus, and Quarter-Turn
        3-Torus. It then counts the number of eigenvalues, and
        returns them to you

    Parameters
    ----------
    dom : np.array
        dom contains the domains you wish to calculate
        and count the eigenvalues of
    nv : int
        parameter that determines how many eigenmodes are calculated
    size : np.array
        The size
    angles : np.array
        The angles
    supr : Boolean
        parameter that determines whether or not to supress printing
        the text

    Returns
    -------
    ANS = weyl(dom,nv,size,angles)
    ANS[0] = L (list): All calculated eigenvalues
    ANS[1] = LHS (list): Number of eigenvalues less than or equal to the
                         unique eigenvalue at the same index
    ANS[2] = RHSV (list): cV * V * \lambda^(3/2)
    ANS[3] = RHSA (list): cA * SA * \lambda
    ANS[4] = RHS (list): RHSV + RHSA
    ANS[5] = S (list): unique eigenvalues

        The length of each of these lists is equal to the length of dom.
        Each list contains a numpy array at each of its indexes.

    ANS[6] = Vol (float): Volume of the fundamental domain
    ANS[7] = SA (float): Surface area of the fundamental domain

    """

    #  Define constants
    cV = (2.0 * pi)**(-3.0) * (4.0 * pi / 3.0)
    cA = (pi / 4.0) * (2.0 * pi)**(-2.0)

    #  Define Arrays
    L = list()
    Lnan = list()
    zer = list()
    S = list()
    LHS = list()
    RHS = list()
    RHSA = list()
    RHSV = list()

    for i in xrange(0, len(dom)):
        L.append([])
        Lnan.append([])
        zer.append([])
        S.append([])
        LHS.append([])
        RHS.append([])
        RHSA.append([])
        RHSV.append([])

    T = from_parameters(size[0], size[1], size[2], angles[0], angles[1], angles[2])
    #Josh's
    # To = np.array([[size[0], size[1]*np.cos(angles[0]), size[2]*np.cos(angles[2])*np.sin(angles[1])],
    #               [0, size[1]*np.sin(angles[0]), size[2]*np.sin(angles[2])*np.sin(angles[1])],
    #               [0, 0, size[2]*np.cos(angles[1])]])



    SA = 2*size[2] * size[0] * np.sin(angles[1]) + 2 * size[2] * size[1] * np.sin(angles[0]) + 2 * size[1] * size[0] *np.sin(angles[2])
    Vol = abs(np.dot(np.cross(T[0],T[1]),T[2]))


    #Define Domains


    #Standard 3 Torus

    def eigend(T,d,size,angles,m,supr=True):
        """
        eigend(T,d,angles,size,m,supr) computes the eigenvalues
        for different domains for eigenmodes (-m,-m,-m) to (m,m,m)
        and excludes values that do not fit within the sphere.

        Parameters
        ----------
        T : np.array
            T is the translation matrix that defines the basis
            vectors for your domain
        d : int
            d contains the domain you wish to calculate
            and count the eigenvalues of
        size : np.array
            The size
        angles : np.array
            The angles
        m : int
            parameter that determines how many eigenmodes are calculated
        supr : Boolean
            parameter that determines whether or not to supress printing
            the text

        Returns
        -------
        Eigen : np.array

        """
        
        # Define Zero array to remove impossible eigenmode
        zero_array = np.array([0,0,0])


        DOM = [[],[],[]]
        DOM[0] = np.linalg.inv(T)
        DOM[1] = np.linalg.inv(np.dot(T,np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])))
        DOM[2] = np.linalg.inv(np.dot(T,np.array([[0, -1, 0], [1, 0, 0,], [0, 0, 1]])))


        #set radius

        radius = np.min(np.sum(np.square(np.dot(DOM[0],(m*np.identity(3)))),axis=0))

        #Standard 3-Torus
        if d == 0:

            nvector = np.arange(-m,m+1,1)
            nst = cartesian([nvector,nvector,nvector])
            nset = np.delete(nst,np.argwhere(np.all((nst-zero_array) == 0,axis=1)),axis=0)

            Tret = np.sum(np.square(np.dot(DOM[0],nset.transpose())), axis = 0)


        #Half-Turn 3-Torus
        elif d == 1:

            npints = np.arange(1,m+1,1)
            nfull = np.arange(-m,m+1,1)


            nset1 = cartesian([npints, nfull, nfull])
            nset2 = cartesian([np.array([0]), npints, nfull])

            nsetlamb1 = np.concatenate((nset1,nset2),axis=0)

            lowb = -m - (-m % 4) + 2
            nvecz2 = np.arange(lowb,m+1,2)
            nset3 = cartesian([np.array([0]),np.array([0]),nvecz2])

            Tret1 = np.sum(np.square(np.dot(DOM[0],nsetlamb1.transpose())),axis=0)
            Tret2 = np.sum(np.square(np.dot(DOM[0],nset3.transpose())), axis=0)

            Tret = np.concatenate((Tret1,Tret2),axis=0)


        #Quarter-Turn 3-Torus
        elif d == 2:

            nxvector = np.arange(1,m+1,1)
            nyvector = np.arange(0,m+1,1)
            nzvector = np.arange(-m,m+1,1)
            lowb = -m - (-m % 4) + 4
            nz2vector = np.arange(lowb,m+1,4)

            nst = cartesian([nxvector,nyvector,nzvector])
            nset = np.delete(nst,np.argwhere(np.all((nst-zero_array) == 0,axis=1)),axis=0)
            nset2 = cartesian([np.array([0]),np.array([0]),nz2vector])

            Tret1 = np.sum(np.square(np.dot(DOM[0],nset.transpose())),axis=0)
            Tret2 = np.sum(np.square(np.dot(DOM[0],nset2.transpose())), axis=0)

            Tret = np.concatenate((Tret1,Tret2),axis=0)

        else:
            return

        outr = np.where(Tret > radius )
        if supr == False:
            print 'Domain: %d   Radius: %d  N(\lambda): %d' % (d, radius, len(Tret) - len(outr[0]))
        #print 'Eliminated %d Values' % (len(outr[0]))

        Tretf = np.delete(Tret,outr)
        Eigen = np.multiply(((2.0*pi)**2.0),Tretf)
        return Eigen

#Sort our list of eigenvalues, count and remove zeroes for log-log plot
    for i in xrange(0,len(dom)):
        L[i] = eigend(T,dom[i],angles,size,nv,supr)

        L[i] = np.sort(L[i],axis=0)

        zer[i] = (np.where(L[i] == 0))[0]

        if len(zer[i]) > 0:
            L[i] = np.delete(L[i],zer[i])

        Lnan[i] = np.where(np.isnan(L[i])==True)[0]
        L[i] = np.delete(L[i],Lnan[i])

        S[i], LHS[i] = np.unique(L[i],return_counts=True)

        RHS[i], RHSV[i], RHSA[i] = np.zeros(len(S[i])), np.zeros(len(S[i])), np.zeros(len(S[i]))

        LHS[i][0] += len(zer[i])

        for j in xrange(1,len(LHS[i])):
            LHS[i][j] += LHS[i][j-1]

        #Calculate the Volume and SA terms
        RHSV[i] = np.multiply(cV * Vol, np.power(S[i], (3.0/2.0)))
        RHSA[i] = np.multiply(cA * SA, S[i])
        RHS[i] = np.add(RHSV[i],RHSA[i])
        
        if float(len(L[2])) != 0:
            ratio = float(len(L[0]))/float(len(L[2]))
        else:
            ratio = None

    return L, LHS, RHSV, RHSA, RHS, S, Vol, SA
    
def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.H * m)
    
def funcd(x,a):
    #ccv = [[1.],[2.],[4.]]

    cA = (pi / 4.0) * (2.0*pi)**(-2.0)
    cV =   (2.0*pi)**(-3.0) * (4.0*pi/3.0)

    ANS = np.zeros(len(x))
    ANS[0] = x[0]
    AV = np.multiply( cV * Vol,np.power(x[1:],(3.0/2.0)))
    AA = np.multiply((cA * SA), x[1:])
    ANS[1:] = np.add(AV,a * AA)
    return ANS

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.

    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.

    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def from_parameters(a, b, c, alpha, beta, gamma):
    """
    Create a Lattice using unit cell lengths and angles (in degrees).

    Args:
        a (float): *a* lattice parameter.
        b (float): *b* lattice parameter.
        c (float): *c* lattice parameter.
        alpha (float): *alpha* angle in degrees.
        beta (float): *beta* angle in degrees.
        gamma (float): *gamma* angle in degrees.

    Returns:
        Lattice with the specified lattice parameters.
    """
    alpha_r = alpha
    beta_r = beta
    gamma_r = gamma
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r))\
        / (np.sin(alpha_r) * np.sin(beta_r))
    #Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)
    vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
    vector_b = [-b * np.sin(alpha_r) * np.cos(gamma_star),
                b * np.sin(alpha_r) * np.sin(gamma_star),
                b * np.cos(alpha_r)]
    vector_c = [0.0, 0.0, float(c)]
    #print gamma_star
    return np.array([vector_a, vector_b, vector_c])


def get_points_in_sphere(kT, T, frac_points, center, r):
    """
    Find all points within a sphere from the point taking into account
    periodic boundary conditions. This includes sites in other periodic
    images.

    Algorithm:

    1. place sphere of radius r in crystal and determine minimum supercell
       (parallelpiped) which would contain a sphere of radius r. for this
       we need the projection of a_1 on a unit vector perpendicular
       to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
       determine how many a_1"s it will take to contain the sphere.

       Nxmax = r * length_of_b_1 / (2 Pi)

    2. keep points falling within r.

    Args:
        frac_points: All points in the lattice in fractional coordinates.
        center: Cartesian coordinates of center of sphere.
        r: radius of sphere.

    Returns:
            fcoords, dists, inds
    """

    recp_len = np.sqrt(np.sum(kT ** 2, axis=1))
    nmax = float(r) * recp_len + 0.01

    pcoords =   np.dot(center,kT)
    center = np.array(center)

    n = len(frac_points)
    fcoords = np.array(frac_points) % 1
    indices = np.arange(n)

    mins = np.floor(pcoords - nmax)
    maxes = np.ceil(pcoords + nmax)
    arange = np.arange(start=mins[0], stop=maxes[0])
    brange = np.arange(start=mins[1], stop=maxes[1])
    crange = np.arange(start=mins[2], stop=maxes[2])
    arange = arange[:, None] * np.array([1, 0, 0])[None, :]
    brange = brange[:, None] * np.array([0, 1, 0])[None, :]
    crange = crange[:, None] * np.array([0, 0, 1])[None, :]
    images = arange[:, None, None] + brange[None, :, None] +\
        crange[None, None, :]

    shifted_coords = fcoords[:, None, None, None, :] + \
        images[None, :, :, :, :]

    cart_coords = np.dot(fcoords,T)
    cart_images = np.dot(images,T)

    coords = cart_coords[:, None, None, None, :] + \
        cart_images[None, :, :, :, :]
    coords -= center[None, None, None, None, :]
    coords **= 2
    d_2 = np.sum(coords, axis=4)

    within_r = np.where(d_2 <= r ** 2)
    return shifted_coords[within_r], np.sqrt(d_2[within_r]), indices[within_r[0]]

def plot_domains(fig,size,angles,LHS,RHS,RHSV,RHSA,nradius,dom):
    """
    Example:
    figure = plt.figure()
    plot_domains(figure, size, angles, LHS, RHS, RHSV, RHSA, n, dom)
    plt.show(figure)

    where

    Parameters
    ----------
    fig : figure
        figure handle of matplotlib figure
     size : np.array
        The size
    angles : np.array
        The angles
    LHS : list
        list containing a np.array of the number of eigenvalues <= to the correspending
        RHS eigenvalue for each domain
    RHS : list
        list containing a np.array of the RHS of the weyl conjecture for each domain
    RHSV : list
        list containing a np.array of the RHS of the volume term in the weyl conjecture
        for each domain
    RHSA : list
        list containing a np.array of the RHS of the area term in the weyl conjecture
        for each domain
    nradius : int
        radius of eigenmodes in fractional coordinates
    dom : np.array
        Numpy array containing the domains plotted

    Returns
    -------
    fig : figure handle of matplotlib figure
    """

    fig.suptitle("Weyl Conecture, L = (%s,%s,%s),  Angles = (%d,%d,%d), n=%d" % (
        size[0], size[1], size[2], angles[0] * 180 / pi, angles[1] * 180 / pi, angles[2] * 180 / pi, nradius,)
        , fontsize=16)

    subplot_num = len(dom)

    for i in xrange(0,subplot_num):
        plt.subplot(subplot_num,1,i+1)
        #  Plot a straight line for the Weyl conjecture
        plt.plot([0, max(sp.log(RHS[i]))], [0, max(sp.log(RHS[i]))], 'b', label='Ideal')
        #  Plot the volume term of the Weyl conjecture vs number of eigenvalues
        plt.plot(sp.log(RHSV[i]), sp.log(LHS[i]), 'ro', label='Volume Term')
        #  Plot the area term of the Weyl conjecture vs number of eigenvalues
        plt.plot(sp.log(RHSA[i]), sp.log(LHS[i]), 'yo', label='Area Term')
        #  Plot the Weyl conjecture
        plt.plot(sp.log(RHS[i]), sp.log(LHS[i]), 'go', label='WC Actual')

        plt.title("Domain = %d" % (dom[i]))
        plt.xlabel('log(RHS)')
        plt.ylabel("log(N)")
        if i == 0:
            plt.legend()
    #  Label the Axes and add title
    return fig


'''

# Define Constants to calculate the weyl conjecture
n=30
dom=np.array([0,1,2])
size = np.array( [1.0, 1.0, 1.0] )
angles = np.array( [pi/2.0, pi/2.0, pi/2.0] )

# Define constants and arrays for use with the get_points_in_sphere function
fp = np.array([[0,0,0],[size[0],0,0],[size[0],size[1],0],[size[0],size[1],size[2]],[size[0],0,size[2]],[0,size[1],0],[0,size[1],size[2]],[0,0,size[2]]])
TT = from_parameters(size[0],size[1],size[2],angles[0],angles[1],angles[2])
kTT = np.linalg.inv(TT)


L, LHS, RHSV, RHSA, RHS, S, Vol, SA = weyl(dom,n,size,angles,supr=True)

print 'Made it this far'
figure = plt.figure()
plot_domains(figure, LHS, RHS, RHSV, RHSA, n, dom)
plt.show(figure)
print 'Finished everything'
#  prints ratio between 1st and 3rd domain
#  print float(len(L[0]))/float(len(L[2]))



'''

'''
xx = np.arange(0.8,pi/2.0,0.05)
XT,YT,ZT = np.mgrid[0.8:pi/2.0:0.05,0.8:pi/2.0:0.05,0.8:pi/2.0:0.05]
Surf = np.zeros((len(xx),len(xx),len(xx)))
for i in xrange(0,len(xx)):
    print 'on row %d' % (i)
    for j in xrange(0,len(xx)):
        for k in xrange(0,len(xx)):
            Surf[i][j][k] = weyl(dom,n,size,[xx[i],xx[j],xx[k]])[-1]
            
SPlot = mlab.contour3d(100.0*XT, 100.0*YT, 100.0*ZT, Surf, opacity=0.5, contours=15)
mlab.show()
'''
'''
for i in xrange(0,len(dom)):

    plt.figure()
    plt.plot([0,max(sp.log(RHS[i]))],[0,max(sp.log(RHS[i]))],'b')
    plt.plot(sp.log(RHSV[i]),sp.log(LHS[i]),'ro')
    plt.plot(sp.log(RHS[i]),sp.log(LHS[i]),'go')
    plt.xlabel('log(RHS)')
    plt.ylabel("log(N)")
    plt.title("Weyl Conj, L = (%s,%s,%s),  Angles = (%d,%d,%d), n=%d, dom=%d" % (size[0],size[1],size[2],angles[0]*180/pi,angles[1]*180/pi,angles[2]*180/pi,n,i))
    #plt.draw()


#Now we must add fitting functions



step = [[],[],[]]
popta = [[],[],[]]
poptb = [[],[],[]]
poptr = [[],[],[]]
pcovr = [[],[],[]]
comp = [[],[],[]]
#for i in dom:
for i in xrange(0,len(dom)):
    step[i] = np.zeros(len(S[i])+1)
    step[i][0] = i
    step[i][1:] = S[i]
    comp[i] = np.zeros(len(LHS[i])+1)
    comp[i][0] = i
    comp[i][1:] = LHS[i]

    poptr[i], pcovr[i] = curve_fit(funcd,step[i],comp[i],p0=[1.0],bounds=(0,sp.inf))#,bounds=(0,sp.inf))

    popta[i] = poptr[i][0]
    #poptb[i] = poptr[i][1]
    print 'popt is '
    print poptr[i]

'''
'''
plt.figure()
plt.plot(sp.log(funcd(step[i],popta[i])[1:]),sp.log(LHS[i]),'ro')
plt.plot(sp.log(funcd(step[i],0)[1: ]),sp.log(LHS[i]),'go')
plt.plot(sp.log(funcd(step[i],1)[1:]),sp.log(LHS[i]),'yo')
plt.plot([0,max(sp.log(funcd(step[i],popta[i])[1:]))],[0,max(sp.log(funcd(step[i],popta[i])[1:]))],'b')
plt.plot([0,max(sp.log(LHS[i]))],[0,max(sp.log(LHS[i]))],'m')
'''