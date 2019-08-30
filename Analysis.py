import weylc2 as wc
import numpy as np
from scipy import pi
import matplotlib.pyplot as plt

# Parameters to be studied.
n=65
dom=np.array([0,1,2])
size = np.array( [1.0, 1.0, 1.0] )
angles = np.array( [pi/2.0, pi/2.0, pi/2.0] )
L, LHS, RHSV, RHSA, RHS, S, Vol, SA = wc.weyl(dom,n,size,angles,supr=True)

fig = plt.figure()
wc.plot_domains(fig, size, angles, LHS, RHS, RHSV, RHSA, n, dom)
plt.show(fig)