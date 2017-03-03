from __future__ import division
import numpy as np
from scipy . integrate import quad
from scipy import linalg
from scipy import fftpack
#import matplotlib.pyplot as plt

G=np.arange(0,50*np.pi,2*np.pi)    # reciprocal lattice vector a=1

k = np.linspace(0,2*np.pi,100)     # klist 

def V(x):                          # potential  
    return np.cos(2*np.pi*x)

V_n = np.empty((25,25),dtype = complex)       # initialize

# potential energy matrix elements
for i in range(0,25):
    for j in range(0,25):
        def f(x):
            return np.exp(1j*G[i]*x)*V(x)*np.exp(-1j*G[j]*x)
        res , err = quad (f , 0 , 1)
        V_n[i,j] = res

# KE diagonal  elements
KEd = np.empty((25),dtype=float)                # initialize

# Bloch function
psi_n = np.empty((25,25),dtype=float)           # has 25 Gn 

EigVal = []
Bloch_k = []
for i in range(0,100):
    for n in range(0,25):
        KEd[n] = 1./2*(k[i]+G[n])**2        # h=1, m=1

    # KE 
    KE = np.diag((KEd))

    # Hamitonian
    H = KE + V_n

    # eigen
    eigenvalue, eigenvector = linalg.eig(H)
    EigVal.append(eigenvalue)
    Bloch_k.append(eigenvector)            # k:0-100, n:0-25
#print "eigenvalue is", EigVal
    #print "eigenvector is", eigenvector
#print Bloch_k[80]

# trail function
c = 1
def g(x,n):
    return np.exp(-(x-n)**2/2*c**2)/np.sqrt(2*c**2*np.pi)          # a = 1

# fourier transformation using fft
#g_f = []
#step = 0.02
#time_vector = np.arange(0,1,step)                      # 50 freqs include >0, <0         
#sample_freq = fftpack.fftfreq(time_vector.size,step)
#positive = np.where(sample_freq >= 0)                  # find freqs >=0
