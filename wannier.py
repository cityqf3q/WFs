from __future__ import division
import numpy as np
from scipy . integrate import quad
from scipy import linalg
from scipy import fftpack
import matplotlib.pyplot as plt

Gnum = 50                                # number of plane waves 
a = 5                                    # lattice constant(A)
G = np.arange(0,2*np.pi/a*Gnum,2*np.pi/a)    # reciprocal lattice vector a=1
knum = 100                               # number of k/unit cells  
k = np.linspace(0,2*np.pi/a,knum)          # klist 

def V(x):                                # potential  
    return np.cos(2*np.pi*x/a)

V_n = np.empty((Gnum,Gnum),dtype = complex)       # initialize

# potential energy matrix elements
for i in range(0,Gnum):
    for j in range(0,Gnum):
        def f(x):
            return np.exp(1j*G[i]*x)*V(x)*np.exp(-1j*G[j]*x)
        res , err = quad (f , 0 , a,limit=500)
        V_n[i,j] = res*1000/a

# KE diagonal  elements
KEd = np.zeros(Gnum)                # initialize

# Bloch function
EigVal = []                # a list of different k 
Bloch_k = []

for i in range(0,knum):
    for n in range(0,Gnum):
        KEd[n] = 1./2*(k[i]+G[n])**2        # h=1, m=1

    # KE 
    KE = np.diag((KEd))

    # Hamitonian
    H = KE + V_n
    #print H

    # eigen
    eigenvalue, eigenvector = linalg.eig(H)
    #index = eigenvalue.argsort()
    #eigenvalue = eigenvalue[index]          # put the energies in order
    #eigenvector = eigenvector[index]
    EigVal.append(eigenvalue)
    Bloch_k.append(eigenvector)            # k:0-100, n:0-25
    #print "eigenvalue is\n", EigVal,'\n'
    #print "eigenvector is\n", eigenvector,'\n'

# draw the bloch function
for n in range(0,knum):
    for m in range(0,5):
        E1 = EigVal[n]
plt.scatter(k[n],E1[m])
plt.show()

# trail function
c = 5
def g(x,n):
    return np.exp(-(x-n*a)**2/2*c**2)/np.sqrt(2*c**2*np.pi)          # a = 5

# draw the trail function        
x=np.arange(0,100,0.01)
plt.plot(x,g(x,5))

# fourier transformation using fft
#g_f = []
#step = 0.02
#time_vector = np.arange(0,1,step)                      # 50 freqs include >0, <0         
#sample_freq = fftpack.fftfreq(time_vector.size,step)
#positive = np.where(sample_freq >= 0)                  # find freqs >=0
#freqs = sample_freq[positive]
#for n in range(0,100):
#    g_fft = fftpack.fft(g(time_vector,n))                   # calculate fft
#    g_f.append(g_fft[positive])

#fourier transformation using integral
g_f0 = np.empty((Gnum),dtype=complex)
for m in range(0,Gnum):
    def f(x):
        return g(x,0)*np.exp(-1j*G[m]*x)
    res,err = quad(f,-3*c,3*c,limit=100)                    # 3*sigma integral
    g_f0[m] = res
#print g_f0.shape,g_f0.size,g_f0

# wave functions of first band
Bloch1 = np.empty((Gnum,knum),dtype=complex)                # 100k,first band(lowest energy,G_n=0)
B1 = np.empty((Gnum,Gnum),dtype=complex)
for i in range(0,knum):
    B1 = Bloch_k[i]
    Bloch1[:,i] = B1[:,0]
#print 'Bloch functions k=0 of first band is', Bloch1[:,0]
#print Bloch_k[0].shape
#print Bloch1.shape

# orthonormalize
AA_nm = np.empty((knum,knum),dtype=complex)
WFs = np.empty((Gnum,knum),dtype=complex)                          # wannier functions

for n in range(0,knum):
    for m in range(0,knum):
        for i in range(0,knum):
            AA_nm[n,m] = AA_nm[n,m]+np.exp(1j*k[i]*(m-n)*(np.dot(Bloch1[:,i],g_f0))**2)

for n in range(0,knum):
    for j in range(0,knum):
        for m in range(0,knum):
            WFs[:,n] = WFs[:,n]+(AA_nm[n,m])**(0.5)*(np.dot(Bloch1[:,i],g_f0))*np.exp(1j*k[i]*(-m))*Bloch1[:,j]

# convert WFs from matrix to function
W_n2 = 0
x = np.arange(0,500,0.01)
#for i in range(knum):
for j in range(Gnum):
    W_n2 =  W_n2+np.exp(1j*G[j]*x)*WFs[j,2]
