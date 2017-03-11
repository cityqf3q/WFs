from __future__ import division
import numpy as np
import scipy
from scipy.integrate import quad
from scipy import linalg
from scipy import fftpack
import matplotlib.pyplot as plt

Gnum = 10                                # number of plane waves 
a = 5                                    # lattice constant(A)
G = np.arange(-np.pi/a*Gnum,np.pi/a*Gnum,2*np.pi/a)    # reciprocal lattice vector a=1
knum = 200                               # number of k/unit cells  
k = np.linspace(-np.pi/a,np.pi/a,knum)          # klist 
def V(x):                                # potential  
    return np.cos(2*np.pi*x/a)+1

# potential energy matrix elements
V_n = np.empty((Gnum,Gnum),dtype = complex)       # initialize

for i in range(0,Gnum):
    for j in range(0,Gnum):
        def real_f(x):
            return scipy.real(np.exp(1j*G[i]*x)*V(x)*np.exp(-1j*G[j]*x))
        def imag_f(x):
            return scipy.imag(np.exp(1j*G[i]*x)*V(x)*np.exp(-1j*G[j]*x))
        real_res, real_err = quad ((real_f) , 0 , a,limit=500)
        imag_res, imag_err = quad ((imag_f) , 0 , a,limit=500)
        res = real_res                                            # ignore the image part of this potential
        #print 'real_err=', real_err, '\n','imag_err=',imag_err
        V_n[i,j] = res/a
#print V_n

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
    H = KE+V_n*0.5 
    #print KE,V_n

    # eigen
    eigenvalue, eigenvector = linalg.eig(H)
    index = eigenvalue.argsort()
    eigenvalue = eigenvalue[index]          # put the energies in order
    eigenvector = eigenvector[index]
    EigVal.append(eigenvalue)
    Bloch_k.append(eigenvector)            # k:0-100, n:0-25
    #print "eigenvalue is\n", eigenvalue,'\n'
    #print "eigenvector is\n", eigenvector,'\n'
    
    # draw the band structure
for n in range(0,knum):
    for m in range(0,Gnum):
        E1 = EigVal[n]
        plt.scatter(k[n],E1[m])
plt.show()

# transform the Bloch function in ordinary representation
x = np.arange(0,20*a,0.1).reshape(1000,1)
G = G.reshape(1,10)
PWs = np.exp(1j*np.dot(x,(G)))   # matrix(1000,10)
BFs = np.zeros((1000,knum),dtype=complex)
BF1 = np.zeros((Gnum,Gnum),dtype=complex)
BF2 = np.exp(1j*np.dot(x,(k.reshape(1,knum))))
print x.shape,BF2.shape
#print PWs
for i in range(0,knum):
    BF1 = Bloch_k[15]
    BFs[:,i] = np.dot(PWs,BF1[:,0].reshape(10,1)).reshape(1000)
BFs = BFs*BF2
plt.plot(x,scipy.real(BFs[:,3]))
plt.show()

# calculate Wannier functions
WFs = 1/np.sqrt(knum)*np.dot(BFs,np.exp(-1j*k*10*a))
plt.plot(x,WFs)
plt.show()
