import numpy as np
import numpy.random as npr
from scipy import signal
import scipy.interpolate as spi
import matplotlib.pyplot as plt

## nombre de segments pour la discretisation
N=25

## grille : points milieu des segments
grid = (np.arange(N) + 0.5)/N - 0.5

## mesure de probabilité pi
## exemple 1 : distribution uniforme sur [-0.5,0.5]
pi = np.ones(N)/N
## exemple 2 : distribution gaussienne sur [-0.5,0.5]
#ecty = 5
#pi = signal.gaussian(N, std=ecty)/(ecty*np.sqrt(2*np.pi))

## fonction H : valeur absolue
def H(x):
    return np.abs(x)

## affichage
plt.figure(1,figsize=(12,6))
plt.plot(grid,pi,'-y')
plt.plot(grid,pi,'ob')
plt.title("mesure π")

plt.figure(2,figsize=(12,6))
plt.plot(grid,H(grid),'-b')
plt.plot(grid,H(grid),'or')
plt.title("fonction H")

plt.show()

## fonction convolution
def convol(a,b):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fft(a)*np.fft.fft(b)))

## fonction calcul de l'integrale (convolution avec padding)
def compute_integral(H, pi, grid):
    n = pi.shape[0]
    pad = int((n+1)/2)
    n_padded = 2*pad+n
    pi_padded = np.zeros(n_padded)
    pi_padded[pad:-pad] = pi
    S = grid[-1] + (grid[1]-grid[0])*(n_padded-n)/2.
    grid_padded = np.linspace(-S, S, n_padded)
    Hgrid = H(grid_padded)
    conv = convol(pi_padded, Hgrid)
    conv = conv[pad:-pad]
    return conv

## affichage
plt.figure(3,figsize=(12,6))
plt.plot(grid,np.real(compute_integral(H,pi,grid)),'-g')
plt.plot(grid,np.real(compute_integral(H,pi,grid)),'or')
plt.title("convolution de H et π avec padding")

plt.show()

## derivee de la fonction H (valeur absolue)
def H_prime(x):
    res = np.zeros_like(x)
    for i in range(x.shape[0]):
        if x[i]==0:
            res[i] = 0
        else:
            res[i] = x[i]/np.abs(x[i])
    return res

## fonction interpolation
def interp(grid,val,p):
    f = spi.interp1d(grid,val)
    return f(p)

## gradient de la fonction F
def Grad_F(p):
    grad = np.zeros_like(p)
    for i in range(p.shape[0]):
        grad[i] = np.sum(H_prime(p[i]-p))
    return grad/(N**2)

## gradient de la fonction G
def Grad_G(pi,grid,p):
    val = compute_integral(H_prime,pi,grid)
    return interp(grid,val,p)/N

## algorithme de descente de gradient pour minimiser J 
iter_max = 500
h = 1e-1
p = npr.uniform(-0.25,0.25,N)
for n_iter in range(iter_max):
    gF = Grad_F(p)
    gG = Grad_G(pi,grid,p)
    grad = np.real(gG-gF)
    p += -h*grad
    n_iter += 1
    if n_iter % 10 == 0 :
        plt.plot(grid,pi)
        plt.plot(p,np.zeros(p.shape[0]),'.b')
        plt.title("iter = "+str(n_iter))
        plt.show()
