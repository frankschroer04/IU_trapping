
# This is a notebook to replace the Mathematica code that processes our electrostatic data from COMSOL. 
# It takes COMSOL data and calculates the electrostatic potentials and secular frequencies.

# %% importing modules libraries


# @title Run this

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import scipy.constants as scipy

from scipy.optimize import curve_fit

import pandas as pd
from io import StringIO
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.ndimage import rotate



from matplotlib import colors

from scipy.integrate import odeint
from scipy.optimize import minimize, fsolve

from numba import jit
import time


# %% text file loading


# @title Files
dc1xy_file = 'Electrode1_XY.txt'
dc1zy_file = 'Electrode1_YZ.txt'
dc1zx_file = 'Electrode1_ZX.txt'

dc2xy_file = 'Electrode2_XY.txt'
dc2zy_file = 'Electrode2_YZ.txt'
dc2zx_file = 'Electrode2_ZX.txt'

dc3xy_file = 'RF1_XY.txt'
dc3zy_file = 'RF1_YZ.txt'
dc3zx_file = 'RF1_ZX.txt'

rfxy_file = 'RF_ENorm_XY.txt'
rfzy_file = 'RF_ENorm_YZ.txt'
rfzx_file = 'RF_Enorm_ZX.txt'


# @title Load Data Function
def load_data(file_name):
  #load data
  data1 = np.loadtxt(file_name)

  #unravel data
  x = np.array(data1[:,0])
  y = np.array(data1[:,1])
  z = np.array(data1[:,2])
  f = np.array(data1[:,3])

  #format data
  data2 = [[x[i],y[i],z[i],f[i]] for i in range(len(x))]

  #delete constant axis
  summed = np.sum(data2,axis=0)
  zero_axis = np.where(summed==0.0)
  data_new = np.delete(data2,zero_axis,axis=1)

  return data_new


DC1xy_data = load_data(dc1xy_file)
DC1zy_data = load_data(dc1zy_file)
DC1zx_data = load_data(dc1zx_file)

DC2xy_data = load_data(dc2xy_file)
DC2zy_data = load_data(dc2zy_file)
DC2zx_data = load_data(dc2zx_file)

DC3xy_data = load_data(dc3xy_file)
DC3zy_data = load_data(dc3zy_file)
DC3zx_data = load_data(dc3zx_file)

RFxy_data = load_data(rfxy_file)
RFzy_data = load_data(rfzy_file)
RFzx_data = load_data(rfzx_file)


# %% Variables and constants

#constants
q = scipy.elementary_charge
eps_o = scipy.epsilon_0
M_Yb = 171*1.67262192e-27
m = 171*1.67262192e-27
d =  1000 #mm/m distance from
frf = 38.6e6
wrf = 2*np.pi*frf

#voltage values
v_rf = 299.8662149
Vdc= [20.23390575, 12.55475617, 16.15436475] #endcaps, second middle, center electrode voltages
DC1 = Vdc[0]
DC2 = Vdc[1]
DC3 = Vdc[2]

#coordiante ranges
xCtr = 0.
xRange = .5
yCtr = 0.
yRange = 0.06
zCtr = 0.
zRange = 0.06;
xMin = xCtr - xRange/2
xMax = xCtr + xRange/2
yMin = yCtr - yRange/2
yMax = yCtr + yRange/2
zMin = zCtr - zRange/2
zMax = zCtr + zRange/2

# %% Voltage functions

def V_calc(q1,q2,data_in):
  #transpose and unpack data
  data_in = data_in.T
  x1 = data_in[0]
  x2 = data_in[1]
  f = data_in[2]
  #prepare points
  qs = np.array([q1,q2]).T
  points = np.array([x1,x2]).T
  #interpolate
  vals = interpolate.griddata(points, f, qs, \
                            method='cubic',fill_value=np.NaN)

  return vals


#DC1 endcaps
def DC1xyf(x,y):
  val = V_calc(x,y,DC1xy_data)
  return val

def DC1zyf(y,z):
  val = V_calc(y,z,DC1zy_data)
  return val

def DC1zxf(x,z):
  val = V_calc(x,z,DC1zx_data)
  return val

#DC2 middle centers
def DC2xyf(x,y):
  val = V_calc(x,y,DC2xy_data)
  return val

def DC2zyf(y,z):
  val = V_calc(y,z,DC2zy_data)
  return val

def DC2zxf(x,z):
  val = V_calc(x,z,DC2zx_data)
  return val

#DC3 center electrode
def DC3xyf(x,y):
  val = V_calc(x,y,DC3xy_data)
  return val

def DC3zyf(y,z):
  val = V_calc(y,z,DC3zy_data)
  return val

def DC3zxf(x,z):
  val = V_calc(x,z,DC3zx_data)
  return val

#RF
def RFxyf(x,y):
  val = V_calc(x,y,RFxy_data)
  return val

def RFzyf(y,z):
  val = V_calc(y,z,RFzy_data)
  return val

def RFzxf(x,z):
  val = V_calc(x,z,RFzx_data)
  return val

# @title Electrostatic functions

#dc potentials
def DCxy(x,y):
  val = DC1*DC1xyf(x,y)+DC2*DC2xyf(x,y)+DC3*DC3xyf(x,y)
  return val

def DCzx(x,z):
  val = DC1*DC1zxf(x,z)+DC2*DC2zxf(x,z)+DC3*DC3zxf(x,z)
  return val

def DCzy(y,z):
  val = DC1*DC1zyf(y,z)+DC2*DC2zyf(y,z)+DC3*DC3zyf(y,z)
  return val

#transform the imported RF data to the rf psuedopotential...
def Vrf_xy(x,y):
  val = q * (RFxyf(x,y)**2) * v_rf**2 / (4 * M_Yb * (wrf**2))
  return val

def Vrf_zx(x,z):
  val = q * (RFzxf(x,z)**2) * v_rf**2 / (4 * M_Yb * (wrf**2))
  return val

def Vrf_zy(y,z):
  val = q * (RFzyf(y,z)**2) * v_rf**2 / (4 * M_Yb * (wrf**2))
  return val

#total potentials
def Uxy(x,y):
  val = DCxy(x,y) + Vrf_xy(x,y)
  return val

def Uzx(x,z):
  val = DCzx(x,z) + Vrf_zx(x,z)
  return val

def Uzy(y,z):
  val = DCzy(y,z) + Vrf_zy(y,z)
  return val


# **1D Plots**

# %% 1D Axial potential plotting 

#Axial Trapping along X-axis

n_vals = 1000
xvals = np.linspace(xMin,xMax,n_vals)
yvals = np.ones(n_vals)*yCtr
zvals = np.ones(n_vals)*zCtr

Uxy_vals = Uxy(xvals,yvals) #this is interpolated from a data set
Uzx_vals = Uxy(xvals,zvals) #this is interpolated from a data set


plt.plot(xvals,Uxy_vals,color='blue',label = 'Uxy',alpha = .3)
plt.plot(xvals,Uzx_vals,color='red',label = 'Uzx',alpha = .3)
plt.xlabel('x (mm)')
plt.ylabel('Potential (eV)')
plt.legend()
plt.show()

trap_depth = np.max(Uxy_vals)-np.min(Uxy_vals)
print("trap depth: " + str(np.round(trap_depth,4))+" eV")


# @title Minimum Position in x
n_vals = 1000
xvals = np.linspace(xMin,xMax,n_vals)
yvals = np.ones(n_vals)*yCtr

Uxy_vals = Uxy(xvals,yvals) #this is interpolated from a data set

# Create the 1D interpolating function
Uxy_1D_inter_x = interpolate.interp1d(xvals, Uxy_vals, kind='cubic', fill_value='extrapolate')

#create functional input for the 1D interpolated function
def Uxy_1D_inter_fun_x(x):
  return Uxy_1D_inter_x(x)

init_guess_x = 1e-10
result_x = minimize(Uxy_1D_inter_fun_x, init_guess_x, method='Nelder-Mead')
print("Minimum value found (eV):", np.round(result_x.fun,4))
print("At the point (um):", np.round((result_x.x-xCtr)*1000,10))

# @title Axial trapping along x-axis close to center
n_vals = 1000
xvals = np.linspace(xMin,xMax,n_vals)
yvals = np.ones(n_vals)*(yCtr+yRange/10)
zvals = np.ones(n_vals)*zCtr

Uxy_vals = Uxy(xvals,yvals)
Uzx_vals = Uxy(xvals,zvals)


plt.plot(xvals,Uxy_vals,color='blue',label = 'Uxy',alpha = .3)
plt.plot(xvals,Uzx_vals,color='red',label = 'Uzx',alpha = .3)
plt.xlabel('x (mm)')
plt.ylabel('Potential (eV)')
plt.legend()
plt.show()


# %% 1D Radial potential plotting


# @title Radial trapping potential along y-axis
n_vals = 1000
xvals = np.ones(n_vals)*xCtr
yvals = np.linspace(yMin,yMax,n_vals)
zvals = np.ones(n_vals)*zCtr

DC_xy = DCxy(xvals,yvals)
RF_xy = Vrf_xy(xvals,yvals)
Uxy_vals = Uxy(xvals,yvals)


plt.plot(yvals,DC_xy,color='blue',label = 'DC',alpha = .3)
plt.plot(yvals,RF_xy,color='red',label = 'RF',alpha = .3)
plt.plot(yvals,Uxy_vals,color='green',label = 'DC + RF',alpha = .3)
plt.xlabel('y (mm)')
plt.ylabel('Potential (eV)')
plt.legend()
plt.show()

trap_depth = np.max(DC_xy)-np.min(DC_xy)
print("trap depth: " + str(np.round(trap_depth,4))+" eV")


# @title Mininum trapping position in y

n_vals = 1000
xvals = np.ones(n_vals)*xCtr
yvals = np.linspace(yMin,yMax,n_vals)

Uxy_vals = Uxy(xvals,yvals) #this is interpolated from a data set

# Create the 1D interpolating function
Uxy_1D_inter_y = interpolate.interp1d(yvals, Uxy_vals, kind='cubic', fill_value='extrapolate')

#create functional input for the 1D interpolated function
def Uxy_1D_inter_fun_y(y):
  return Uxy_1D_inter_y(y)

init_guess_y = 1e-10
result_y = minimize(Uxy_1D_inter_fun_y, init_guess_y, method='Nelder-Mead')
print("Minimum value found (eV):", np.round(result_y.fun,4))
print("At the point (um):", np.round((result_y.x-yCtr)*1000,10))

# @title Radial Trapping along z-axis
n_vals = 1000
xvals = np.ones(n_vals)*xCtr
yvals = np.ones(n_vals)*yCtr
zvals = np.linspace(zMin,zMax,n_vals)

DCzy_vals = DCzy(yvals,zvals)
RFzy_vals = Vrf_zy(yvals,zvals)
Uzy_vals = Uzy(yvals,zvals)


plt.plot(zvals,DCzy_vals,color='blue',label = 'DC',alpha = .3)
plt.plot(zvals,RFzy_vals,color='red',label = 'RF',alpha = .3)
plt.plot(zvals,Uzy_vals,color='green',label = 'DC + RF',alpha = .3)
plt.xlabel('z (mm)')
plt.ylabel('Potential (eV)')
plt.legend()
plt.show()


# @title Minimized trapping location in z

n_vals = 1000
yvals = np.ones(n_vals)*yCtr
zvals = np.linspace(zMin,zMax,n_vals)

Uzy_vals = Uzy(yvals,zvals) #this is interpolated from a data set

# Create the 1D interpolating function
Uzy_1D_inter_z = interpolate.interp1d(zvals, Uzy_vals, kind='cubic', fill_value='extrapolate')

#create functional input for the 1D interpolated function
def Uzy_1D_inter_fun_z(z):
  return Uzy_1D_inter_z(z)

init_guess_z = 1e-10
result_z = minimize(Uzy_1D_inter_fun_z, init_guess_z, method='Nelder-Mead')
print("Minimum value found (eV):", np.round(result_z.fun,4))
print("At the point (um):", np.round((result_z.x-zCtr)*1000,10))
# %% 2D potential plotting RF + DC


# @title 2D RF psuedopotential

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

nvals = 500
X1 = np.linspace(xMin,xMax,nvals)
Y1 = np.linspace(yMin,yMax,nvals)
X1,Y1 = np.meshgrid(X1,Y1)
Vrfxy = Vrf_xy(X1,Y1)

nvals = 500
Y2 = np.linspace(yMin,yMax,nvals)
Z2 = np.linspace(zMin,zMax,nvals)
Y2,Z2 = np.meshgrid(Y2,Z2)
Vrfzy = Vrf_zy(Y2,Z2)

Vrfxy = rotate(Vrfxy, 90,reshape=True)
Vrfzy = rotate(Vrfzy, 90,reshape=True)

cont1 =  axs[0].contourf(X1,Y1,Vrfxy, levels = 20,cmap = 'hsv')
cont2 =  axs[1].contourf(Z2,Y2,Vrfzy, levels = 20,cmap = 'hsv')


axs[0].set_xlabel('X (mm)')
axs[0].set_ylabel('Y (mm)')
axs[0].set_title('RF pseudopotential in XY plane')

axs[1].set_xlabel('Z (mm)')
axs[1].set_ylabel('Y (mm)')
axs[1].set_title('RF pseudopotential in YZ plane')

fig.colorbar(cont1, ax=axs[0])
fig.colorbar(cont2, ax=axs[1])

plt.show()


# @title 2D DC Potential
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

nvals = 500
X1 = np.linspace(xMin,xMax,nvals)
Y1 = np.linspace(yMin,yMax,nvals)
X1,Y1 = np.meshgrid(X1,Y1)
dcxy = DCxy(X1,Y1)

nvals = 500
Y2 = np.linspace(yMin,yMax,nvals)
Z2 = np.linspace(zMin,zMax,nvals)
Y2,Z2 = np.meshgrid(Y2,Z2)
dczy = DCzy(Y2,Z2)

dcxy = rotate(dcxy, 90,reshape=True)
dczy = rotate(dczy, 90,reshape=True)

cont1 =  axs[0].contourf(X1,Y1,dcxy, levels = 50,cmap = 'hsv')
cont2 =  axs[1].contourf(Y2,Z2,dczy, levels = 50,cmap = 'hsv')

axs[0].set_ylabel('Y (mm)')
axs[0].set_xlabel('X (mm)')
axs[0].set_title('DC potential in XY plane')

axs[1].set_xlabel('Y (mm)')
axs[1].set_ylabel('Z (mm)')
axs[1].set_title('DC potential in YZ plane')

fig.colorbar(cont1, ax=axs[0])
fig.colorbar(cont2, ax=axs[1])

plt.show()


# @title 2D Total Potential

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

nvals = 500
X1 = np.linspace(xMin,xMax,nvals)
Y1 = np.linspace(yMin,yMax,nvals)
X1,Y1 = np.meshgrid(X1,Y1)
uxy = Uxy(X1,Y1)

nvals = 500
Y2 = np.linspace(yMin,yMax,nvals)
Z2 = np.linspace(zMin,zMax,nvals)
Y2,Z2 = np.meshgrid(Y2,Z2)
uzy = Uzy(Y2,Z2)

uxy = rotate(uxy, 90,reshape=True)
uzy = rotate(uzy, 90,reshape=True)

cont1 =  axs[0].contourf(X1,Y1,uxy, levels = 50,cmap = 'hsv')
cont2 =  axs[1].contourf(Z2,Y2,uzy, levels = 50,cmap = 'hsv')

axs[0].set_ylabel('Y (mm)')
axs[0].set_xlabel('X (mm)')
axs[0].set_title('Total potential in XY plane')

axs[1].set_xlabel('Z (mm)')
axs[1].set_ylabel('Y (mm)')
axs[1].set_title('Total potential in YZ plane')

fig.colorbar(cont1, ax=axs[0])
fig.colorbar(cont2, ax=axs[1])

plt.show()


# %% Principal axes and secular frequencies- from a fit to 2d quadrupoles

# @title Uzy fits: $U_{zy}(y,z)=Ay^{2}+Byz+Cz^{2}+Dy+Ez+F$

def Uzy_fits(coords,A,B,C,D,E,F):
  y,z = coords
  fun = A*y**2 + B*y*z + C*z**2 + D*y + E*z + F
  return fun

num_points = 100
nvals_y = (yMax - yMin)/100
nvals_z = (zMax - zMin)/100

#create mesh
Y = np.arange(yMin,yMax,nvals_y)
Z = np.arange(zMin,zMax,nvals_z)
Y,Z = np.meshgrid(Y,Z)

uzy = Uzy(Y,Z)

uzy = rotate(uzy,90,reshape=True)

#unravel the data (curve_fit requires 1D data :/)
uzy_flat = uzy.ravel()
Y_flat = Y.ravel()
Z_flat = Z.ravel()
coords_zy = np.vstack((Y_flat,Z_flat)) #format data vertically (y,z)

fits_zy, other_stuff_zy = curve_fit(Uzy_fits,coords_zy, uzy_flat)


Uzy_fitted = Uzy_fits(coords_zy,*fits_zy)
Uzy_fitted = Uzy_fitted.reshape(Y.shape) #reshape according to Y data
# print(fits_zy)


# plt.contourf(Y,Z,uzy,levels = 20,cmap = 'hsv',alpha=0.3)
# plt.contour(Y,Z,uzy,levels = 20,colors = 'black',alpha=0.7)

# plt.contourf(Y,Z,Uzy_fitted,levels = 20,cmap = 'hsv',alpha = .3)
# plt.contour(Y,Z,Uzy_fitted,levels = 20,colors = 'blue',alpha = .7)

# plt.colorbar()
# plt.xlabel('Y (mm)')
# plt.ylabel('Z (mm)')
# plt.title('Total potential in YZ plane')
# plt.show()

# @title Minimum yz values
def Uzy_secular(coords):
  val = Uzy_fits(coords,*fits_zy)
  return val

init_guess_zy = [1e-6,1e-6]

result_zy = minimize(Uzy_secular, init_guess_zy, method='L-BFGS-B')
y_min = result_zy.x[0]
z_min = result_zy.x[1]

min_yz = [y_min,z_min]
# print("Minimum value found (eV):", np.round(result_zy.fun,4))
# print("At the point (um):", (np.round(y_min,10),np.round(z_min,10)))


# @title Uxy fits: $U_{xy}(x,y)=Ax^{2}+Bxy+Cy^{2}+Dx+Ey+F$
#define function for fitting
def Uxy_fits(coords,A,B,C,D,E,F):
  x,y = coords
  fun = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
  return fun

num_points = 1000
nvals_x = (xMax - xMin)/100
nvals_y = (yMax - yMin)/100

#create mesh
X = np.arange(xMin,xMax,nvals_x)
Y = np.arange(yMin,yMax,nvals_y)
X,Y = np.meshgrid(X,Y)

uxy = Uxy(X,Y) #functional data
uxy = rotate(uxy,90,reshape=True)

#unravel data
X_flat = X.ravel()
Y_flat = Y.ravel()
uxy_flat = uxy.ravel()
coords_xy = np.vstack((X_flat,Y_flat)) #format data vertically (x,y)

guesses = [0,0,0,0,0,0]

fits_xy, other_stuff_xy = curve_fit(Uxy_fits,coords_xy, uxy_flat)

Uxy_fitted = Uxy_fits(coords_xy,*fits_xy)
Uxy_fitted = Uxy_fitted.reshape(X.shape) #reshape according to X data
# print(fits_xy)

# plt.contourf(X,Y,uxy,levels = 20,cmap = 'hsv',alpha = 0.3)
# plt.contour(X,Y,uxy,levels = 20,colors='black',alpha=.7)

# plt.contourf(X,Y,Uxy_fitted,levels=20,cmap = 'hsv',alpha = .3)
# plt.contour(X,Y,Uxy_fitted,levels=20,colors='blue',alpha = .7)
# plt.colorbar()
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')
# plt.title('Total potential in xy plane')
# plt.show()

# @title Minimum xy values
def Uxy_secular(coords):
  val = Uxy_fits(coords,*fits_xy)
  return val

init_guess_xy = [1e-6,1e-6]

result_xy = minimize(Uxy_secular, init_guess_xy, method='L-BFGS-B')

x_min = result_xy.x[0]
y_min = result_xy.x[1]

min_xy = [x_min,y_min]
# print("Minimum value found (eV):", np.round(result_xy.fun,4))
# print("At the point (um):", (np.round(x_min,10),np.round(y_min,10)))


# For a function of the form:
# 
# $f(x,y)=ax^{2}+by^{2}+c*xy+d*x+e*y+f$, \
# the rotation angle will be
# 
# $\theta = \frac{1}{2}Arctan(\frac{c}{b-a})$

# @title Compute angle of ellipse relative to original axes
def theta(fits):
  a,b,c,d,e,f = fits
  theta = 0.5*np.arctan((b)/(a-c))
  return theta


theta_x = np.round(theta(fits_xy),10)
theta_y = -(np.round(theta(fits_zy),5))
theta_z = theta_y + np.pi /2

# print(theta_x,theta_y,theta_z)

# @title Compute axes of ellipse
def cos(theta):
  return np.cos(theta)

def sin(theta):
  return np.sin(theta)

def y_axes(y_min,z_min,amp,theta_y,theta_z):

  y_prime = y_min + amp*cos(theta_y)
  z_prime = z_min + amp*sin(theta_y)
  coords = np.array([y_prime,z_prime])
  return coords.T


def z_axes(y_min,z_min,amp,theta_y,theta_z):

  y_prime = y_min + amp*cos(theta_z)
  z_prime = z_min + amp*sin(theta_z)
  coords = np.array([y_prime,z_prime])
  return coords.T

amp_y = np.arange(yMin,yMax,.001)
amp_z = np.arange(zMin,zMax,.001)

prin_z_axes = z_axes(y_min,z_min,amp_z,theta_y,theta_z)
prin_y_axes = y_axes(y_min,z_min,amp_y,theta_y,theta_z)


fig, axs = plt.subplots(1, 1)

nvals = 500
Y2 = np.linspace(yMin,yMax,nvals)
Z2 = np.linspace(zMin,zMax,nvals)
Y2,Z2 = np.meshgrid(Y2,Z2)
uzy = Uzy(Y2,Z2)

cont2 =  axs.contourf(Z2,Y2,uzy, levels = 50,cmap = 'hsv')

axs.set_xlabel('Z (mm)')
axs.set_ylabel('Y (mm)')
axs.set_title('Total potential in YZ plane')

fig.colorbar(cont2, ax=axs)

plt.plot(prin_z_axes.T[0], prin_z_axes.T[1], label='prin z-axis')
plt.plot(prin_y_axes.T[0], prin_y_axes.T[1], label='prin y-axis')
plt.legend()
plt.show()


print('theta_y (degree): ',theta_y / np.pi * 180)
print('theta_z (degree): ',theta_z / np.pi * 180)


wy = (1/2/np.pi)*np.sqrt(1e6*scipy.e*2*(fits_zy[0]*cos(theta_y)**2+fits_zy[1]*sin(theta_y)*cos(theta_y)+fits_zy[2]*sin(theta_y)**2)/m)
wz = (1/2/np.pi)*np.sqrt(1e6*scipy.e*2*(fits_zy[0]*sin(theta_y)**2-fits_zy[1]*sin(theta_y)*cos(theta_y)+fits_zy[2]*cos(theta_y)**2)/m)
wx = (1/2/np.pi)*np.sqrt(1e6*scipy.e*2*(fits_xy[0]*cos(theta_x)**2-fits_xy[1]*sin(theta_x)*cos(theta_x)+fits_xy[2]*sin(theta_x)**2)/m)

print('pseudoharmonic apporximation Wx, Wy, Wz: (MHz)', wx/1e6, wy/1e6, wz/1e6)
# %% 1D Axial Radial potential polynomial fit using numpy polyfit

xvals = np.linspace(xMin, xMax, 1000)
yvals = np.linspace(yMin, yMax, 1000)
zvals = np.linspace(zMin, zMax, 1000)

Ux_vals = Uxy(xvals,np.zeros(len(xvals)))  # Replace with your actual function call to get Uxy values
Uy_vals = Uxy(np.zeros(len(xvals)),yvals)  # Replace with your actual function call to get Uxy values
Uz_vals = Uzy(np.zeros(len(xvals)),zvals)  # Replace with your actual function call to get Uxy values


# Fit polynomial of degree 2 (quadratic)
x_coeffs = np.polyfit(xvals, Ux_vals, 6)  # The second argument here is the degree of the polynomial
y_coeffs = np.polyfit(yvals, Uy_vals, 2)  # The second argument here is the degree of the polynomial
z_coeffs = np.polyfit(zvals, Uz_vals, 2)  # The second argument here is the degree of the polynomial


# Create a polynomial function from the fitted coefficients
x_poly = np.poly1d(x_coeffs)
y_poly = np.poly1d(y_coeffs)
z_poly = np.poly1d(z_coeffs)


plt.plot(xvals, Ux_vals, label="Original Data")
plt.plot(xvals, x_poly(xvals), label="Polynomial Fit", linestyle='--')
plt.xlabel('x')
plt.ylabel('Vx (Potential)')
plt.legend()
plt.show()

plt.plot(yvals, Uy_vals, label="Original Data")
plt.plot(yvals, y_poly(yvals), label="Polynomial Fit", linestyle='--')
plt.xlabel('y')
plt.ylabel('Vy (Potential)')
plt.legend()
plt.show()

plt.plot(zvals, Uz_vals, label="Original Data")
plt.plot(zvals, z_poly(zvals), label="Polynomial Fit", linestyle='--')
plt.xlabel('z')
plt.ylabel('Vz (Potential)')
plt.legend()
plt.show()

# %% Reconstructed 1D potential with polynomial coefficients

# x_coeffs_rev = x_coeffs[::-1]

x = np.linspace(-0.25, 0.25, 100)
Vx = sum([x_coeffs_rev[j] * (x**j) for j in range(len(x_coeffs_rev))])
Vx = sum([x_coeffs[j] * (x**(6-j)) for j in range(len(x_coeffs))])
Vx = np.polyval(x_coeffs, x)

plt.plot(x, Vx)
plt.xlabel('x')
plt.ylabel('Potential V')
plt.show()


# y_coeffs_rev = y_coeffs[::-1]

y = np.linspace(-0.03, 0.03, 100)
Vy = sum([y_coeffs_rev[j] * (y**j) for j in range(len(y_coeffs_rev))])
Vy = sum([y_coeffs[j] * (y**(6-j)) for j in range(len(y_coeffs))])
Vy = np.polyval(y_coeffs, y)

plt.plot(y, Vy)
plt.xlabel('y')
plt.ylabel('Potential V')
plt.show()


# z_coeffs_rev = z_coeffs[::-1]

z = np.linspace(-0.03, 0.03, 100)
Vz = sum([z_coeffs_rev[j] * (z**j) for j in range(len(z_coeffs_rev))])
Vz = sum([z_coeffs[j] * (z**(6-j)) for j in range(len(z_coeffs))])
Vz = np.polyval(z_coeffs, z)

plt.plot(z, Vz)
plt.xlabel('z')
plt.ylabel('Potential V')
plt.show()



# %% # of ions / Electric potential function
x_coeffs_rev = x_coeffs[::-1]
y_coeffs_rev = y_coeffs[::-1]
z_coeffs_rev = z_coeffs[::-1]

#definitons
e = scipy.elementary_charge
eps_o = scipy.epsilon_0
m = 171*1.67262192e-27
delta_k = np.sqrt(2)*2*np.pi/(355e-9)
hbar = scipy.hbar

# lc = (e**2/(4*np.pi*eps_o*m*wz**2))**(1/3)
recoil = (hbar*(delta_k)**2 ) / (2*m)

# number of ions
N = 10

@jit(nopython=True, fastmath=True)
def potential_energy(positions):
    # wx, wy, wz, N, m, e, epsilon_0 = params
    xs = positions[:N]
    ys = positions[N:2*N]
    zs = positions[2*N:3*N]
    
    # Initialize potential energy terms
    Vx = np.zeros_like(xs)
    Vy = np.zeros_like(ys)
    Vz = np.zeros_like(zs)
    
    # Calculate Vx, Vy, and Vz using a manual loop for summing
  
    for j in range(len(x_coeffs_rev)):
        Vx += x_coeffs_rev[j] * ((1000*xs)**j)
        
    for j in range(len(y_coeffs_rev)):
        Vy += y_coeffs_rev[j] * ((1000*ys)**j)
        
    for j in range(len(z_coeffs_rev)):
        Vz += z_coeffs_rev[j] * ((1000*zs)**j) 
    
    # Harmonic energy
    harm_energy = e * np.sum(Vx + Vy + Vz) 
    
    # electronic interaction
    interaction = 0
    for i in range(N):
        for j in range(N):
            if j != i:
                interaction += 1/np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)
    interaction = e**2/8/np.pi/eps_o * interaction
    
    # print(harm_energy+interaction)
    return harm_energy + interaction


xs_0 = np.linspace(-1,1,N) * 1e-5
ys_0 = np.linspace(-1,1,N) * 0
zs_0 = np.linspace(-1,1,N) * 0

pos = np.append(np.append(xs_0, ys_0), zs_0)

pot_en = potential_energy(pos)
print('Potential Energy', pot_en)

plt.plot(xs_0 * 1e6, ys_0 * 1e6, '.', color='royalblue', markersize=12, markeredgecolor='darkblue')
plt.xlabel('x distance (micron)')
plt.ylabel('y distance (micron)')
plt.title('Initial ion position guess')
plt.grid()
plt.show()

# %% Minimize potential energy to determine ion crystal positions
bounds = [(xMin*1e-3, xMax*1e-3)] * N + [(yMin*1e-3, yMax*1e-3)] * N + [(zMin*1e-3, zMax*1e-3)] * N  # bounds for N ions in x, y, z

# run bad guess through optimization with on a few iterations
pos_better = minimize(potential_energy, pos, method='COBYLA', bounds=bounds, options={'tol':1e-30, 'maxiter':2000})

# Get fine results with better initial best and many iterations
res = minimize(potential_energy, pos_better.x, method='COBYLA', bounds=bounds, options={'tol':1e-30, 'maxiter':80000})

# %% Plotting equilibrium positions and uniformity
# Plot ion positions

xs_f = res.x[:N]
ys_f = res.x[N:2*N]
zs_f = res.x[2*N:3*N]


sorted_indices = np.argsort(xs_f)
zs_f = zs_f[sorted_indices]
xs_f = xs_f[sorted_indices]
ys_f = ys_f[sorted_indices]

fig, axes = plt.subplots(1,3,figsize=(15,5))

axes[0].plot(xs_f * 1e6, ys_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
axes[0].set_title('Ion equilibrium position')
axes[0].set_xlabel('x distance (um)')
axes[0].set_ylabel('y distance (um)')
axes[0].set_ylim(-100, 100)
axes[0].set_xlim(-100, 100)

# axes[0].set_ylim(-5, 5)
# axes[0].set_xlim(-10, 10)
axes[0].grid()

# Add index labels for the first plot
# for i in range(len(xs_f)):
#     axes[0].text(xs_f[i] * 1e6+10, ys_f[i] * 1e6+10, f'{i}', color='black', fontsize=10, ha='left', va='top')

axes[1].plot(zs_f * 1e6, ys_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
axes[1].set_title('Ion equilibrium position')
axes[1].set_xlabel('z distance (um)')
axes[1].set_ylabel('y distance (um)')
axes[1].set_ylim(-100, 100)
axes[1].set_xlim(-100, 100)

# axes[1].set_ylim(-5, 5)
# axes[1].set_xlim(-10, 10)

axes[1].grid()

# Add index labels for the second plot
# for i in range(len(zs_f)):
#     axes[1].text(zs_f[i] * 1e6+10, ys_f[i] * 1e6+10, f'{i}', color='black', fontsize=10, ha='left', va='top')
#     axes[1].text(zs_f[i], ys_f[i], f'{i}', color='black', fontsize=10, ha='left', va='top')


axes[2].plot(xs_f * 1e6, zs_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
axes[2].set_title('Ion equilibrium position')
axes[2].set_xlabel('x distance (mm)')
axes[2].set_ylabel('z distance (mm)')
axes[2].set_ylim(-100, 100)
axes[2].set_xlim(-100, 100)

# axes[2].set_ylim(-20, 20)
# axes[2].set_xlim(-20, 20)

axes[2].grid()

# Add index labels for the third plot
# for i in range(len(xs_f)):
#     axes[2].text(xs_f[i] * 1e6+10, zs_f[i] * 1e6+10, f'{i}', color='black', fontsize=10, ha='left', va='top')

plt.tight_layout()
plt.show()


print('Uniformity: ', np.round(100*(1-np.std(np.diff(xs_f))/np.mean(np.diff(xs_f))),3),' %')
# %% Axial Frequencies (x axis)
const1 = e**2/8/np.pi/eps_o*(2/m)
const2 = e*(2/m)

def axial_eigensystem(x, y, z, coeffs, N):
  """
  Calculates the mode spectrum for a given set of ions at specific trap frequencies.

  Parameters:
  frad (float): Radial Trap frequency in Hz.
  fax (float): Axial trap frequency in Hz.
  num_ions (int): The number of ions in the system.
  lc (float): The lattice constant in meters.

  Returns:
  eigenvalues (array): Normal mode frequencies in MHz. Need to be
  multiplied by 10^6 to get in Hz.

  eigenvectors (array): Array of eigenvectors.
  """

  Anm = np.empty((N,N))

  for n in range(N):
    for m in range(N):
      if n == m:
        sum1 = 0.0
        for p in range(N):
          if(p!=m):
            sum1 += (2*(x[m] - x[p])**2 - (y[m] - y[p])**2 - (z[m] - z[p])**2) / abs((x[m] - x[p])**2+(y[m] - y[p])**2+(z[m] - z[p])**2)**(5/2)
        sum11 = 0.0
        for i in range(len(coeffs)):
            if i >= 2:
                sum11 += coeffs[i]*i*(i-1)*(x[n]**(i-2))*((1e3)**i)
        Anm[n][n]= (const2*sum11 + const1*sum1)
      elif n!=m:
        sum2 = (2*(x[m] - x[n])**2 - (y[m] - y[n])**2 - (z[m] - z[n])**2) /abs((x[m] - x[n])**2+(y[m] - y[p])**2+(z[m] - z[n])**2)**(5/2)
        Anm[n][m]= -const1*sum2
        
  # print(Anm)

  eigenvalues, eigenvectors = np.linalg.eig(Anm)
  eigenvalues = np.sqrt(eigenvalues)
  eigenvectors = eigenvectors

  return eigenvalues, eigenvectors
 

w_axial, axial_normal_modes = axial_eigensystem(xs_f, ys_f, zs_f, x_coeffs_rev, N)

f_axial = w_axial / 2 / np.pi # Hz

print('Axial frequencies (kHz): ', f_axial/1e3)
print('Axial COM frequency (kHz): ', np.min(f_axial)/1e3)


fig,axes = plt.subplots(1,N,figsize=(3*N,N))

for i in range(N):
    axes[i].plot(xs_f*1e6, axial_normal_modes.T[i],'o', markersize=5, color='royalblue', markeredgecolor='k')
    for j in range(len(xs_f)):
        axes[i].text(xs_f[j] * 1e6+10, axial_normal_modes.T[i][j], f'{j}', color='black', fontsize=15, ha='left', va='top')
    axes[i].set_title(f'axial normal mode {i}')
    axes[i].set_xlabel('x distance (micron)')
    axes[i].set_ylabel('normal mode amplitude')
    axes[i].set_ylim(-1,1)
    axes[i].grid()

plt.tight_layout()
plt.show()

# %% Transverse Frequencies (y axis)

const1 = e**2/8/np.pi/eps_o*(2/m)
const2 = e*(2/m)

def transverse1_eigensystem(x, y, z, coeffs, N):
  """
  Calculates the mode spectrum for a given set of ions at specific trap frequencies.

  Parameters:
  frad (float): Radial Trap frequency in Hz.
  fax (float): Axial trap frequency in Hz.
  num_ions (int): The number of ions in the system.
  lc (float): The lattice constant in meters.

  Returns:
  eigenvalues (array): Normal mode frequencies in MHz. Need to be
  multiplied by 10^6 to get in Hz.

  eigenvectors (array): Array of eigenvectors.
  """

  Anm = np.empty((N,N))

  for n in range(N):
    for m in range(N):
      if n == m:
        sum1 = 0.0
        for p in range(N):
          if(p!=m):
            sum1 += (2*(y[m] - y[p])**2 - (z[m] - z[p])**2 - (x[m] - x[p])**2) / abs((x[m] - x[p])**2+(y[m] - y[p])**2+(z[m] - z[p])**2)**(5/2)
        sum11 = 0.0
        for i in range(len(coeffs)):
            if i >= 2:
                sum11 += coeffs[i]*i*(i-1)*(y[n]**(i-2))*((1e3)**i)
        Anm[n][n]= (const2*sum11 + const1*sum1)

      elif n!=m:
        sum2 = (2*(y[m] - y[n])**2 - (z[m] - z[n])**2 - (x[m] - x[n])**2) /abs((x[m] - x[n])**2+(y[m] - y[p])**2+(z[m] - z[n])**2)**(5/2)
        Anm[n][m]= -const1*sum2
        
  
  # Anm = np.abs(Anm)
  # print(Anm)

  eigenvalues, eigenvectors = np.linalg.eig(Anm)
  eigenvalues = np.sqrt(eigenvalues)
  eigenvectors = eigenvectors

  return eigenvalues, eigenvectors
 
w_trans1, transverse1_normal_modes = transverse1_eigensystem(xs_f, ys_f, zs_f, y_coeffs_rev, N)

f_trans1 = w_trans1 / 2 / np.pi # Hz

print('Transverse1 frequencies(MHz): ', f_trans1/1e6)
print('Transverse1 COM frequency (MHz): ', np.max(f_trans1)/1e6)

fig,axes = plt.subplots(1,N,figsize=(3*N,N))

for i in range(N):
    axes[i].plot(xs_f*1e6, transverse1_normal_modes.T[i],'o', markersize=5, color='royalblue', markeredgecolor='k')
    for j in range(len(xs_f)):
        axes[i].text(xs_f[j] * 1e6+10, transverse1_normal_modes.T[i][j], f'{j}', color='black', fontsize=15, ha='left', va='top')
    axes[i].set_title(f'transverse1 normal mode {i}')
    axes[i].set_xlabel('x distance (micron)')
    axes[i].set_ylabel('normal mode amplitude')
    axes[i].set_ylim(-1,1)
    axes[i].grid()

plt.tight_layout()
plt.show()

# %% Transverse Frequencies (z axis)


def theta(V1, V2, V3, Vrf, N):
    #constants
    q = scipy.elementary_charge
    eps_o = scipy.epsilon_0
    M_Yb = 171*1.67262192e-27
    m = 171*1.67262192e-27
    d =  1000 #mm/m distance from
    frf = 38.6e6
    wrf = 2*np.pi*frf
    
    
    #definitons
    e = scipy.elementary_charge
    m = 171*1.67262192e-27
    delta_k = np.sqrt(2)*2*np.pi/(355e-9)
    hbar = scipy.hbar

    # lc = (e**2/(4*np.pi*eps_o*m*wz**2))**(1/3)
    recoil = (hbar*(delta_k)**2 ) / (2*m)

    DC1 = V1
    DC2 = V2
    DC3 = V3

    #coordiante ranges
    xCtr = 0.
    xRange = .5
    yCtr = 0.
    yRange = 0.06
    zCtr = 0.
    zRange = 0.06;
    xMin = xCtr - xRange/2
    xMax = xCtr + xRange/2
    yMin = yCtr - yRange/2
    yMax = yCtr + yRange/2
    zMin = zCtr - zRange/2
    zMax = zCtr + zRange/2

    # Voltage functions

    def V_calc(q1,q2,data_in):
      #transpose and unpack data
      data_in = data_in.T
      x1 = data_in[0]
      x2 = data_in[1]
      f = data_in[2]
      #prepare points
      qs = np.array([q1,q2]).T
      points = np.array([x1,x2]).T
      #interpolate
      vals = interpolate.griddata(points, f, qs, \
                                method='cubic',fill_value=np.NaN)

      return vals


    #DC1 endcaps
    def DC1xyf(x,y):
      val = V_calc(x,y,DC1xy_data)
      return val

    def DC1zyf(y,z):
      val = V_calc(y,z,DC1zy_data)
      return val

    def DC1zxf(x,z):
      val = V_calc(x,z,DC1zx_data)
      return val

    #DC2 middle centers
    def DC2xyf(x,y):
      val = V_calc(x,y,DC2xy_data)
      return val

    def DC2zyf(y,z):
      val = V_calc(y,z,DC2zy_data)
      return val

    def DC2zxf(x,z):
      val = V_calc(x,z,DC2zx_data)
      return val

    #DC3 center electrode
    def DC3xyf(x,y):
      val = V_calc(x,y,DC3xy_data)
      return val

    def DC3zyf(y,z):
      val = V_calc(y,z,DC3zy_data)
      return val

    def DC3zxf(x,z):
      val = V_calc(x,z,DC3zx_data)
      return val

    #RF
    def RFxyf(x,y):
      val = V_calc(x,y,RFxy_data)
      return val

    def RFzyf(y,z):
      val = V_calc(y,z,RFzy_data)
      return val

    def RFzxf(x,z):
      val = V_calc(x,z,RFzx_data)
      return val

    # @title Electrostatic functions

    #dc potentials
    def DCxy(x,y):
      val = DC1*DC1xyf(x,y)+DC2*DC2xyf(x,y)+DC3*DC3xyf(x,y)
      return val

    def DCzx(x,z):
      val = DC1*DC1zxf(x,z)+DC2*DC2zxf(x,z)+DC3*DC3zxf(x,z)
      return val

    def DCzy(y,z):
      val = DC1*DC1zyf(y,z)+DC2*DC2zyf(y,z)+DC3*DC3zyf(y,z)
      return val

    #transform the imported RF data to the rf psuedopotential...
    def Vrf_xy(x,y):
      val = q * (RFxyf(x,y)**2) * v_rf**2 / (4 * M_Yb * (wrf**2))
      return val

    def Vrf_zx(x,z):
      val = q * (RFzxf(x,z)**2) * v_rf**2 / (4 * M_Yb * (wrf**2))
      return val

    def Vrf_zy(y,z):
      val = q * (RFzyf(y,z)**2) * v_rf**2 / (4 * M_Yb * (wrf**2))
      return val

    #total potentials
    def Uxy(x,y):
      val = DCxy(x,y) + Vrf_xy(x,y)
      return val

    def Uzx(x,z):
      val = DCzx(x,z) + Vrf_zx(x,z)
      return val

    def Uzy(y,z):
      val = DCzy(y,z) + Vrf_zy(y,z)
      return val


    # **1D Plots**

    # 1D Axial potential plotting 

    #Axial Trapping along X-axis

    n_vals = 1000
    xvals = np.linspace(xMin,xMax,n_vals)
    yvals = np.ones(n_vals)*yCtr
    zvals = np.ones(n_vals)*zCtr

    Uxy_vals = Uxy(xvals,yvals) #this is interpolated from a data set
    Uzx_vals = Uxy(xvals,zvals) #this is interpolated from a data set

    # @title Minimum Position in x
    n_vals = 1000
    xvals = np.linspace(xMin,xMax,n_vals)
    yvals = np.ones(n_vals)*yCtr

    Uxy_vals = Uxy(xvals,yvals) #this is interpolated from a data set

    # Create the 1D interpolating function
    Uxy_1D_inter_x = interpolate.interp1d(xvals, Uxy_vals, kind='cubic', fill_value='extrapolate')

    #create functional input for the 1D interpolated function
    def Uxy_1D_inter_fun_x(x):
      return Uxy_1D_inter_x(x)

    init_guess_x = 1e-10
    result_x = minimize(Uxy_1D_inter_fun_x, init_guess_x, method='Nelder-Mead')
   
    # @title Axial trapping along x-axis close to center
    n_vals = 1000
    xvals = np.linspace(xMin,xMax,n_vals)
    yvals = np.ones(n_vals)*(yCtr+yRange/10)
    zvals = np.ones(n_vals)*zCtr

    Uxy_vals = Uxy(xvals,yvals)
    Uzx_vals = Uxy(xvals,zvals)

    # 1D Radial potential plotting

    #create functional input for the 1D interpolated function
    def Uxy_1D_inter_fun_y(y):
      return Uxy_1D_inter_y(y)

    init_guess_y = 1e-10
    result_y = minimize(Uxy_1D_inter_fun_y, init_guess_y, method='Nelder-Mead')
  
    # @title Radial Trapping along z-axis
    n_vals = 1000
    xvals = np.ones(n_vals)*xCtr
    yvals = np.ones(n_vals)*yCtr
    zvals = np.linspace(zMin,zMax,n_vals)

    DCzy_vals = DCzy(yvals,zvals)
    RFzy_vals = Vrf_zy(yvals,zvals)
    Uzy_vals = Uzy(yvals,zvals)

    # @title Minimized trapping location in z

    n_vals = 1000
    yvals = np.ones(n_vals)*yCtr
    zvals = np.linspace(zMin,zMax,n_vals)

    Uzy_vals = Uzy(yvals,zvals) #this is interpolated from a data set

    # Create the 1D interpolating function
    Uzy_1D_inter_z = interpolate.interp1d(zvals, Uzy_vals, kind='cubic', fill_value='extrapolate')

    #create functional input for the 1D interpolated function
    def Uzy_1D_inter_fun_z(z):
      return Uzy_1D_inter_z(z)

    init_guess_z = 1e-10
    result_z = minimize(Uzy_1D_inter_fun_z, init_guess_z, method='Nelder-Mead')
 
    # Principal axes and secular frequencies- from a fit to 2d quadrupoles

    # @title Uzy fits: $U_{zy}(y,z)=Ay^{2}+Byz+Cz^{2}+Dy+Ez+F$

    def Uzy_fits(coords,A,B,C,D,E,F):
      y,z = coords
      fun = A*y**2 + B*y*z + C*z**2 + D*y + E*z + F
      return fun

    num_points = 100
    nvals_y = (yMax - yMin)/100
    nvals_z = (zMax - zMin)/100

    #create mesh
    Y = np.arange(yMin,yMax,nvals_y)
    Z = np.arange(zMin,zMax,nvals_z)
    Y,Z = np.meshgrid(Y,Z)

    uzy = Uzy(Y,Z)

    uzy = rotate(uzy,90,reshape=True)

    #unravel the data (curve_fit requires 1D data :/)
    uzy_flat = uzy.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    coords_zy = np.vstack((Y_flat,Z_flat)) #format data vertically (y,z)

    fits_zy, other_stuff_zy = curve_fit(Uzy_fits,coords_zy, uzy_flat)


    Uzy_fitted = Uzy_fits(coords_zy,*fits_zy)
    Uzy_fitted = Uzy_fitted.reshape(Y.shape) #reshape according to Y data
   
  
    # @title Uxy fits: $U_{xy}(x,y)=Ax^{2}+Bxy+Cy^{2}+Dx+Ey+F$
    #define function for fitting
    def Uxy_fits(coords,A,B,C,D,E,F):
      x,y = coords
      fun = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
      return fun

    num_points = 1000
    nvals_x = (xMax - xMin)/100
    nvals_y = (yMax - yMin)/100

    #create mesh
    X = np.arange(xMin,xMax,nvals_x)
    Y = np.arange(yMin,yMax,nvals_y)
    X,Y = np.meshgrid(X,Y)

    uxy = Uxy(X,Y) #functional data
    uxy = rotate(uxy,90,reshape=True)

    #unravel data
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    uxy_flat = uxy.ravel()
    coords_xy = np.vstack((X_flat,Y_flat)) #format data vertically (x,y)

    guesses = [0,0,0,0,0,0]

    fits_xy, other_stuff_xy = curve_fit(Uxy_fits,coords_xy, uxy_flat)

    Uxy_fitted = Uxy_fits(coords_xy,*fits_xy)
    Uxy_fitted = Uxy_fitted.reshape(X.shape) #reshape according to X data
 
 
    # @title Compute angle of ellipse relative to original axes
    def theta(fits):
      a,b,c,d,e,f = fits
      theta = 0.5*np.arctan((b)/(a-c))
      return theta


    theta_x = np.round(theta(fits_xy),10)
    theta_y = -(np.round(theta(fits_zy),5))
    theta_z = theta_y + np.pi /2
    
    print('theta_y: ',theta_y)

    return theta_y


#%% Uniformity and theta axial com frequency function
def U(V1, V2, V3, Vrf):
    #constants
    q = scipy.elementary_charge
    eps_o = scipy.epsilon_0
    M_Yb = 171*1.67262192e-27
    m = 171*1.67262192e-27
    d =  1000 #mm/m distance from
    frf = 38.6e6
    wrf = 2*np.pi*frf
    
    
    #definitons
    e = scipy.elementary_charge
    m = 171*1.67262192e-27
    delta_k = np.sqrt(2)*2*np.pi/(355e-9)
    hbar = scipy.hbar

    lc = (e**2/(4*np.pi*eps_o*m*wz**2))**(1/3)
    recoil = (hbar*(delta_k)**2 ) / (2*m)
    
    
    DC1 = V1
    DC2 = V2
    DC3 = V3

    #coordiante ranges
    xCtr = 0.
    xRange = .5
    yCtr = 0.
    yRange = 0.06
    zCtr = 0.
    zRange = 0.06;
    xMin = xCtr - xRange/2
    xMax = xCtr + xRange/2
    yMin = yCtr - yRange/2
    yMax = yCtr + yRange/2
    zMin = zCtr - zRange/2
    zMax = zCtr + zRange/2

    # Voltage functions

    def V_calc(q1,q2,data_in):
      #transpose and unpack data
      data_in = data_in.T
      x1 = data_in[0]
      x2 = data_in[1]
      f = data_in[2]
      #prepare points
      qs = np.array([q1,q2]).T
      points = np.array([x1,x2]).T
      #interpolate
      vals = interpolate.griddata(points, f, qs, \
                                method='cubic',fill_value=np.NaN)

      return vals


    #DC1 endcaps
    def DC1xyf(x,y):
      val = V_calc(x,y,DC1xy_data)
      return val

    def DC1zyf(y,z):
      val = V_calc(y,z,DC1zy_data)
      return val

    def DC1zxf(x,z):
      val = V_calc(x,z,DC1zx_data)
      return val

    #DC2 middle centers
    def DC2xyf(x,y):
      val = V_calc(x,y,DC2xy_data)
      return val

    def DC2zyf(y,z):
      val = V_calc(y,z,DC2zy_data)
      return val

    def DC2zxf(x,z):
      val = V_calc(x,z,DC2zx_data)
      return val

    #DC3 center electrode
    def DC3xyf(x,y):
      val = V_calc(x,y,DC3xy_data)
      return val

    def DC3zyf(y,z):
      val = V_calc(y,z,DC3zy_data)
      return val

    def DC3zxf(x,z):
      val = V_calc(x,z,DC3zx_data)
      return val

    #RF
    def RFxyf(x,y):
      val = V_calc(x,y,RFxy_data)
      return val

    def RFzyf(y,z):
      val = V_calc(y,z,RFzy_data)
      return val

    def RFzxf(x,z):
      val = V_calc(x,z,RFzx_data)
      return val

    # @title Electrostatic functions

    #dc potentials
    def DCxy(x,y):
      val = DC1*DC1xyf(x,y)+DC2*DC2xyf(x,y)+DC3*DC3xyf(x,y)
      return val

    def DCzx(x,z):
      val = DC1*DC1zxf(x,z)+DC2*DC2zxf(x,z)+DC3*DC3zxf(x,z)
      return val

    def DCzy(y,z):
      val = DC1*DC1zyf(y,z)+DC2*DC2zyf(y,z)+DC3*DC3zyf(y,z)
      return val

    #transform the imported RF data to the rf psuedopotential...
    def Vrf_xy(x,y):
      val = q * (RFxyf(x,y)**2) * Vrf**2 / (4 * M_Yb * (wrf**2))
      return val

    def Vrf_zx(x,z):
      val = q * (RFzxf(x,z)**2) * Vrf**2 / (4 * M_Yb * (wrf**2))
      return val

    def Vrf_zy(y,z):
      val = q * (RFzyf(y,z)**2) * Vrf**2 / (4 * M_Yb * (wrf**2))
      return val

    #total potentials
    def Uxy(x,y):
      val = DCxy(x,y) + Vrf_xy(x,y)
      return val

    def Uzx(x,z):
      val = DCzx(x,z) + Vrf_zx(x,z)
      return val

    def Uzy(y,z):
      val = DCzy(y,z) + Vrf_zy(y,z)
      return val


    def Uzy_fits(coords,A,B,C,D,E,F):
      y,z = coords
      fun = A*y**2 + B*y*z + C*z**2 + D*y + E*z + F
      return fun

    num_points = 100
    nvals_y = (yMax - yMin)/100
    nvals_z = (zMax - zMin)/100

    #create mesh
    Y = np.arange(yMin,yMax,nvals_y)
    Z = np.arange(zMin,zMax,nvals_z)
    Y,Z = np.meshgrid(Y,Z)

    uzy = Uzy(Y,Z)

    uzy = rotate(uzy,90,reshape=True)

    #unravel the data (curve_fit requires 1D data :/)
    uzy_flat = uzy.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    coords_zy = np.vstack((Y_flat,Z_flat)) #format data vertically (y,z)

    fits_zy, other_stuff_zy = curve_fit(Uzy_fits,coords_zy, uzy_flat)


    Uzy_fitted = Uzy_fits(coords_zy,*fits_zy)
    Uzy_fitted = Uzy_fitted.reshape(Y.shape) #reshape according to Y data
 

    # @title Minimum yz values
    def Uzy_secular(coords):
      val = Uzy_fits(coords,*fits_zy)
      return val

    init_guess_zy = [1e-6,1e-6]

    result_zy = minimize(Uzy_secular, init_guess_zy, method='L-BFGS-B')
    y_min = result_zy.x[0]
    z_min = result_zy.x[1]

    min_yz = [y_min,z_min]

    # @title Uxy fits: $U_{xy}(x,y)=Ax^{2}+Bxy+Cy^{2}+Dx+Ey+F$
    #define function for fitting
    def Uxy_fits(coords,A,B,C,D,E,F):
      x,y = coords
      fun = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
      return fun

    num_points = 1000
    nvals_x = (xMax - xMin)/100
    nvals_y = (yMax - yMin)/100

    #create mesh
    X = np.arange(xMin,xMax,nvals_x)
    Y = np.arange(yMin,yMax,nvals_y)
    X,Y = np.meshgrid(X,Y)

    uxy = Uxy(X,Y) #functional data
    uxy = rotate(uxy,90,reshape=True)

    #unravel data
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    uxy_flat = uxy.ravel()
    coords_xy = np.vstack((X_flat,Y_flat)) #format data vertically (x,y)

    guesses = [0,0,0,0,0,0]

    fits_xy, other_stuff_xy = curve_fit(Uxy_fits,coords_xy, uxy_flat)

    Uxy_fitted = Uxy_fits(coords_xy,*fits_xy)
    Uxy_fitted = Uxy_fitted.reshape(X.shape) #reshape according to X data

    # @title Minimum xy values
    def Uxy_secular(coords):
      val = Uxy_fits(coords,*fits_xy)
      return val

    init_guess_xy = [1e-6,1e-6]

    result_xy = minimize(Uxy_secular, init_guess_xy, method='L-BFGS-B')

    x_min = result_xy.x[0]
    y_min = result_xy.x[1]

    min_xy = [x_min,y_min]


    # @title Compute angle of ellipse relative to original axes
    def theta(fits):
      a,b,c,d,e,f = fits
      theta = 0.5*np.arctan((b)/(a-c))
      return theta


    theta_x = np.round(theta(fits_xy),10)
    theta_y = -(np.round(theta(fits_zy),5))
    theta_z = theta_y + np.pi /2

    # 1D Axial Radial potential polynomial fit using numpy polyfit

    xvals = np.linspace(xMin, xMax, 1000)
    yvals = np.linspace(yMin, yMax, 1000)
    zvals = np.linspace(zMin, zMax, 1000)

    Ux_vals = Uxy(xvals,np.zeros(len(xvals)))  # Replace with your actual function call to get Uxy values
    Uy_vals = Uxy(np.zeros(len(xvals)),yvals)  # Replace with your actual function call to get Uxy values
    Uz_vals = Uzy(np.zeros(len(xvals)),zvals)  # Replace with your actual function call to get Uxy values


    # Fit polynomial of degree 2 (quadratic)
    x_coeffs = np.polyfit(xvals, Ux_vals, 6)  # The second argument here is the degree of the polynomial
    y_coeffs = np.polyfit(yvals, Uy_vals, 2)  # The second argument here is the degree of the polynomial
    z_coeffs = np.polyfit(zvals, Uz_vals, 2)  # The second argument here is the degree of the polynomial

    # number of ions / Electric potential function
    x_coeffs_rev = x_coeffs[::-1]
    y_coeffs_rev = y_coeffs[::-1]
    z_coeffs_rev = z_coeffs[::-1]

   

    @jit(nopython=True, fastmath=True)
    def potential_energy(positions):
        # wx, wy, wz, N, m, e, epsilon_0 = params
        xs = positions[:N]
        ys = positions[N:2*N]
        zs = positions[2*N:3*N]
        
        # Initialize potential energy terms
        Vx = np.zeros_like(xs)
        Vy = np.zeros_like(ys)
        Vz = np.zeros_like(zs)
        
        # Calculate Vx, Vy, and Vz using a manual loop for summing
      
        for j in range(len(x_coeffs_rev)):
            Vx += x_coeffs_rev[j] * ((1000*xs)**j)
            
        for j in range(len(y_coeffs_rev)):
            Vy += y_coeffs_rev[j] * ((1000*ys)**j)
            
        for j in range(len(z_coeffs_rev)):
            Vz += z_coeffs_rev[j] * ((1000*zs)**j) 
        
        # Harmonic energy
        harm_energy = e * np.sum(Vx + Vy + Vz) 
        
        # electronic interaction
        interaction = 0
        for i in range(N):
            for j in range(N):
                if j != i:
                    interaction += 1/np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)
        interaction = e**2/8/np.pi/eps_o * interaction
        
        # print(harm_energy+interaction)
        return harm_energy + interaction


    xs_0 = np.linspace(-1,1,N) * 1e-6
    ys_0 = np.linspace(-1,1,N) * 0
    zs_0 = np.linspace(-1,1,N) * 0

    pos = np.append(np.append(xs_0, ys_0), zs_0)

    # Minimize potential energy to determine ion crystal positions
    bounds = [(xMin*1e-3, xMax*1e-3)] * N + [(yMin*1e-3, yMax*1e-3)] * N + [(zMin*1e-3, zMax*1e-3)] * N  # bounds for N ions in x, y, z

    # run bad guess through optimization with on a few iterations
    pos_better = minimize(potential_energy, pos, method='COBYLA', bounds=bounds, options={'tol':1e-30, 'maxiter':500})

    # Get fine results with better initial best and many iterations
    res = minimize(potential_energy, pos_better.x, method='COBYLA', bounds=bounds, options={'tol':1e-30, 'maxiter':80000})

    # Plotting equilibrium positions and uniformity

    xs_f = res.x[:N]
    ys_f = res.x[N:2*N]
    zs_f = res.x[2*N:3*N]


    sorted_indices = np.argsort(xs_f)
    zs_f = zs_f[sorted_indices]
    xs_f = xs_f[sorted_indices]
    ys_f = ys_f[sorted_indices]
    
    # Axial Frequencies (x axis)
    const1 = e**2/8/np.pi/eps_o*(2/m)
    const2 = e*(2/m)

    def axial_eigensystem(x, y, z, coeffs, N):
   
      Anm = np.empty((N,N))

      for n in range(N):
        for m in range(N):
          if n == m:
            sum1 = 0.0
            for p in range(N):
              if(p!=m):
                sum1 += (2*(x[m] - x[p])**2 - (y[m] - y[p])**2 - (z[m] - z[p])**2) / abs((x[m] - x[p])**2+(y[m] - y[p])**2+(z[m] - z[p])**2)**(5/2)
            sum11 = 0.0
            for i in range(len(coeffs)):
                if i >= 2:
                    sum11 += coeffs[i]*i*(i-1)*(x[n]**(i-2))*((1e3)**i)
            Anm[n][n]= (const2*sum11 + const1*sum1)
          elif n!=m:
            sum2 = (2*(x[m] - x[n])**2 - (y[m] - y[n])**2 - (z[m] - z[n])**2) /abs((x[m] - x[n])**2+(y[m] - y[p])**2+(z[m] - z[n])**2)**(5/2)
            Anm[n][m]= -const1*sum2
            
      # print(Anm)

      eigenvalues, eigenvectors = np.linalg.eig(Anm)
      eigenvalues = np.sqrt(eigenvalues)
      eigenvectors = eigenvectors

      return eigenvalues, eigenvectors
     

    w_axial, axial_normal_modes = axial_eigensystem(xs_f, ys_f, zs_f, x_coeffs_rev, N)

    f_axial = w_axial / 2 / np.pi # Hz

    # print('Axial frequencies (kHz): ', f_axial/1e3)
    print('Axial COM frequency (kHz): ', np.min(f_axial)/1e3)
    print('theta_y: ', theta_y/ np.pi * 180)
    print('Uniformity (%): ', np.round(100*(1-np.std(np.diff(xs_f))/np.mean(np.diff(xs_f))),3))
    
    return theta_y / np.pi * 180, np.round(100*(1-np.std(np.diff(xs_f))/np.mean(np.diff(xs_f))),3), np.min(f_axial)/1e3


#%% 
#%% For loop Searching particular voltage set satisfying conditions(axial COM, uniformity, principal axes) 

from itertools import product
from tqdm import tqdm

# Define grid search range
V1_range = np.linspace(0, 50, 2)   # 2 points between 0 and 50
V2_range = np.linspace(-50, 50, 2)
V3_range = np.linspace(-50, 50, 2)
Vrf_range = np.linspace(300, 300, 1)

# Create grid search generator
grid = product(V1_range, V2_range, V3_range, Vrf_range)

# Calculate the number of valid grid points after applying the new constraint (V1 > V2 and V3 > V2)
valid_combinations = [
    (V1, V2, V3, Vrf) for V1, V2, V3, Vrf in product(V1_range, V2_range, V3_range, Vrf_range) 
    if V1 > V2 and V3 > V2
]
total_valid_points = len(valid_combinations)

# Step 1: Filter based on conditions (V1 > V2, V3 > V2, and theta and omega conditions)
filtered_set = []
for V1, V2, V3, Vrf in tqdm(grid, total=total_valid_points, desc="Filtering by constraints"):
    if V1 > V2 and V3 > V2:  # New constraints
        # Calculate U values once and reuse
        U_vals = U(V1, V2, V3, Vrf)
        
        # Apply all conditions using U_vals[0], U_vals[1], U_vals[2]
        if U_vals[0] < 0.5 and U_vals[1] > 96 and U_vals[2] > 200:
            filtered_set.append((V1, V2, V3, Vrf))

print(f"Filtered sets: {len(filtered_set)} found")

# Results
for V1, V2, V3, Vrf in filtered_set:
    print(f"V1={V1}, V2={V2}, V3={V3}, Vrf={Vrf}, "
          f"theta={theta(V1, V2, V3, Vrf)}, "
          f"omega={omega(V1, V2, V3, Vrf)}, "
          f"U={U(V1, V2, V3, Vrf)}")

    
#%% Global minimization (Bayesian optimization)

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import time

# List to store best candidates
best_candidates = []

# Define the objective function
def objective(params):
    V1, V2, V3, Vrf = params
    
    # Directly reject invalid points (enforcing the constraint)
    if not (V1 > V3 > V2):
        return np.inf  # Direct rejection by returning infinity
    
    try:
        U_vals = U(V1, V2, V3, Vrf)  # Compute U values
        
        penalty = 0
        # Priority 1: First condition (most important)
        if U_vals[0] >= 0.5:
            penalty += 1e6
        
        # Priority 2: Second condition
        if U_vals[1] <= 96:
            penalty += 1e5
        
        # Priority 3: Third condition
        if U_vals[2] <= 200:
            penalty += 1e2
        
        # Store intermediate candidates that are "almost" satisfying the constraints
        tolerance1 = 0.1  # Define how close is "good enough"
        tolerance2 = 1
        if (
            U_vals[0] < (0.5 + tolerance1) and
            U_vals[1] > (96 - tolerance2)
        ):
            best_candidates.append((V1, V2, V3, Vrf, penalty, U_vals))
        
        # If all conditions are satisfied, minimize U[0]
        if penalty == 0:
            return U_vals[0]
        else:
            return penalty
    
    except Exception as e:
        print(f"Error: {e} for params: {params}")
        return np.inf  # Direct rejection if an error occurs

# Provide an initial guess that satisfies V1 > V3 > V2
initial_point = [[20, 13, 16, 300]]

# Start the timestamp for execution time measurement
start_time = time.time()

# Perform Bayesian Optimization with the objective function
result = gp_minimize(
    objective,
    [Real(0, 100, name='V1'), Real(-100, 100, name='V2'), Real(-100, 100, name='V3'), Real(250, 400, name='Vrf')],
    n_calls=100,
    random_state=42,
    n_initial_points=1,  # Don't automatically generate initial points
    x0=initial_point,  # Use the provided initial point
    acq_func="EI",  # Expected Improvement for better exploration
)

# End the timestamp after the optimization is done
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")

# Print the best result
print(f"\nOptimal parameters: {result.x}")
print(f"Objective value: {result.fun}")

# Show stored best candidates
print("\nClose candidates:")
for candidate in best_candidates:
    V1, V2, V3, Vrf, penalty, U_vals = candidate
    print(f"V1={V1:.3f}, V2={V2:.3f}, V3={V3:.3f}, Vrf={Vrf:.3f}, "
          f"Penalty={penalty:.3f}, U={U_vals}")

print(f"\nObjective value: {result.fun}")

#%% Dual annealing

import numpy as np
from scipy.optimize import dual_annealing
import time

# List to store close results
best_candidates2 = []

# Transform parameters to enforce V1 > V3 > V2 constraint
def transform_params(params):
    a, b, c, Vrf = params
    V1 = a
    V3 = b
    V2 = c
    return V1, V2, V3, Vrf

# Objective function with priorities and storing close candidates
def objective(params):
    V1, V2, V3, Vrf = transform_params(params)
    
    # Enforce V1 > V3 > V2
    if not (V1 > V3 > V2):
        return 1e8  # Large penalty if the constraint is violated
    
    U_vals = U(V1, V2, V3, Vrf)
    
    penalty = 0
    high_priority_weight = 1e6
    mid_priority_weight = 1e5
    low_priority_weight = 1e2
    
    # First priority: U[0] < 0.5
    if U_vals[0] >= 0.5:
        penalty += mid_priority_weight * (U_vals[0] - 0.5)
    
    # Second priority: U[1] > 96
    if U_vals[1] <= 96:
        penalty += high_priority_weight * (96 - U_vals[1])
    
    # Third priority: U[2] > 200
    if U_vals[2] <= 200:
        penalty += low_priority_weight * (200 - U_vals[2])
    
    # ✅ Save intermediate solutions that are "close enough"
    tolerance1 = 0.1
    tolerance2 = 1
    tolerance3 = 10
    if (
        U_vals[0] < (0.5 + tolerance1) and
        U_vals[1] > (96 - tolerance2) 
    ):
        best_candidates2.append((V1, V2, V3, Vrf, penalty, U_vals))
    
    return penalty

# Bounds for reparameterized variables
lower_bound = [0, -100, -100, 250]
upper_bound = [100, 100, 100, 400]

# Initial guess
initial_guess = [1, -2, -30, 300]

# Start timer
start_time = time.time()

# Dual annealing optimization
result = dual_annealing(
    objective,
    bounds=list(zip(lower_bound, upper_bound)),
    x0=initial_guess,
    maxiter=100
)

# End timer
end_time = time.time()

# Best result
V1, V2, V3, Vrf = transform_params(result.x)
print(f"Optimal parameters: V1={V1:.3f}, V2={V2:.3f}, V3={V3:.3f}, Vrf={Vrf:.3f}")
print(f"Objective value: {result.fun:.3f}")
print(f"Execution time: {end_time - start_time:.2f} seconds")

# ✅ Show stored close candidates
print("\nClose candidates:")
for candidate2 in best_candidates2:
    V1, V2, V3, Vrf, penalty, U_vals = candidate2
    print(f"V1={V1:.3f}, V2={V2:.3f}, V3={V3:.3f}, Vrf={Vrf:.3f}, Penalty={penalty:.3f}, U={U_vals}")
    
    
#%% CMA method

import cma
import time

# List to store best candidates
best_candidates3 = []

# Objective function for CMA-ES with priorities and penalties
def objective(params):
    V1, V2, V3, Vrf = params
    try:
        # Call U with the current parameters
        U_vals = U(V1, V2, V3, Vrf)
        
        penalty = 0
        
        # ✅ Priority 1: First condition (most important)
        if U_vals[0] >= 0.5:
            penalty += 1e6
        
        # ✅ Priority 2: Second condition
        if U_vals[1] <= 96:
            penalty += 1e5
        
        # ✅ Priority 3: Third condition (least important)
        if U_vals[2] <= 200:
            penalty += 1e2
        
        # ✅ Store close candidates that nearly satisfy the constraints
        tolerance1 = 0.1  # Adjust for closeness threshold
        tolerance2 = 3
        if (
            U_vals[0] < (0.5 + tolerance1) and
            U_vals[1] > (96 - tolerance2) 
        ):
            best_candidates3.append((V1, V2, V3, Vrf, penalty, U_vals))
        
        # ✅ If all conditions are satisfied, minimize U[0]
        if penalty == 0:
            return U_vals[0]
        else:
            return penalty
    
    except Exception as e:
        print(f"Error: {e} for params: {params}")
        return 1e6  # Large penalty if an error occurs during computation

# Constraint: V1 > V3 > V2
def constraint(params):
    V1, V2, V3, Vrf = params
    return V1 > V3 > V2

# Set the bounds for the variables
lower_bound = [0, -100, -100, 250]   # Lower bounds for V1, V2, V3, Vrf
upper_bound = [100, 100, 100, 400]    # Upper bounds for V1, V2, V3, Vrf

# Create an initial guess
initial_guess = [1, -2, -30, 300]

# Start timestamp
start_time = time.time()

# Configure CMA-ES with bounds and population size
es = cma.CMAEvolutionStrategy(
    initial_guess, 1.0, 
    {'bounds': [lower_bound, upper_bound], 
     'popsize': 20,
     'maxiter': 1000}
)

# Optimization loop
while not es.stop():
    # Get the current population (candidates)
    solutions = es.ask()
    
    # ✅ Apply constraint V1 > V3 > V2
    fitness_values = [
        objective(sol) if constraint(sol) else float('inf') 
        for sol in solutions
    ]
    
    # Update the optimization state
    es.tell(solutions, fitness_values)
    
    # Show the status and progress
    es.disp()

# ✅ Best solution found
best_solution = es.result.xbest
print(f"\nOptimal parameters: {best_solution}")
print(f"Objective value: {objective(best_solution)}")

# ✅ Execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution Time: {execution_time:.2f} seconds")

# ✅ Show stored best candidates
print("\nClose candidates:")
for candidate3 in best_candidates3:
    V1, V2, V3, Vrf, penalty, U_vals = candidate3
    print(f"V1={V1:.3f}, V2={V2:.3f}, V3={V3:.3f}, Vrf={Vrf:.3f}, "
          f"Penalty={penalty:.3f}, U={U_vals}")
    



#%% Genetic algorithms 


import random
import numpy as np
from deap import base, creator, tools, algorithms
import time

# Define the problem as a minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def transform_params(params):
    a, b, c, Vrf = params
    V1 = a
    V3 = a - b  # Assuming you want V3 to be less than V1 by b
    V2 = V3 - c  # V2 should be less than V3 by c
    return V1, V2, V3, Vrf

def objective(individual):
    V1, V2, V3, Vrf = transform_params(individual)
    
    # Enforce V1 > V3 > V2
    if not (V1 > V3 > V2):
        return 1e8,  # High penalty for constraint violation
    
    penalty = 0
    high_priority_weight = 1e6
    mid_priority_weight = 1e5
    low_priority_weight = 1e1
    
    # Apply penalties for conditions (without U function)
    if V1 >= 0.5:  # Replace with your actual conditions
        penalty += mid_priority_weight * (V1 - 0.5)
    if V2 <= 96:  # Replace with your actual conditions
        penalty += high_priority_weight * (96 - V2)
    if V3 <= 200:  # Replace with your actual conditions
        penalty += low_priority_weight * (200 - V3)
    
    return penalty,  # Return a tuple with the penalty value

# Create the initial population from your provided voltages
def create_individual(initial_voltages):
    return initial_voltages

# Example of initial voltages (V1, V2, V3, Vrf)
initial_voltages_list = [
    [20, 13, 16, 300],  # First set of initial voltages
    [25, 15, 18, 310],  # Second set of initial voltages
    [30, 18, 20, 305],  # Third set of initial voltages
    # Add more sets of initial voltages as needed
]

# Create a toolbox for the GA with the given initial voltages
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective)

# Set parameters for GA
population_size = len(initial_voltages_list)  # Set population size based on your initial voltages list
generations = 50
cx_prob = 0.7
mut_prob = 0.2

# Start the timestamp for execution time measurement
start_time = time.time()

# Initialize population with your provided initial voltages
population = toolbox.population(n=population_size)

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations, 
                    stats=None, halloffame=None, verbose=True)

# End the timestamp
end_time = time.time()

# Get the best individual (solution)
best_individual = tools.selBest(population, 1)[0]
V1, V2, V3, Vrf = transform_params(best_individual)
print(f"Optimal parameters: V1={V1:.3f}, V2={V2:.3f}, V3={V3:.3f}, Vrf={Vrf:.3f}")
print(f"Objective value: {objective(best_individual)[0]:.3f}")
print(f"Execution time: {end_time - start_time:.2f} seconds")

#%% Reinforcement Learning method

import numpy as np
import random
import time

# Define parameters for Q-learning
ALPHA = 0.1   # Learning rate
GAMMA = 0.9   # Discount factor
EPSILON = 0.2 # Exploration rate
N_EPISODES = 100  # Number of episodes
MAX_STEPS = 100  # Maximum steps per episode

# Initialize the Q-table (state-action values)
q_table = {}

# Transform voltages to enforce constraints and calculate penalty
def transform_params(params):
    a, b, c, Vrf = params
    V1 = a
    V3 = a - b  # V3 < V1
    V2 = V3 - c  # V2 < V3
    return V1, V2, V3, Vrf

def objective(params):
    V1, V2, V3, Vrf = transform_params(params)
    
    # Apply penalties based on your conditions (replace with your real conditions)
    penalty = 0
    high_priority_weight = 1e6
    mid_priority_weight = 1e5
    low_priority_weight = 1e1
    
    if V1 >= 0.5:  # Example condition (replace with actual)
        penalty += mid_priority_weight * (V1 - 0.5)
    if V2 <= 96:  # Example condition (replace with actual)
        penalty += high_priority_weight * (96 - V2)
    if V3 <= 200:  # Example condition (replace with actual)
        penalty += low_priority_weight * (200 - V3)
    
    # Return the penalty as the reward
    return penalty

# Define the state-action space
def discretize_state(state):
    # Example discretization of state space (adjust if needed)
    return tuple(np.digitize(state, bins=np.linspace(-50, 50, 11)))

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:  # Exploration
        return random.choice(range(4))  # 4 possible actions: change V1, V2, V3, or Vrf
    else:  # Exploitation
        return np.argmax(q_table.get(state, [0, 0, 0, 0]))

# Update the Q-table based on the action taken
def update_q_table(state, action, reward, next_state):
    current_q_value = q_table.get(state, [0, 0, 0, 0])[action]
    max_next_q_value = max(q_table.get(next_state, [0, 0, 0, 0]))
    new_q_value = current_q_value + ALPHA * (reward + GAMMA * max_next_q_value - current_q_value)
    
    # Update the Q-table
    q_table[state][action] = new_q_value

# Run the Q-learning process
def q_learning():
    total_rewards = []
    
    for episode in range(N_EPISODES):
        state = np.random.uniform(-50, 50, 4)  # Random initial state (V1, V2, V3, Vrf)
        total_reward = 0
        
        for step in range(MAX_STEPS):
            action = choose_action(discretize_state(state))  # Choose an action
            
            # Apply action (e.g., update one of the voltages)
            new_state = state.copy()
            if action == 0:
                new_state[0] += random.uniform(-1, 1)  # V1
            elif action == 1:
                new_state[1] += random.uniform(-1, 1)  # V2
            elif action == 2:
                new_state[2] += random.uniform(-1, 1)  # V3
            elif action == 3:
                new_state[3] += random.uniform(-1, 1)  # Vrf
            
            reward = -objective(new_state)  # Negative penalty as the reward (since we minimize penalty)
            total_reward += reward
            
            # Update the Q-table
            update_q_table(discretize_state(state), action, reward, discretize_state(new_state))
            
            # Move to the next state
            state = new_state
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}/{N_EPISODES}, Total Reward: {total_reward}")
    
    return total_rewards

# Run the Q-learning algorithm
start_time = time.time()
total_rewards = q_learning()
end_time = time.time()

print(f"\nExecution time: {end_time - start_time:.2f} seconds")