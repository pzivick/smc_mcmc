'''
pm_func_edr3.py

Based on the original pm_func.py library from my old SMC DR2 work.
Most of those were built on using just numpy arrays without headers.
All of the functions below are written with headers involved and with an
eye for the updated EDR3 catalogs.

'''

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib
import warnings
from astropy.modeling import models, fitting
from astropy.modeling import functional_models as funcmod
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy
import astropy.units as u
import astropy.coordinates as coord
import emcee
import time


################################################################
#### Convert to Gaia cartesian coordinates

def wcs2gaiaxy(ra, dec, center):
    x = np.cos(dec) * np.sin(ra-center[0])
    y = np.sin(dec) * np.cos(center[1]) - np.cos(dec) * np.sin(center[1]) * np.cos(ra - center[0])

    x = np.rad2deg(x)
    y = np.rad2deg(y)

    return x,y

################################################################

################################################################
#### Translate the mean proper motions into velocities

def pm2vel(mue, mun, dist):

    # Total center of mass (CM) proper motion
    mutran = (mue**2 + mun**2)**(0.5)

    # angle of motion for the CM PM and convert from traditional definition of on-sky angles
    thtran = np.deg2rad(np.rad2deg(np.arctan2(mue, mun)) + 90.0)

    # total CM velocity
    vtran = 4.7403885 * mutran * dist

    return thtran, vtran

################################################################

################################################################
#### Calculate the distance of points in a skyplane
#### Taken from Eq. 8 of vdMC01

def calc_dist(rho, phi, dist0, incl=0.0000001, theta=0.0000001):

    dist = dist0 * ( np.cos(incl) / ((np.cos(incl) * np.cos(rho)) - \
    (np.sin(incl) * np.sin(rho) * np.sin(phi - theta))))

    return dist

################################################################

################################################################
#### Function to calculate the viewing perspective correction
#### for a given system and set of points
#### Required inputs:
# vtran (the transverse velocity total
# thtran (angle of the transverse vector)
# vsys (systemic radial velocity)
# thet (angle of intersection between the galaxy and viewing plane)
# dist0 (distance to the assumed center of the galaxy)
# ra0 - observed RA center of the galaxy
# dec0 - observed Dec center of the galaxy
# ra - column from an array containing the RA coordinates of the points to calculate
# dec - column from an array containing the Dec coordinates of the points to calculate
# incl - assumed inclination angle of the galaxy to the plane of the sky

def viewing_cor(vtran, thtran, vsys, dist0, ra0, dec0, ra, dec, incl=0.0001, theta=0.0001):

# Calculate observational coordinates of the input columns
    rho, phi = wcs2ang(ra0, dec0, ra, dec)

# Calculate the vector components of the CoM motion using given coordinates and parameters
    v1, v2, v3 = make_cm_angvec(vtran, thtran, vsys, rho, phi)

# Calculate the gamma factor for the proper motion calculation
    cosG, sinG = calc_gamma(ra0, dec0, ra, dec, rho)

# Calculate the proper motion for each coordinate
    muw, mun = ang2wcs_vec(dist0, v2, v3, cosG, sinG, rho, phi, theta=theta, incl=incl)

# Calculate the CoM proper motion at the assumed center
    muwsys = (np.cos(thtran) * vtran) / (4.7403895 * dist0)
    munsys = (np.sin(thtran) * vtran) / (4.7403895 * dist0)


# Subtract the CoM PM from the calculated coordinate PMs to get the residuals
    muwcor = muw - muwsys
    muncor = mun - munsys

    rvcor = v1 - vsys

    return muwcor, muncor, rvcor

################################################################

################################################################
#### Function to convert from WCS to angular coordinates

def wcs2ang(ra0, dec0, ra, dec):

    rho = np.arccos( np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0) + np.sin(dec) * np.sin(dec0) )

    phi = np.arccos( (-1.0*np.cos(dec) * np.sin(ra - ra0)) / (np.sin(rho)) )
  #phi = np.arcsin( ((np.sin(dec)*np.cos(dec0)) - (np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0))) / (np.sin(rho)) )

# Calculate Cartesian coordinates to test for correcting the angle

    testx, testy = wcs2gaiaxy(ra, dec, np.asarray([ra0, dec0]))

    for i in range(len(testy)):
        if (testy[i] < 0.0):
            phi[i] = -1.0*phi[i]

    return rho, phi

################################################################

################################################################
#### Function to convert velocities in the vdM02 vx, vy, vz
## space into the more spherical v1, v2, v3 space

def vel_xyz2sph(vel, theta, incl, rho, phi):

# Calculate the components of the transformation matrix from v' to v

    la = np.sin(rho) * np.cos(phi-theta)
    lb = np.cos(rho) * np.cos(phi-theta)
    lc = -1.0 * np.sin(phi-theta)
    ld = np.sin(rho) * np.cos(incl) * np.sin(phi-theta) + np.cos(rho) * np.sin(incl)
    le = np.cos(rho) * np.cos(incl) * np.sin(phi-theta) - np.sin(rho) * np.sin(incl)
    lf = np.cos(incl) * np.cos(phi-theta)
    lg = np.sin(rho) * np.sin(incl) * np.sin(phi-theta) - np.cos(rho) * np.cos(incl)
    lh = np.cos(rho) * np.sin(incl) * np.sin(phi-theta) + np.sin(rho) * np.cos(incl)
    li = np.sin(incl) * np.cos(phi-theta)

    vx, vy, vz = 0, 1, 2

    v1 = la * vel[:,vx] + ld * vel[:,vy] + lg * vel[:,vz]
    v2 = lb * vel[:,vx] + le * vel[:,vy] + lh * vel[:,vz]
    v3 = lc * vel[:,vx] + lf * vel[:,vy] + li * vel[:,vz]
#
    return v1, v2, v3

################################################################

################################################################
#### Function to convert X/Y coordinates (that were transformed from RA/Dec) to angular coordinates

def wcsxy2ang(x, y, dist0):

    rho = np.arctan2( ((x**2 + y**2)**(0.5)), (dist0**(0.5)))
    phi = np.arctan2(y, x)

    return rho, phi

################################################################

##########################################################################
#### Function to create vectors (v1, v2, v3) for the center of
#### mass motion

def make_cm_angvec(vtran, thtran, vsys, rho, phi):

    v1 = vtran*np.sin(rho)*np.cos(phi - thtran) + vsys*np.cos(rho)
    v2 = vtran*np.cos(rho)*np.cos(phi - thtran) - vsys*np.sin(rho)
    v3 = -1.0 * vtran * np.sin(phi - thtran)

    return v1, v2, v3

##########################################################################

##########################################################################
#### Function to calculate the Gamma factor for vector transformations

def calc_gamma(ra0, dec0, ra, dec, rho):

    cosG = (np.sin(dec) * np.cos(dec0) * np.cos(ra-ra0) - np.cos(dec)*np.sin(dec0)) / np.sin(rho)

    sinG = (np.cos(dec0) * np.sin(ra - ra0)) / np.sin(rho)

    return cosG, sinG

##########################################################################

##########################################################################
#### Function to convert an angular vector to wcs

def ang2wcs_vec(dist0, v2, v3, cosG, sinG, rho, phi, theta=np.deg2rad(0.00001), incl=np.deg2rad(0.00001)):

# Calculates the scaling quantity for the proper motion

    propercon = (np.cos(incl) * np.cos(rho) - np.sin(incl)*np.sin(rho)*np.sin(phi-theta)) \
    / (dist0 * np.cos(incl))

# Use the scaling quantity and the vector components in the skyplane
# to calculate the observable proper motions

    muwe = (propercon * (-1.0*sinG*(v2) - cosG*(v3))) / (4.7403895)
    muno = (propercon * (cosG*(v2) - sinG*(v3))) / (4.7403895)

    return muwe, muno

##########################################################################

################################################################
#### Convert uRA uDec vectors to Gaia cartesian coordinates

def vecwcs2gaiaxy(ura, udec, ra, dec, center):

    ux = ura * np.cos(ra - center[0]) - udec * np.sin(dec) * np.sin(ra - center[0])

    uy = ura * np.sin(center[1]) * np.sin(ra-center[0]) + udec * (np.cos(dec) * np.cos(center[1]) + np.sin(dec) * np.sin(center[1]) * np.cos(ra-center[0]))

    return ux, uy


###################################################################

###################################################################
#### Convert numpy array to a tuple list (for use in the mask code)

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

###################################################################

###################################################################
#### Function to select all points within a provided polygon

def select_points(array, ind1, ind2, verts):

    tupverts = totuple(verts)
    path = Path(tupverts)
    select = path.contains_points(np.transpose(np.asarray([array[ind1], array[ind2]])))

    newarray = np.copy(array[select])

    return newarray

###################################################################

################################################################
#### Averages given quantities based on gridding for other
#### quantities

def vec_grid(array, xwidth, xmax, xmin, ywidth, ymax, ymin, xind, yind, xvind, yvind, avg3 = False, v3ind = 0):

    xn = int((xmax - xmin)/xwidth)
    yn = int((ymax - ymin)/ywidth)

    if (avg3):
        new = np.zeros((xn*yn, 5))
        newx, newy, newvx, newvy, newv3 = 0, 1, 2, 3, 4

    else:
        new = np.zeros((xn*yn, 4))
        newx, newy, newvx, newvy = 0, 1, 2, 3

    for i in range(xn):
        for j in range(yn):
            xlow = xmin + i*xwidth
            xhigh = xmin + (i+1)*xwidth
            ylow = ymin + j*ywidth
            yhigh = ymin + (j+1)*ywidth

            ind = i*yn + j

            new[ind][newx] = (xlow+xhigh)/2.0
            new[ind][newy] = (ylow+yhigh)/2.0

            subset = array[(array[xind] > xlow) & (array[xind] < xhigh) & \
                     (array[yind] > ylow) & (array[yind] < yhigh)]

            if (len(subset) == 0):
                new[ind][newvx] = 0.0
                new[ind][newvy] = 0.0

                if (avg3):
                    new[ind][newv3] = 0.0

            else:
                new[ind][newvx] = np.average(subset[xvind])
                new[ind][newvy] = np.average(subset[yvind])


                if (avg3):
                    new[ind][newv3] = np.average(subset[v3ind])

    return new

################################################################

################################################################
####

def xy2rt(x, y, vx, vy, test=False, error=False, vxerr=0, vyerr=0):

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vr = (x*vx + y*vy)/(np.sqrt(x**2 + y**2))
    vt = (x*vy - vx*y)/(np.sqrt(x**2 + y**2))


    if (test):
        for a in range(len(x)):
            print(vx[a], vy[a], vr[a], vt[a])

    if (error):
        vrerr = (x*vxerr + y*vyerr)/(np.sqrt(x**2 + y**2))
        vterr = (x*vyerr - vxerr*y)/(np.sqrt(x**2 + y**2))

        return r, theta, vr, vt, vrerr, vterr

    else:
        return r, theta, vr, vt

################################################################

################################################################
#### Averages given quantities based on gridding for other
#### quantities

def vec_1dgrid(array, xwidth, xmax, xmin, xind, zind, weighted=False, zerrind=0, testing=False):

    nsteps = int((xmax - xmin)/xwidth)

    new = np.zeros((nsteps, 4))
    newx, zavg, zstd, znum = 0, 1, 2, 3

    for i in range(nsteps):
        xlow = xmin + i*xwidth
        xhigh = xmin + (i+1)*xwidth

        ind = i

        new[ind][newx] = (xlow+xhigh)/2.0

        subset = array[(array[:,xind] > xlow) & (array[:,xind] < xhigh)]

        if (len(subset) == 0):
            new[ind][zavg] = 0.0
            new[ind][zstd] = 0.0
            new[ind][znum] = 0.0

        else:

            if (weighted):
                zweight = 1.0 / (subset[:,zerrind]**2)
                new[ind][zavg] = np.average(subset[:,zind], weights=zweight)
                new[ind][zstd] = np.sqrt(1.0/(np.sum(zweight)))
                new[ind][znum] = len(subset)

            else:
                new[ind][zavg] = np.average(subset[:,zind])
                new[ind][zstd] = np.std(subset[:,zind])
                new[ind][znum] = len(subset)


    return new

################################################################

################################################################
#### Function to create a horizontal expansion in a given frame
#### Note that for some rotations, need to enter the negative
#### of the desired angle for rotation

def rot3d(x, y, z, matrix):

    new = np.zeros((len(x),3))

    new[:,0] = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z
    new[:,1] = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z
    new[:,2] = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z

    return new

################################################################

################################################################
#### Function to calculate the rotation matrix for a cartesian
#### system about the z-axis given an input angle

def rot_xaxis(ang):
    temp = np.zeros((3,3))

  #first row
    temp[0][0] = 1.0
    temp[0][1] = 0.0
    temp[0][2] = 0.0

  #second row
    temp[1][0] = 0.0
    temp[1][1] = np.cos(ang)
    temp[1][2] = -np.sin(ang)

  #third row
    temp[2][0] = 0.0
    temp[2][1] = np.sin(ang)
    temp[2][2] = np.cos(ang)

    return temp

################################################################

################################################################
#### Function to calculate the rotation matrix for a cartesian
#### system about the z-axis given an input angle

def rot_yaxis(ang):
    temp = np.zeros((3,3))

  #first row
    temp[0][0] = np.cos(ang)
    temp[0][1] = 0.0
    temp[0][2] = np.sin(ang)

  #second row
    temp[1][0] = 0.0
    temp[1][1] = 1.0
    temp[1][2] = 0.0

  #third row
    temp[2][0] = -np.sin(ang)
    temp[2][1] = 0.0
    temp[2][2] = np.cos(ang)

    return temp

################################################################

################################################################
#### Function to calculate the rotation matrix for a cartesian
#### system about the z-axis given an input angle

def rot_zaxis(ang):
    temp = np.zeros((3,3))

  #first row
    temp[0][0] = np.cos(ang)
    temp[0][1] = -1.0 * np.sin(ang)
    temp[0][2] = 0.0

  #second row
    temp[1][0] = np.sin(ang)
    temp[1][1] = np.cos(ang)
    temp[1][2] = 0.0

  #third row
    temp[2][0] = 0.0
    temp[2][1] = 0.0
    temp[2][2] = 1.0

    return temp

################################################################

################################################################
#### Function to convert angular coordinates to cartesian in
#### the frame of the galaxy rotation

def ang2xyz(rho, phi, dist, dist0, theta=0.00001, incl=0.00001):

    new = np.zeros((len(phi),3))

    new[:,0] = dist * np.sin(rho) * np.cos(phi - theta)

    new[:,1] = dist * (np.sin(rho) * np.cos(incl) * np.sin(phi-theta) + np.cos(rho) * np.sin(incl)) - (dist0 * np.sin(incl))

    new[:,2] = dist * (np.sin(rho) * np.sin(incl) * np.sin(phi-theta) - np.cos(rho) * np.cos(incl)) + (dist0 * np.cos(incl))

    return new

################################################################

################################################################
#### Function to create vectors (v1, v2, v3) for the internal
#### motions of the galaxy by first creating (vx', vy', vz')
#### given a rotating velocity field

def make_int_angvec_plane(rad0, vel0, theta, incl, dist0, rho, phi, dist, checkcurve=False, checkfile="", usevdM02=False, n0=0.0):

# Calculates the (vx', vy', vz') vector in the frame of the galaxy

    newcoord = ang2xyz(rho, phi, dist, dist0, theta=theta, incl=incl)
    x1, y1, z1 = 0, 1, 2

    vframe = np.zeros((len(rho),3))
    vx, vy, vz = 0, 1, 2

    if (checkcurve):
        if (usevdM02):
            rotval = rotcurve_vdM02(newcoord[:,x1], newcoord[:,y1], rad0, vel0, checkcurve=checkcurve, checkfile=checkfile, n0=n0)
        else:
            rotval = rotcurve(newcoord[:,x1], newcoord[:,y1], rad0, vel0, checkcurve=checkcurve, checkfile=checkfile)
    else:
        if (usevdM02):
            rotval = rotcurve_vdM02(newcoord[:,x1], newcoord[:,y1], rad0, vel0, n0=n0)
        else:
            rotval = rotcurve(newcoord[:,x1], newcoord[:,y1], rad0, vel0)

    rotval.shape = (len(rho),)



    vframe[:,vx] = rotval * (newcoord[:,y1]/(newcoord[:,x1]**2 + newcoord[:,y1]**2)**(0.5))
    vframe[:,vy] = -1.0 * rotval * (newcoord[:,x1]/(newcoord[:,x1]**2 + newcoord[:,y1]**2)**(0.5))

# Calculate the components of the transformation matrix from v' to v

    v1, v2, v3 = vel_xyz2sph(vframe, theta, incl, rho, phi)
#

    return v1, v2, v3

################################################################

################################################################
#### Function to create a rotation curve in the frame of the
#### galaxy

def rotcurve(x, y, r0, v0, checkcurve=False, checkfile=""):
  rad = (x**2 + y**2)**(0.5)
  vrot = np.zeros((len(rad),1))

  for ii in range(len(rad)):
    if (rad[ii] < r0):
      vrot[ii] = (rad[ii]/r0) * v0
    else:
      vrot[ii] = v0

  if (checkcurve):
    plt.clf()
    plt.scatter(rad, vrot, s=4, marker="+", color="gray", alpha=0.8)
    plt.xlabel(r'Radius (kpc)')
    plt.ylabel(r'Velocity (km/s)')
    plt.ylim(0.0, 60.0)
    plt.tight_layout()
    plt.savefig(checkfile)

  return vrot

################################################################

################################################################
#### Convert spherical angular coordinates to sky xyz vframe

def sph_2_skyxyz(phi, theta, r):

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z

################################################################

################################################################
#### Convert from sky plane xyz to galaxy x'y'z'

def skyxyz_2_galxyz(x, y, z, theta, incl):

    gx = x*np.cos(theta) + y*np.sin(theta)

    gy = -1.*x*np.sin(theta)*np.cos(incl) + y*np.cos(theta)*np.cos(incl) - z*np.sin(incl)

    gz = -1.*x*np.sin(theta)*np.sin(incl) + y*np.cos(theta)*np.sin(incl) + z*np.cos(incl)

    return gx, gy, gz



################################################################
#### Calculate the tidal expansion components

def calc_tidal(theta, incl, dist0, rho, phi, dist, phi_tid, theta_tid, v_tid):

    newcoord = ang2xyz(rho, phi, dist, dist0, theta=theta, incl=incl)
    xg, yg, zg = 0, 1, 2

    vframe = np.zeros((len(rho),3))
    vx, vy, vz = 0, 1, 2

    xs_tid, ys_tid, zs_tid = sph_2_skyxyz(phi_tid, theta_tid, 1.0)

    xg_tid, yg_tid, zg_tid = skyxyz_2_galxyz(xs_tid, ys_tid, zs_tid, theta, incl)

    dist_tid = xg_tid*newcoord[:,xg] + yg_tid*newcoord[:,yg] + zg_tid*newcoord[:,zg]

    vframe[:,vx] = xg_tid * dist_tid * v_tid
    vframe[:,vy] = yg_tid * dist_tid * v_tid
    vframe[:,vz] = zg_tid * dist_tid * v_tid

    v1, v2, v3 = vel_xyz2sph(vframe, theta, incl, rho, phi)

    return v1, v2, v3

################################################################

################################################################
#### Function to calculate the expected proper motions for a
#### given set of RA/Dec locations and model parameters
#
## Required inputs:
# vtran: the transverse velocity total
# thtran: angle of the transverse vector
# vsys: systemic radial velocity
# dist0: distance to the center of the galaxy
# ra0: RA center of the galaxy
# dec0: Dec center of the galaxy
# ra: column from an array containing the RA coordinates of the points to calculate the CM vector
# dec: column from an array containing the Dec coordinates of the points to calculate the CM vector
# incl: inclination angle of the galaxy to the plane of the sky
# theta: angle of intersection between the galaxy and viewing plane
#
## Outputs:
#

def pmmodel(muE0, muN0, dist0, vsys, ra0, dec0, ra, dec, incl, theta, rad0, rotvel0, phi_tid, theta_tid, v_tid):

# Calculate the transverse velocity and angle of the CoM
    vtran, thtran = calc_transverse(muE0, muN0, dist0)

# Calculate observational coordinates of the input columns
    rho, phi = wcs2ang(ra0, dec0, ra, dec)

# Calculate the distance for each star, based on its location and the plane orientation
    dist = calc_dist(rho, phi, dist0, incl=incl, theta=theta)

# Calculate the vector components of the CoM motion using given coordinates and parameters
    v1_cm, v2_cm, v3_cm = make_cm_angvec(vtran, thtran, vsys, rho, phi)

# Calculate the vector components of the internal rotation
    v1_rot, v2_rot, v3_rot = make_int_angvec_plane(rad0, rotvel0, theta, incl, dist0, rho, phi, dist)

# Calculate the vector components of the tidal expansion
    v1_tid, v2_tid, v3_tid = calc_tidal(theta, incl, dist0, rho, phi, dist, phi_tid, theta_tid, v_tid)

    v1 = v1_cm + v1_rot + v1_tid
    v2 = v2_cm + v2_rot + v2_tid
    v3 = v3_cm + v3_rot + v3_tid

# Calculate the gamma factor for the proper motion calculation
    cosG, sinG = calc_gamma(ra0, dec0, ra, dec, rho)

# Calculate the proper motion for each coordinate
    muw, mun = ang2wcs_vec(dist0, v2, v3, cosG, sinG, rho, phi, theta=theta, incl=incl)

# Flip the mu_west PMs to mu_east orientation
    mue = -1.0 * muw

    return mue, mun, v1

######################################################################

################################################################
####

def calc_transverse(muE, muN, dist):

    mutran = (muE**2 + muN**2)**(0.5)         #total CM proper motion

    thtran = np.deg2rad(np.rad2deg(np.arctan2(muE,muN)) + 90.0) #angle of motion for the CM PM

    vtran = 4.7403885 * mutran * dist        #total CM velocity

    return vtran, thtran

################################################################

################################################################
#### Function to take in RA, Dec and muRA, muDec
#### and output ready to plot vectors for total PM
#### and relative PM

def plot_prep(ra, dec, mura, mudec, ra0, dec0, mura0, mudec0, dist0, vsys, incl, theta):

    vtran, thtran = calc_transverse(mura0, mudec0, dist0)

    viewcor = np.zeros((len(ra), 3))
    w, n, crv = 0, 1, 2

    viewcor[:,w], viewcor[:,n], viewcor[:,crv] = viewing_cor(vtran, thtran, vsys, dist0, ra0, dec0, ra, dec, incl=incl, theta=theta)

    muW = -1.*mura0

    mura_rel = mura - (-1.0*(muW + viewcor[:,w]))
    mudec_rel = mudec - (mudec0 + viewcor[:,n])

    ##

    mux, muy = vecwcs2gaiaxy(mura, mudec, ra, dec, [ra0, dec0])

    mux.shape = (len(mux),)
    muy.shape = (len(muy),)

    mux_rel, muy_rel = vecwcs2gaiaxy(mura_rel, mudec_rel, ra, dec, [ra0, dec0])

    mux_rel.shape = (len(mux_rel),)
    muy_rel.shape = (len(muy_rel),)



    return mux, muy, mux_rel, muy_rel

################################################################

################################################################
#### Function to load in priors and prior ranges

def get_priors(perm):

    if perm == 'standard':
        nparams = 11
        params = ['pmra0', 'pmdec0', 'thet', 'incl', 'rad0', 'rotvel', 'sigma_pmra', 'sigma_pmdec', \
                  'phi_tid', 'theta_tid', 'vel_tid']
        priors = np.asarray([[-5., 5.], [-5., 5.], [0., 360.], [-90., 90.], [-2., 1.], [-50., 50.],
                            [-2., 0.], [-2., 0.], [0., 360.], [-90., 90.], [-1., 2.]])

    elif perm == 'full':
        nparams = 15
        params = ['pmra', 'pmdec', 'thet', 'incl', 'rad0', 'rotvel', 'sigma_pmra', 'sigma_pmdec', \
                  'phi_tid', 'theta_tid', 'vel_tid', 'ra0', 'dec0', 'dist0', 'vsys']
        priors = np.asarray([[-5., 5.], [-5., 5.], [0., 360.], [-90., 90.], [-2., 1.], [-50., 50.], \
                            [-3., 0.], [-3., 0.], [0., 360.], [-90., 90.], [-1., 2.], \
                            [0., 30.], [-75., -65.],[55., 65.], 120., 170.])

    else:
        nparams = 11
        params = ['pmra', 'pmdec', 'thet', 'incl', 'rad0', 'rotvel', 'sigma_pmra', 'sigma_pmdec', \
                  'phi_tid', 'theta_tid', 'vel_tid']
        priors = np.asarray([[-5., 5.], [-5., 5.], [0., 360.], [-90., 90.], [-2., 1.], [-50., 50.],
                            [-3., 0.], [-3., 0.], [0., 360.], [-90., 90.], [-1., 2.]])

    return priors, nparams, params

################################################################

################################################################

def log_prob(theta):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5*np.sum(theta**2)

################################################################

################################################################
#### Function to create a 2D color map based on a 3rd quantity

def create_grid(array, xwidth, xmax, xmin, ywidth, ymax, ymin, xind, yind, \
zind=0, countn=False):

    xn = int((xmax - xmin)/xwidth)
    yn = int((ymax - ymin)/ywidth)

    xgrid = np.zeros((yn, xn))
    ygrid = np.zeros((yn, xn))
    zgrid = np.zeros((yn, xn))

    for i in range(xn):
        for j in range(yn):
            xlow = xmin + i*xwidth
            xhigh = xmin + (i+1)*xwidth
            ylow = ymin + j*ywidth
            yhigh = ymin + (j+1)*ywidth

            ind = i*yn + j

            xgrid[j][i] = (xlow+xhigh)/2.0
            ygrid[j][i] = (ylow+yhigh)/2.0

            subset = array[(array[xind] > xlow) & (array[xind] < xhigh) & \
                     (array[yind] > ylow) & (array[yind] < yhigh)]

            if (len(subset) == 0):
                zgrid[j][i] = 0.0

            else:
                if (countn):
                    zgrid[j][i] = len(subset)
                else:
                    zgrid[j][i] = np.average(subset[zind])

    return xgrid, ygrid, zgrid

################################################################

################################################################
#### Generalized function to transform between xyz frames

def xyz_transform(x, y, z, theta, incl, direction="2sky"):

    position = np.zeros((len(x), 3, 1))
    position[:,0] = x
    position[:,1] = y
    position[:,2] = z

    matrix = np.zeros((3, 3))
    matrix[0][0] = np.cos(theta)
    matrix[0][1] = -1.*np.sin(theta)*np.cos(incl)
    matrix[0][2] = -1.*np.sin(theta)*np.sin(incl)
    matrix[1][0] = np.sin(theta)
    matrix[1][1] = np.cos(theta)*np.cos(incl)
    matrix[1][2] = np.cos(theta)*np.sin(incl)
    matrix[2][0] = 0.
    matrix[2][1] = -1.*np.sin(incl)
    matrix[2][2] = np.cos(incl)

    if (direction=="2sky"):
        temp = np.dot(matrix, position)

    else:
        temp = np.dot(np.linalg.inv(matrix), position)

    x2 = np.reshape(temp[0], (len(temp[0]),))
    y2 = np.reshape(temp[1], (len(temp[1]),))
    z2 = np.reshape(temp[2], (len(temp[2]),))

    return x2, y2, z2

################################################################
#### Function to transform WCS vectors (PM_west, PM_north,
#### RV) to (vx, vy, vz) coordinates

def wcs2xyz_vec(ra, dec, rho, phi, dist, ra0, dec0, pmw, pmn, v1):

    # First handle the matrix for between WCS and v2/v3
    cosG, sinG = calc_gamma(ra0, dec0, ra, dec, rho)

    matrix = np.zeros((len(ra), 2, 2))
    matrix[:,0,0] = -1.*sinG
    matrix[:,0,1] = -1.*cosG
    matrix[:,1,0] = cosG
    matrix[:,1,1] = sinG

    # Next prepare the vectors for pmw/pmn and v2,v3

    pm = np.zeros((len(ra), 2, 1))
    pm[:,0] = pmw
    pm[:,1] = pmn

    vec_ang = np.zeros((len(ra), 3, 1))
    vec_ang[:,0] = v1


    temp = np.zeros((len(ra), 2, 1))
    # Now transform between the frames
    for i in range(len(matrix)):
        temp[i] = dist[i] * np.dot(np.linalg.inv(matrix[i]),pm[i])

    vec_ang[:,1] = temp[:,0]
    vec_ang[:,2] = temp[:,1]


    # Now create the next matrix for going from v123 to vxyz

    matrixB = np.zeros((len(ra), 3, 3))
    matrixB[:,0,0] = np.sin(rho)*np.cos(phi)
    matrixB[:,0,1] = np.sin(rho)*np.sin(phi)
    matrixB[:,0,2] = -1.*np.cos(rho)
    matrixB[:,1,0] = np.cos(rho)*np.cos(phi)
    matrixB[:,1,1] = np.cos(rho)*np.sin(phi)
    matrixB[:,1,2] = np.sin(rho)
    matrixB[:,2,0] = -1.*np.sin(phi)
    matrixB[:,2,1] = np.cos(phi)
    matrixB[:,2,2] = 0.

    tempB = np.zeros((len(ra), 3, 1))
    for i in range(len(matrixB)):
        tempB[i] = np.dot(np.linalg.inv(matrixB[i]), vec_ang[i])

    vx = np.reshape(tempB[:,0], (len(temp),))
    vy = np.reshape(tempB[:,1], (len(temp),))
    vz = np.reshape(tempB[:,2], (len(temp),))

    return vx, vy, vz

################################################################

################################################################
#### Function to calculate the slit function of a third property
#### on a 1D axis after averaging over a second axis.
#### e.g., average of Z as a function of X averaged over an X/Y
#### window

def slit_function(array, xwidth, xmax, xmin, \
ymax, ymin, xind, yind, zind=0, countn=False, func='avg'):

    xn = int((xmax - xmin) / xwidth)

    xvals = np.zeros((xn, ))
    zvals = np.zeros((xn, ))

    for i in range(xn):
        xlow = xmin + i*xwidth
        xhigh = xmin + (i+1)*xwidth

        xvals[i] = (xlow+xhigh)/2.0

        subset = array[(array[xind] > xlow) & \
        (array[xind] < xhigh) & (array[yind] < ymax) & \
        (array[yind] > ymin)]

        if (len(subset) == 0):
            zvals[i] = 0.0

        else:
            if (countn):
                zvals[i] = len(subset)
            else:
                if (func == 'avg'):
                    zvals[i] = np.average(subset[zind])
                elif (func == 'std'):
                    zvals[i] = np.std(subset[zind])
                else:
                    zvals[i] = np.average(subset[zind])

    return xvals, zvals

################################################################
