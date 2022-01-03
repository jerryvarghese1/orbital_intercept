# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 13:42:34 2021

@author: vargh
"""
from astropy import units as u

from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit

import numpy as np

from scipy.integrate import odeint
import matplotlib.pyplot as plt

from gekko import GEKKO

plt.style.use('dark_background')

def eom(y, t, mu):
    
    r1, r2, r3, r1dot, r2dot, r3dot = y
    
    r = np.sqrt(r1**2 + r2**2 + r3**2)
    ydot = [r1dot, r2dot, r3dot, -mu*r1/r**3, -mu*r2/r**3, -mu*r3/r**3]
    return ydot

tof = 400000

max_u = .00008

arg_p = 270 * u.deg
i = 15.4 * u.deg
e = .8 * u.one
a = 26600 * u.km
RAAN = 45 * u.deg
theta = 170 * u.deg

arg_p_f = 10 * u.deg
i_f = 90.4 * u.deg
e_f = .0 * u.one
a_f = 50000 * u.km
RAAN_f = 70 * u.deg

theta_f = -80 * u.deg # starting true anomaly, not intercept true anomlay

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])

tmp_orbit = Orbit.from_classical(Earth, a, e, i, RAAN, arg_p, theta)
r0 = np.array([tmp_orbit.r[i].value for i in range(len(tmp_orbit.r))])
v0 = np.array([tmp_orbit.v[i].value for i in range(len(tmp_orbit.v))])
y0 = np.array(np.hstack([r0, v0])).transpose()

fin_orbit = Orbit.from_classical(Earth, a_f, e_f, i_f, RAAN_f, arg_p_f, theta_f)
rf_prop = np.array([fin_orbit.r[i].value for i in range(len(fin_orbit.r))])
vf_prop = np.array([fin_orbit.v[i].value for i in range(len(fin_orbit.v))])
yf_prop = np.array(np.hstack([rf_prop, vf_prop])).transpose()

t_theta = np.linspace(0, fin_orbit.period.to(u.s).value) * u.s

theta_data = np.array([fin_orbit.propagate(a_theta).nu.value for a_theta in t_theta])
ind = np.where(abs(theta_data - theta_f.to(u.rad).value) < 1e-1)[0][0]
orb_start = t_theta[ind]

theta_int = fin_orbit.propagate(orb_start + tof*u.s).nu

act_fin_orbit = Orbit.from_classical(Earth, a_f, e_f, i_f, RAAN_f, arg_p_f, theta_int)
rf = np.array([act_fin_orbit.r[i].value for i in range(len(act_fin_orbit.r))])
vf = np.array([act_fin_orbit.v[i].value for i in range(len(act_fin_orbit.v))])
yf = np.array(np.hstack([rf, vf])).transpose()

mu = Earth.k.to(u.km**3/u.s**2).value
r_e = Earth.R.to(u.km).value

t_list = np.linspace(0, tmp_orbit.period.to(u.s).value, 500)
s_transf = odeint(eom, y0, t_list, args=(mu,))

orb_x = s_transf[:,0]
orb_y = s_transf[:,1]
orb_z = s_transf[:,2]

ax.plot(orb_x, orb_y, orb_z, 'w', label = 'Initial',)

t_list2 = np.linspace(0, fin_orbit.period.to(u.s).value, 500)
s_transf2 = odeint(eom, yf_prop, t_list2, args=(mu,))

orb_x2 = s_transf2[:,0]
orb_y2 = s_transf2[:,1]
orb_z2 = s_transf2[:,2]

ax.plot(orb_x2, orb_y2, orb_z2, 'm', label = 'Final',)

m = GEKKO(remote=True)
m.time = np.linspace(0, tof, 100)

#tof.STATUS = 1
t = m.Param(value = m.time) 


r1 = m.Var(y0[0])
r2 = m.Var(y0[1])
r3 = m.Var(y0[2])

r1dot = m.Var(y0[3])
r2dot = m.Var(y0[4])
r3dot = m.Var(y0[5])

u1 = m.Var(lb = -max_u, ub = max_u)
u2 = m.Var(lb = -max_u, ub = max_u)
u3 = m.Var(lb = -max_u, ub = max_u)

#u1.MEAS = 0
#u1.STATUS = 1  # allow optimizer to change
#u1.DCOST = 0.1 # smooth out gas pedal movement
#u1.DMAX = 20   # slow down change of gas pedal

#u2.MEAS = 0
#u2.STATUS = 1  # allow optimizer to change
#u2.DCOST = 0.1 # smooth out gas pedal movement
#u2.DMAX = 20   # slow down change of gas pedal

#u3.MEAS = 0
#u3.STATUS = 1  # allow optimizer to change
#u3.DCOST = 0.1 # smooth out gas pedal movement
#u3.DMAX = 20   # slow down change of gas pedal

m.Equation(r1.dt() == r1dot)
m.Equation(r2.dt() == r2dot)
m.Equation(r3.dt() == r3dot)

r = m.Intermediate(m.sqrt(r1**2 + r2**2 + r3**2))
v = m.Intermediate(m.sqrt(r1dot**2 + r2dot**2 + r3dot**3))

m.Equation(-mu*r1/r**3 == r1dot.dt() + u1)
m.Equation(-mu*r2/r**3 == r2dot.dt() + u2)
m.Equation(-mu*r3/r**3 == r3dot.dt() + u3)

m.fix_final(r1, yf[0])
m.fix_final(r2, yf[1])
m.fix_final(r3, yf[2])

#m.fix_final(r1dot, yf[3])
#m.fix_final(r2dot, yf[4])
#m.fix_final(r3dot, yf[5])

#m.Minimize(tof)
m.Minimize(m.integral(u1**2 + u2**2 + u3**2))

m.options.IMODE = 6
m.options.solver = 3
#m.options.ATOL = 1e-3
m.options.MAX_ITER = 300
m.solve(disp=True)    # solve

orb_x = np.array(r1.value)
orb_y = np.array(r2.value)
orb_z = np.array(r3.value)

ax.plot(orb_x, orb_y, orb_z, label = 'transfer')

a, b = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = r_e*np.cos(a)*np.sin(b)
y = r_e*np.sin(a)*np.sin(b)
z = r_e*np.cos(b)
ax.plot_surface(x, y, z, color="b")

ax.legend()

max_range = np.array([orb_x.max()-orb_x.min(), orb_y.max()-orb_y.min(), orb_z.max()-orb_z.min()]).max() / 1.6

mid_x = (orb_x.max()+orb_x.min()) * 0.5
mid_y = (orb_y.max()+orb_y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)


