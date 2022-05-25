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
import matplotlib as mpl

from gekko import GEKKO

import pandas as pd

plt.style.use('dark_background')

def eom(y, t, mu):
    r1, r2, r3, r1dot, r2dot, r3dot = y

    r = np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    ydot = [r1dot, r2dot, r3dot, -mu * r1 / r ** 3, -mu * r2 / r ** 3, -mu * r3 / r ** 3]
    return ydot

def plot_orbit(ax, t_list, r, v, mu):
    y0 = np.hstack([r, v])
    s_transf = odeint(eom, y0, t_list, args=(mu,))

    orb_x = s_transf[:, 0]
    orb_y = s_transf[:, 1]
    orb_z = s_transf[:, 2]

    ax.plot(orb_x, orb_y, orb_z, 'w')

def fix_graph_bound(ax, orb_x, orb_y, orb_z):
    max_range = np.array([orb_x.max() - orb_x.min(), orb_y.max() - orb_y.min(), orb_z.max() - orb_z.min()]).max() / 1.6

    mid_x = (orb_x.max() + orb_x.min()) * 0.5
    mid_y = (orb_y.max() + orb_y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

class boundaries:
    def __init__(self, cent_body, arg_p, i, e, a, RAAN, theta):
        self.arg_p = arg_p
        self.i = i
        self. e = e
        self.a = a
        self.RAAN = RAAN

        self.theta = np.linspace(theta, theta + 360, 300)*u.deg

        self.orbit = Orbit.from_classical(Earth, a, e, i, RAAN, arg_p, theta*u.deg)

        self.r = np.array([Orbit.from_classical(Earth, a, e, i, RAAN, arg_p, theta).r for theta in self.theta])
        self.v = np.array([Orbit.from_classical(Earth, a, e, i, RAAN, arg_p, theta).v for theta in self.theta])


tof = 500000
t_steps = 2000
max_u = .0001
mu = Earth.k.to(u.km**3/u.s**2).value

arg_p = 270 * u.deg
i = 15.4 * u.deg
e = .5 * u.one
a = 26600 * u.km
RAAN = 45 * u.deg

arg_p_f = 200 * u.deg
i_f = 25.4 * u.deg
e_f = .2 * u.one
a_f = 40000 * u.km
RAAN_f = 30 * u.deg

initial_o = boundaries(Earth, arg_p, i, e, a, RAAN, 0)
final_o = boundaries(Earth, arg_p_f, i_f, e_f, a_f, RAAN_f, 45)

t_list = np.linspace(0, tof, t_steps)

m = GEKKO(remote=True)
m.time = t_list

final = np.zeros(len(m.time))
final[-1] = 1
final = m.Param(value=final)

r1 = m.Var(initial_o.r[0, 0])
r2 = m.Var(initial_o.r[0, 1])
r3 = m.Var(initial_o.r[0, 2])

r1dot = m.Var(initial_o.v[0, 0])
r2dot = m.Var(initial_o.v[0, 1])
r3dot = m.Var(initial_o.v[0, 2])

u1 = m.Var(lb = -max_u, ub = max_u)
u2 = m.Var(lb = -max_u, ub = max_u)
u3 = m.Var(lb = -max_u, ub = max_u)

m.Equation(r1.dt() == r1dot)
m.Equation(r2.dt() == r2dot)
m.Equation(r3.dt() == r3dot)

r = m.Intermediate(m.sqrt(r1**2 + r2**2 + r3**2))
v = m.Intermediate(m.sqrt(r1dot**2 + r2dot**2 + r3dot**3))

m.Equation(-mu*r1/r**3 == r1dot.dt() + u1)
m.Equation(-mu*r2/r**3 == r2dot.dt() + u2)
m.Equation(-mu*r3/r**3 == r3dot.dt() + u3)

#m.fix_final(r1, final_o.r[0, 0])
#m.fix_final(r2, final_o.r[0, 1])
#m.fix_final(r3, final_o.r[0, 2])

m.Obj(final*(r1-final_o.r[0, 0])**2)
m.Obj(final*(r2-final_o.r[0, 1])**2)
m.Obj(final*(r3-final_o.r[0, 2])**2)

m.Obj(final*(r1dot-final_o.v[0, 0])**2)
m.Obj(final*(r2dot-final_o.v[0, 1])**2)
m.Obj(final*(r3dot-final_o.v[0, 2])**2)

m.Minimize(m.integral(u1**2 + u2**2 + u3**2))

m.options.IMODE = 9
m.options.solver = 3
m.options.OTOL = 1e-9
m.options.MAX_ITER = 300
m.solve(disp=True)    # solve



orb_x = np.array(r1.value)
orb_y = np.array(r2.value)
orb_z = np.array(r3.value)

mpl.use('Qt5Agg')
plt.style.use('dark_background')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])

ax.plot(orb_x, orb_y, orb_z, label = 'transfer')

r_e = Earth.R.to_value(u.km)
a, b = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = r_e*np.cos(a)*np.sin(b)
y = r_e*np.sin(a)*np.sin(b)
z = r_e*np.cos(b)
ax.plot_surface(x, y, z, color="b")

plot_orbit(ax, t_list, initial_o.r[0, :], initial_o.v[0, :], mu)
plot_orbit(ax, t_list, final_o.r[0, :], final_o.v[0, :], mu)

fix_graph_bound(ax, orb_x, orb_y, orb_z)

plt.figure()
plt.plot(t_list, u1.value)
plt.plot(t_list, u2.value)
plt.plot(t_list, u3.value)

plt.show()

