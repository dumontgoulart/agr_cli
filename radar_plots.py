# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:05:12 2021

@author: morenodu
"""

# import numpy as np
# import matplotlib.pyplot as plt

# # Fixing random state for reproducibility
# def circle_clim(tmx,dtr,precip, scen_title):
#     print(scen_title)
#     # Compute pie slices
#     np.random.seed(19680801)
#     N = 3
#     theta = np.array([0.523599,5*0.523599,4.71239])
#     radii = np.array([tmx,dtr,precip])
#     width = np.array([2.0944,2.0944,2.0944])
    
#     cmap = plt.get_cmap("tab20c")
#     outer_colors = cmap(np.arange(3)*4)
    
#     ax = plt.subplot(111, projection='polar')
#     ax.bar(theta, radii, width=width, bottom=0.0, color = ['k','b','r'], alpha=0.9)
#     ax.set_rlabel_position(-135)
#     ax.set_ylim(0,2000)
#     ax.set_theta_zero_location('W')
#     # ax.grid(False)
#     # ax.spines['polar'].set_visible(True)
#     # # ax.set_rticks([])
#     ax.set_xticklabels([''])
#     ax.set_title(f'Univariate occurance of 2012 analogues for {scen_title}')
#     plt.tight_layout()
#     plt.show()

# circle_clim(tmx = 43, dtr = 3, precip = 89, scen_title = 'PD' )
# circle_clim(tmx = 503, dtr = 3, precip = 165, scen_title = '2C')
# circle_clim(tmx = 1346, dtr = 0, precip = 241, scen_title = '3C')





# fig, ax = plt.subplots()

# size = 1
# vals = np.array([[120., 120.], [120., 120.], [120., 120.]])

# cmap = plt.get_cmap("tab20c")
# outer_colors = cmap(np.arange(3)*4)
# inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

# ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
#        wedgeprops=dict(width=size, edgecolor='w'))

# ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
#        wedgeprops=dict(width=size, edgecolor='w'))

# ax.set(aspect="equal", title='Pie plot with `ax.pie`')
# plt.show()


#########################
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

#########################################
data = [['Tmx', 'Dtr', 'Precip'],
        ('Basecase', [[43, 4, 89],
            [503, 3, 165],
            [1346, 0, 241],])]

N = len(data[0])
theta = radar_factory(N, frame='circle')

spoke_labels = data.pop(0)
title, case_data = data[0]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)

ax.set_ylim(0,2000)
ax.set_rgrids([0, 500, 1000, 1500, 2000])
ax.set_title(title,  position=(0.5, 1.1), ha='center')
    
for d in case_data:
    line = ax.plot(theta, d)
    ax.fill(theta, d,  alpha=0.9)
ax.set_varlabels(spoke_labels)

plt.show()


def radar_plot_scen(tmx = 43, dtr = 3, precip = 89, scen_title = 'PD' ):
    data = [['Tmx', 'Dtr', 'Precip'],
            (f'Univariate occurance of 2012 analogues for {scen_title}', [[tmx, dtr, precip]])]     
    
    N = len(data[0])
    theta = radar_factory(N, frame='circle')
    
    spoke_labels = data.pop(0)
    title, case_data = data[0]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    
    ax.set_ylim(0,2000)
    ax.set_rgrids([0, 500, 1000, 1500, 2000])
    ax.set_title(title,  position=(0.5, 1.1), ha='center')
    
    for d in case_data:
        line = ax.plot(theta, d)
        ax.fill(theta, d,  alpha=0.9)
    ax.set_varlabels(spoke_labels)
    
    return fig
    

fig_radar_PD = radar_plot_scen(tmx = 43, dtr = 3, precip = 89, scen_title = 'PD' )
fig_radar_2C = radar_plot_scen(tmx = 503, dtr = 3, precip = 165, scen_title = '2C')
fig_radar_3C = radar_plot_scen(tmx = 1346, dtr = 0, precip = 241, scen_title = '3C')




# Figure with 3 plots - Seasons exceeding 2012 conditions ########################################################
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5), dpi=500, subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)
    
# Subplot 1 - PD
data = [['Tmx', 'Dtr', 'Precip'], (f'a) PD scenario', [[43, 3, 89]])]     
N = len(data[0])
theta = radar_factory(N, frame='circle')
spoke_labels = data.pop(0)
title, case_data = data[0]

ax1.set_ylim(0,2000)
ax1.set_rgrids([0, 500, 1000, 1500, 2000])
ax1.set_title(title,  position=(0.5, 1.1), ha='center')

for d in case_data:
    line = ax.plot(theta, d)
    ax1.fill(theta, d,  alpha=0.9)
ax1.set_varlabels(spoke_labels)

# Subplot 2 - 2C
data = [['Tmx', 'Dtr', 'Precip'], (f'b) 2C scenario', [[503, 3, 165]])]     
N = len(data[0])
theta = radar_factory(N, frame='circle')
spoke_labels = data.pop(0)
title, case_data = data[0]

ax2.set_ylim(0,2000)
ax2.set_rgrids([500, 1000, 1500, 2000])
ax2.set_title(title,  position=(0.5, 1.1), ha='center')

for d in case_data:
    line = ax.plot(theta, d)
    ax2.fill(theta, d,  alpha=0.9)
ax2.set_varlabels(spoke_labels)

# Subplot 3 - 3C
data = [['Tmx', 'Dtr', 'Precip'], (f'c) 3C scenario', [[1346, 0, 241]])]     
N = len(data[0])
theta = radar_factory(N, frame='circle')
spoke_labels = data.pop(0)
title, case_data = data[0]

ax3.set_ylim(0,2000)
ax3.set_rgrids([0, 500, 1000, 1500, 2000])
ax3.set_title(title,  position=(0.5, 1.1), ha='center')

for d in case_data:
    line = ax.plot(theta, d)
    ax3.fill(theta, d,  alpha=0.9)
ax3.set_varlabels(spoke_labels)

# plt.tight_layout()    
fig.savefig('paper_figures/radar_2012_2.png', format='png', dpi=500)

#######################################
