#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = lambda: None

def set_plt_font(SMALL_SIZE=14, MEDIUM_SIZE=16, BIGGER_SIZE=18):
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# import seaborn as sns
# sns.set(context="paper", font="monospace")
def add_line_label(ax, yloc=None, is_color=False):
    # Add label of line at the end, instead of using legend
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    for i, l in enumerate(ax.lines):
        y_this = l.get_xydata()[-1, 1] if yloc is None else yloc[i]
        color = l.get_c() if is_color else 'black'
        ax.text(xmax, y_this, l.get_label(), horizontalalignment='left', verticalalignment='center', color=color)

def autolabel(rects, ax):
    '''Attach text labels to bar plots
    Args:
        rects: bar plot handke
        ax: axis
    Note:
        From: https://matplotlib.org/examples/api/barchart_demo.html
    '''

    if rects[0].get_x() == rects[1].get_x(): # axis from a barh
        min_loc = max(min(rect.get_width() for rect in rects), 0.1)
        for rect in rects:
            loc = rect.get_width()
            ax.text(loc + min_loc * 0.6, rect.get_y(),
                    '%.1f' % round(loc, 1),
                    ha='center', va='bottom')
    else: # axis from a bar
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.1f' % round(height, 1),
                    ha='center', va='bottom')

def ColorSchemes():
    """
        ColorSchemes() defines and register color schemes, which include
    RWWB: red-white-blue, with more space for white in the middle; good for prec anomalies
    BWWR: blue-white-red, with more space for white in the middle, and transition colors in between, good for temperature anomalies
    """

    ## Define colors
    cdict1 = {'red': ((0.0, 0.0, 1.0),
                  (0.33,1.0, 1.0),
                  (0.66, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                  (0.33, 1.0, 1.0),
                  (0.66, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                  (0.33, 1.0, 1.0),
                  (0.66, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        }


    cdict2 = {'red': ((0.0, 0.0, 0.0),
              (0.3, 0.0, 0.0),
              (0.375, 1.0, 1.0),
              (1.0, 1.0, 1.0)),

     'green': ((0.0, 0.0, 0.0),
               (0.3, 1.0, 1.0),
              (0.7, 1.0, 1.0),
              (1.0, 0.0, 0.0)),

     'blue':  ((0.0, 0.0, 1.0),
              (0.3, 1.0, 1.0),
              (0.6, 1.0, 1.0),
              (0.8, 0.0, 0.0),
              (1.0, 0.0, 0.0))
    }

    plt.register_cmap(name = 'RWWB', data = cdict1)
    plt.register_cmap(name = 'BWWR', data = cdict2)

def PlotColorMap(lon, lat, data, title = '', cScheme = 'bwr', cRange = None, nInt = 10, levels = None, coastColor = 'k'):
    """
    PlotColorMap(lon, lat, data, title = '', cScheme = 'bwr', cRange = None, nInt = None, levels = None) plots
    2D color maps in designated region. It allows for nonlinear color scale if levels are provided in such way
    lon, lat, data: longitude, latitude and data to plot
    title: title of figure
    cScheme: color scheme; Can use USDF ones like bwwr
    cRange: Range of colorbar
    nInt: number of color intervals
    levels: contour levels
    """
    lons, lats = np.meshgrid(lon, lat)

    # color ranges
    if not cRange:
        cRange = np.ceil(data.std() * 3)

    # contour levels: if not provided, using linearly discretized levels
    if levels is None:
        levels = np.linspace(-cRange, cRange , nInt + 1)

    # color schemes and levels
    ColorSchemes() # USDF color schemes from ClimPlot
    cmap = plt.cm.get_cmap(cScheme)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)


    # create Basemap instance.
    plt.subplots_adjust(hspace = 0.3)
    plt.subplots_adjust(wspace = 0.2)
    if lat.ptp() > 175 and lon.ptp() > 355: # If plotting global maps, use Robinson
        m = Basemap(projection = 'robin', lon_0 = 0, resolution = 'c')
        nLat = 30; nLon = 60
    else: # use cyl for regional map
        m = Basemap(resolution = 'c', projection = 'cyl',
                    llcrnrlon = lon[0], llcrnrlat = lat[0], urcrnrlon = lon[-1], urcrnrlat = lat[-1])
        nLat = 20; nLon = 30

    # plot data:
    #im1 = m.pcolormesh(lons, lats, data, cmap = cmap, shading = 'gouraud', latlon = True, vmin = -cRange, vmax = cRange, norm = norm, alpha = 0.8)
    #im1 = m.imshow(data, cmap = cmap, vmin = -cRange, vmax = cRange, norm = norm, alpha = 0.8, interpolation='spline16')
    im1 = m.contourf(lons, lats, data, cmap = cmap, levels = levels, norm = norm, alpha = 0.8, latlon = True)

    # draw parallels and meridians, labelling them; set linewidth to 0 to not show
    m.drawcoastlines(color = coastColor)
    m.drawparallels(np.arange(-90., 99., nLat), linewidth = 0.0, labels = [1, 0, 0, 0])
    m.drawmeridians(np.arange(-180., 189., nLon), linewidth = 0.0, labels = [0, 0, 0, 1])

    # add colorbar
    cb = m.colorbar(im1, "bottom", size = "5%", pad = "10%", ticks = levels)

    # add a title.
    PltTitle(title)


def PltTitle(title, tSize = 18, tWeight = 'normal', vPad = 0.05):
    """
    PltTitle(title, tSize = 18, tWeight = 'normal', vPad = 0.05) add a title with parameters different from the default
    tSize: font size
    tWegiht: font weight
    vPad: additional padding from the figure below
    """
    tlt = plt.title(title, fontdict = {'weight' : tWeight, 'size'   : tSize})
    tltPos = tlt.get_position()
    tlt.set_position((tltPos[0], tltPos[1] + vPad))
