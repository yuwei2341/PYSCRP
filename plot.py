#!/usr/bin/env python

import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="paper", font="monospace")

def autolabel(rects, ax):
    # attach some text labels
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