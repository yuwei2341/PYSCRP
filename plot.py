#!/usr/bin/env python
"""
Helpers for Plot 
"""
import matplotlib.pyplot as plt


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