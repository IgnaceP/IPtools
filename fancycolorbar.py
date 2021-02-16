import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def fancyColorbar(ax, fig, im, label, color = 'black', labelsize = 8, orientation = 'horizontal'):
    """
    Method to plot a fancy colorbar on a figure

    ax: (Required) axes as reference
    fig: (Required) matplotlib figure
    im: (Required) imshow, contour or tripcolor object
    label: (Required) string with text to place on the label
    color: (Optional, defaults to 'black') color of the labels
    labelsize: (Optional, defaults to 8) size of the labels
    orientation: (Optional, defaults to 'horizontal') orientation string of the colorbar
    """

    divider = make_axes_locatable(ax)
    if orientation == 'horizontal': loc = 'bottom'
    else: loc = 'right'
    cax = divider.append_axes(loc, size="5%", pad=0.05)
    cb = fig.colorbar(im, orientation = orientation, cax = cax)
    if orientation == 'horizontal': loc = 'center'
    else: loc = 'right'
    cb.ax.set_title(label, size = labelsize+2, color = color, loc = loc)
    cb.ax.tick_params(labelsize= labelsize, color = color, labelcolor = color)
