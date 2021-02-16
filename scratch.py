import numpy as np
from shapely.geometry import MultiPolygon, Polygon
import polplot as pp
import matplotlib.pyplot as plt
from pol2npy import *
from rasterize import *

def main():
    pol = Polygon([[0,0],[10,22],[12,24],[16,18],[20,17],[5,2]],[[[6,10],[10,18],[10,10]]])
    arr = rasterize(pol, res = 0.1)
