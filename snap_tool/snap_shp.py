import argparse
from shp2mpol import shp2mpol
from mpol2shp import mpol2shp

"""
Script to apply the snap function from the command line
"""

# function for boolean ArgumentParser
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# construct argument parser
ap = argparse.ArgumentParser()

ap.add_argument('-i','--input', required = True, help= " The polygon (.shp) to snap to a reference polygon.")
ap.add_argument('-r','--reference', required = True, help = " The reference polygon to which the input polygon will be snapped.")
ap.add_argument('-pi', '--projection_input', required = False, help = " Projection of the input polygon: WGS84 or UTM.")
ap.add_argument('-pr', '--projection_ref', required = False, help = " Projection of the reference polygon: WGS84 or UTM.")
ap.add_argument('-u', '--utm_zone', required = False, help = " If projection is WGS84, indicate the UTM zone (example: '17M').")
ap.add_argument('-s', '--snap_sensitivity', required = True, help = " The maximum distance allowed for snapping.")
ap.add_argument('-o', '--output', required = False, help = " The output file directory.")
ap.add_argument('-pp','--print_progress',  type=str2bool, default = True, help = " Print the progress to the console.")

args = vars(ap.parse_args())

inputfile = args['input']
reffile = args['reference']
if args['projection_input']: proj_input = args['projection_input']
else: proj_input = 'UTM'
if args['projection_ref']: proj_ref = args['projection_ref']
else: proj_ref = 'UTM'
snapsens = args['snap_sensitivity']
utm = args['utm_zone']
of = args['output']
pp = args['print_progress']

if proj_ref != proj_input: print('Warning! Projection of input and reference polygon is different. This will likely cause a bad snapping!')

# transform to UTM coordinates if needed. Note that combining UTM and WGS84 will probably cause problems
if proj_input == 'WGS84':
    pol2snap = shp2mpol(inputfile, project_to_UTM = utm)
else:
    pol2snap = shp2mol(inputfile)

if proj_ref == 'WGS84':
    pol2snap2 = shp2mpol(reffile, project_to_UTM = utm)
else:
    pol2snap2 = shp2mol(reffile)

# snap the polygon
pol_snapped = snapPol2Pol(pol2plot, pol2plot2, Snap_Sensitivity = snapsens, print_progressbar = pp)

# export the polygon as shp
if not of:
    if of[-4:] != '.shp': of = of + '.shp'
    mpol2shp(pol_snapped, of)
else:
    mpol2shp(pol_snapped, inputfile+'_snapped.shp')
